from typing import Optional

import unittest
from copy import deepcopy
from warnings import catch_warnings

import torch
from torch import nn
from torch.fx import symbolic_trace
from torchvision.models.resnet import resnet18, resnet34
from torchvision.models.segmentation.fcn import fcn_resnet50

from nn_lib.models.graph_module_plus import GraphModulePlus
from nn_lib.models.graph_utils import prefix_all_nodes
from nn_lib.utils.models import frozen


class ModuleTestCase(unittest.TestCase):
    def assertModulesEqual(self, modA: nn.Module, modB: nn.Module):
        """Helper function to assert that two modules are identical."""
        self.assertSequenceEqual(
            [k for k, _ in modA.named_parameters()],
            [k for k, _ in modB.named_parameters()],
        )
        self.assertSequenceEqual(
            [k for k, _ in modA.named_modules()],
            [k for k, _ in modB.named_modules()],
        )

        stateA = modA.state_dict()
        stateB = modB.state_dict()
        for keyA, paramA in stateA.items():
            self.assertIn(keyA, stateB, f"Key {keyA} not found in stateB")
            paramB = stateB[keyA]
            self.assertTrue(
                torch.equal(paramA, paramB),
                f"Parameters for key {keyA} are not equal",
            )


class FakeStitchingLayerForTesting(nn.Module):
    def __init__(self, in_f, out_f):
        super().__init__()
        self.conv = nn.Conv2d(in_f, out_f, kernel_size=1, padding=0, stride=1)

    def forward(self, x):
        return self.conv(x)

    def update_weight(self, mul):
        self.conv.weight.data *= mul


class DummyModuleWithMethodsAndAssertions(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor):
        # This forward() function is designed to test symbolic tracing of method calls. The first
        # is a method call on a tensor. The second is a function call on a package.

        # Call a method on a tensor
        sz = x.dim()

        # Make a traceable assertion
        torch._assert(sz == 2, "requires 2D tensor input")

        # Call a method of torch
        x = torch.permute(x, (1, 0))

        return x


class DummyModuleWithLongSkipConnection(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(2, 4)
        self.fc2 = nn.Linear(4, 6)
        self.fc3 = nn.Linear(6, 8)
        self.fc4 = nn.Linear(8, 10)
        self.skipper = nn.Linear(2, 10)

    def forward(self, x):
        out1 = self.fc1(x)
        out2 = self.fc2(out1)
        out3 = self.fc3(out2)
        out4 = self.fc4(out3)
        return out4 + self.skipper(x)


class DummyModuleWithDictOutput(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = nn.Linear(2, 10)

    def forward(self, x):
        out1 = self.linear1(x)
        return {"out": out1}

class ShapeCheckedNet(nn.Module):
    """A net that validates its input, like a vision transformer does.

    The assertion consumes x and produces nothing, so the branch from x to the
    assert has users but no path to the output.
    """

    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(8, 8)
        self.fc2 = nn.Linear(8, 4)

    def forward(self, x):
        torch._assert(x.shape[1] == 8, "expected width 8")
        return self.fc2(self.fc1(x) + x)

class OptionalArgsNet(nn.Module):
    """A net whose forward takes optional arguments, as timm transformers do.

    Traced, this records three placeholders whose defaults disagree, which is
    what the shared-input merge used to refuse. The optional arguments are used
    arithmetically so that they survive as real inputs, and their defaults are
    the identity so that calling with the image alone is unchanged.
    """

    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(8, 8)

    def forward(self, x: torch.Tensor, bias: float = 0.0, scale: float = 1.0):
        return self.fc(x) * scale + bias


class TestGraphModulePlus(ModuleTestCase):

    def setUp(self):
        # Setup code to initialize a GraphModulePlus instance and any necessary nodes
        self.reference_module = resnet18()
        self.gm = GraphModulePlus.new_from_trace(self.reference_module)
        self.dummy_input = torch.randn(1, 3, 224, 224)

    def _get_dummy_rep(self, node_name):
        return GraphModulePlus.new_from_copy(self.gm).set_output(node_name)(self.dummy_input)

    def test_superseded(self):
        # Thanks to the @supersedes decorator, even built-in fx code that would normally return a
        # fx.GraphModule should return a GraphModulePlus instead
        gm = symbolic_trace(self.reference_module)
        self.assertIsInstance(gm, GraphModulePlus)

    def test_inputs(self):
        inputs = self.gm.inputs
        self.assertEqual(len(inputs), 1)
        self.assertEqual(inputs[0].name, "x")

    def test_output(self):
        self.assertEqual(self.gm.output.name, "output")
        self.assertEqual(self.gm.output_value.name, "fc")

    def test_set_inputs(self):
        # With eliminate_dead, the method should just work without warnings
        og_rep = self._get_dummy_rep("maxpool")
        og_output = self.gm(self.dummy_input)
        with catch_warnings(record=True) as w:
            self.gm.set_inputs(["maxpool"])
        print("Warnings:", *w, sep="\n")
        self.assertEqual(len(w), 0)
        inputs = self.gm.inputs
        self.assertEqual(len(inputs), 1)
        self.assertEqual(inputs[0].name, "maxpool")
        new_output = self.gm(og_rep)
        torch.testing.assert_close(og_output, new_output)

    def test_set_output(self):
        og_result = self._get_dummy_rep("layer1_1_relu")
        num_nodes_before = len(list(self.gm.graph.nodes))
        the_node = self.gm._resolve_nodes("layer1_1_relu")[0]
        self.gm.set_output("layer1_1_relu")
        self.assertEqual(self.gm.output_value, the_node)

        num_nodes_after = len(list(self.gm.graph.nodes))
        self.assertLess(num_nodes_after, num_nodes_before)

        new_result = self.gm(self.dummy_input)
        torch.testing.assert_close(og_result, new_result)

    def test_set_dict_outputs(self):
        self.gm.set_dict_outputs(outputs=["add_1", "add_2", "add_3"])
        out = self.gm(self.dummy_input)
        self.assertTrue(isinstance(out, dict))
        self.assertEqual(len(out), 3)
        self.assertIn("add_1", out)
        self.assertIn("add_2", out)
        self.assertIn("add_3", out)
        self.assertTrue(isinstance(out["add_1"], torch.Tensor))
        self.assertTrue(isinstance(out["add_2"], torch.Tensor))
        self.assertTrue(isinstance(out["add_3"], torch.Tensor))

    def test_set_inputs_and_output_noop(self):
        num_nodes_before = len(list(self.gm.graph.nodes))
        dummy_output_before = self.gm(self.dummy_input)

        self.gm.set_inputs_and_output(self.gm.inputs, self.gm.output_value)

        num_nodes_after = len(list(self.gm.graph.nodes))
        dummy_output_after = self.gm(self.dummy_input)

        self.assertEqual(num_nodes_before, num_nodes_after)
        self.assertTrue(torch.allclose(dummy_output_before, dummy_output_after))

    def test_buffers(self):
        gm_buffers = dict(self.gm.named_buffers())
        og_buffers = dict(self.reference_module.named_buffers())
        self.assertEqual(set(gm_buffers.keys()), set(og_buffers.keys()))
        for k in gm_buffers:
            self.assertTrue(torch.equal(gm_buffers[k], og_buffers[k]), f"Buffer {k} differs")

    def test_squash_conv_bn(self):
        self.gm.eval()
        output_before = self.gm(self.dummy_input)
        new_gm = self.gm.squash_all_conv_batchnorm_pairs()
        output_after = new_gm(self.dummy_input)

        def is_softmax_close(a, b):
            return torch.allclose(torch.softmax(a, dim=-1), torch.softmax(b, dim=-1), atol=1e-6)

        self.assertTrue(is_softmax_close(output_before, output_after))

        # The thing about batchnorm is that in 'training' mode, it updates itself on any forward
        # call. So we can check it all BN were eliminated by verifying that outputs are not changing
        # during train mode.
        new_gm.train()
        new_gm(self.dummy_input)
        output_after_2 = new_gm(self.dummy_input)
        self.assertTrue(is_softmax_close(output_after, output_after_2))

    def test_freeze_subgraph(self):
        params_before = {k: v.clone() for k, v in self.gm.named_parameters()}
        with frozen(self.gm.extract_subgraph(inputs=["add_1"], output="add_5")):
            opt = torch.optim.SGD(self.gm.parameters(), lr=0.1)
            for _ in range(10):
                opt.zero_grad()
                self.gm(self.dummy_input).sum().backward()
                opt.step()

        params_after = {k: v.clone() for k, v in self.gm.named_parameters()}

        for k in params_before:
            # We chose 'add_1' and 'add_5' above because they bracket the layer2 and layer3 parts
            # of the resnet18 model. So we expect the parameters in those blocks to be frozen.
            if k.startswith("layer2") or k.startswith("layer3"):
                self.assertTrue(
                    torch.allclose(params_before[k], params_after[k]),
                    f"Expected {k} to be frozen",
                )
            else:
                self.assertFalse(
                    torch.allclose(params_before[k], params_after[k]),
                    f"Expected {k} to be updated",
                )

        # Outside the with context, train again and assert that everything changed
        opt = torch.optim.SGD(self.gm.parameters(), lr=0.1)
        for _ in range(10):
            opt.zero_grad()
            self.gm(self.dummy_input).sum().backward()
            opt.step()

        params_after_after = {k: v.clone() for k, v in self.gm.named_parameters()}

        for k in params_after:
            self.assertFalse(
                torch.allclose(params_after[k], params_after_after[k]),
                f"Expected {k} to be updated",
            )

    def test_merge_graphmodules_leaves_originals_unchanged(self):
        """Merging to create a new GraphModulePlus should not modify the original models. Since
        new_from_merge contains cases for model type, this test isolates the case where the models
        are themselves instances of GraphModulePlus.
        """
        modelA = GraphModulePlus.new_from_trace(resnet18())
        modelB = GraphModulePlus.new_from_trace(resnet34())

        # Make a deepcopy snapshot of all the attrs of modelA and modelB so we can assert later
        # that merging did not modify the original models
        modelA_copy, modelB_copy = deepcopy(modelA), deepcopy(modelB)

        _ = GraphModulePlus.new_from_merge(
            modules={"modelA": modelA, "modelB": modelB},
            rewire_inputs={"modelB_layer2_1_relu_1": "modelA_add_3"},
            auto_trace=True,
        )

        self.assertModulesEqual(modelA_copy, modelA)
        self.assertModulesEqual(modelB_copy, modelB)

    def test_merge_modules_leaves_originals_unchanged(self):
        """Same as the previous test, but now with tracing some standard nn.Module models."""
        modelA = resnet18()
        modelB = resnet34()

        # Make a deepcopy snapshot of all the attrs of modelA and modelB so we can assert later
        # that merging did not modify the original models
        modelA_copy, modelB_copy = deepcopy(modelA), deepcopy(modelB)

        _ = GraphModulePlus.new_from_merge(
            modules={"modelA": modelA, "modelB": modelB},
            rewire_inputs={"modelB_layer2_1_relu_1": "modelA_add_3"},
            auto_trace=True,
        )

        self.assertModulesEqual(modelA_copy, modelA)
        self.assertModulesEqual(modelB_copy, modelB)

    def test_merge_two_modules_auto_trace(self):
        modelA, modelB = resnet18(), resnet34()
        merged_model = GraphModulePlus.new_from_merge(
            modules={"modelA": modelA, "modelB": modelB},
            rewire_inputs={"modelB_layer2_1_relu_1": "modelA_add_3"},
            auto_trace=True,
        )

        self.assertEqual(merged_model.__class__.__name__, "MergedResNetResNet")

        # Make sure we can still run the model and get outputs without crashing
        merged_model(self.dummy_input)

        # With auto_trace, we expect the merged model to have lots of nodes (copying ops from
        # modelA and from modelB).
        self.assertGreater(len(merged_model.graph.nodes), 100)

        # Nodes downstream of the rewire in A or upstream in B should no longer exist
        with self.assertRaises(ValueError):
            merged_model._resolve_nodes("modelA_add_4")

        with self.assertRaises(ValueError):
            merged_model._resolve_nodes("modelB_add_1")

    def test_merge_three_modules_stitching(self):
        modelA = GraphModulePlus.new_from_trace(resnet18())
        modelB = GraphModulePlus.new_from_trace(resnet34())
        targetB = modelB.insert_noop("add_3")
        stitcher = FakeStitchingLayerForTesting(128, 128)
        merged_model = GraphModulePlus.new_from_merge(
            modules={"modelA": modelA, "stitching_layer": stitcher, "modelB": modelB},
            rewire_inputs={
                "stitching_layer": "modelA_add_2",
                f"modelB_{targetB}": "stitching_layer",
            },
            auto_trace=False,
        )

        self.assertEqual(
            merged_model.__class__.__name__, "MergedResNetFakeStitchingLayerForTestingResNet"
        )

        # Nodes downstream of the rewire in A or upstream in B should no longer exist
        with self.assertRaises(ValueError):
            merged_model._resolve_nodes("modelA_add_4")

        with self.assertRaises(ValueError):
            merged_model._resolve_nodes("modelB_add_1")

        # We should still have access to the stitching layer's methods, and it should affect the
        # merged model's behavior because the underlying parameters are shared.
        out1 = merged_model(self.dummy_input)
        merged_model.stitching_layer.update_weight(0.5)
        out2 = merged_model(self.dummy_input)

        self.assertFalse(torch.allclose(out1, out2))

    def test_merge_three_modules_stitching_auto_trace(self):
        """Same as the previous test, but now with auto_trace=True. It's expected behavior that
        we can no longer access the update_weight method of the *traced* stitching layer. A
        to-do for someday is to make it so that we can still access the methods of the original
        modules even when tracing.
        """
        modelA = GraphModulePlus.new_from_trace(resnet18())
        modelB = GraphModulePlus.new_from_trace(resnet34())
        stitcher = FakeStitchingLayerForTesting(512, 128)
        merged_model = GraphModulePlus.new_from_merge(
            modules={"modelA": modelA, "stitching_layer": stitcher, "modelB": modelB},
            rewire_inputs={
                "stitching_layer_conv": "modelA_add_2",
                "modelB_layer2_1_relu_1": "stitching_layer_conv",
            },
            auto_trace=True,
        )

        self.assertEqual(
            merged_model.__class__.__name__, "MergedResNetFakeStitchingLayerForTestingResNet"
        )

        with self.assertRaises(AttributeError):
            merged_model.stitching_layer.update_weight(0.5)

    def test_merge_manages_skip_connections_merge_inputs(self):
        # Inspired by the challenge of stitching an ImageNet-trained (classification) model into
        # a COCO-trained (segmentation) model. The COCO models often involve some input-dependent
        # ops that affect later layers, e.g. reshaping the output. This breaks our typical stitching
        # assumption that models can be split at the given layer(s) in a way that cleanly separates
        # inputs from outputs.
        modelA = nn.Sequential(
            nn.Linear(2, 4),  # Gets node name 'modelA__0'
            nn.Linear(4, 6),  # Gets node name 'modelA__1'
            nn.Linear(6, 8),  # Gets node name 'modelA__2'
            nn.Linear(8, 10),  # Gets node name 'modelA__3'
        )
        modelB = DummyModuleWithLongSkipConnection()
        merged_model = GraphModulePlus.new_from_merge(
            modules={"modelA": modelA, "modelB": modelB},
            rewire_inputs={"modelB_fc3": "modelA__1"},
            auto_trace=True,
            share_inputs=True,
        )
        # The merged model should run without crashing, and the skip connection in modelB should be
        # properly rewired to the new input from modelA.
        dummy_input = torch.randn(3, 2)
        out = merged_model(dummy_input)
        self.assertEqual(out.shape, (3, 10), f"Expected output shape (3, 10), got {out.shape}")

    def test_merge_manages_skip_connections_separate_inputs(self):
        # Same as test_merge_manages_skip_connections_merge_inputs, but where modelA expects a
        # different input size than modelB and thus requires its own separate input.
        modelA = nn.Sequential(
            nn.Linear(5, 4),  # Gets node name 'modelA__0'
            nn.Linear(4, 6),  # Gets node name 'modelA__1'
            nn.Linear(6, 8),  # Gets node name 'modelA__2'
            nn.Linear(8, 10),  # Gets node name 'modelA__3'
        )
        modelB = DummyModuleWithLongSkipConnection()

        merged_model = GraphModulePlus.new_from_merge(
            modules={"modelA": modelA, "modelB": modelB},
            rewire_inputs={"modelB_fc3": "modelA__1"},
            auto_trace=True,
            share_inputs=False,
        )

        # The merged model *expects* two inputs; calling it with just one should raise an error
        with self.assertRaises(TypeError):
            dummy_input = torch.randn(3, 2)
            _ = merged_model(dummy_input)

        with self.assertRaises(TypeError):
            dummy_input = torch.randn(3, 5)
            _ = merged_model(dummy_input)

        # It should work with (modelA_x, modelB_x) inputs positionally
        dummy_inputs = torch.randn(3, 5), torch.randn(3, 2)
        out = merged_model(*dummy_inputs)
        self.assertEqual(out.shape, (3, 10), f"Expected output shape (3, 10), got {out.shape}")

    def test_merge_manages_skip_connections_resnet(self):
        # Inspired by the challenge of stitching an ImageNet-trained (classification) model into
        # a COCO-trained (segmentation) model. The COCO models often involve some input-dependent
        # ops that affect later layers, e.g. reshaping the output. This breaks our typical stitching
        # assumption that models can be split at the given layer(s) in a way that cleanly separates
        # inputs from outputs.
        modelA = GraphModulePlus.new_from_trace(resnet18())
        modelB = GraphModulePlus.new_from_trace(fcn_resnet50()).set_output("interpolate")
        stitcher = FakeStitchingLayerForTesting(128, 128)
        # in resnet18, 'layer2_1_relu_1' has shape (128, 28, 28)
        # in fcn_resnet50, 'layer2_1_relu_1' also happens to have shape (128, 28, 28)
        merged_model = GraphModulePlus.new_from_merge(
            modules={"modelA": modelA, "stitching_layer": stitcher, "modelB": modelB},
            rewire_inputs={
                "stitching_layer_conv": "modelA_layer2_1_relu_1",
                "modelB_backbone_layer2_1_relu_1": "stitching_layer_conv",
            },
            auto_trace=True,
        )

        dummy_input = torch.randn(1, 3, 128, 128)
        out = merged_model(dummy_input)
        self.assertEqual(
            out.shape,
            (1, 21, 128, 128),
            f"Expected output shape (1, 21, 128, 128), got {out.shape}",
        )

    def test_prefix_leaves_original_unchanged(self):
        """Assert that prefix_all_nodes treats its inputs as immutable."""
        my_gm = GraphModulePlus.new_from_trace(resnet18())
        gm_nodes_before = set(map(str, my_gm.graph.nodes))
        new_graph = prefix_all_nodes(my_gm.graph, prefix="pre")
        new_nodes_after = set(map(str, new_graph.nodes))
        gm_nodes_after = set(map(str, my_gm.graph.nodes))

        self.assertNotEqual(gm_nodes_before, new_nodes_after, "Node names should all be different")
        self.assertEqual(gm_nodes_before, gm_nodes_after, "Original graph should not be modified")

    def test_prefix_simple(self):
        new_graph = prefix_all_nodes(self.gm.graph, prefix="pre")
        new_gm = GraphModulePlus(root=nn.ModuleDict({"pre": self.gm}), graph=new_graph)

        self.assertTrue(hasattr(new_gm, "pre"))
        self.assertTrue(hasattr(self.gm, "conv1"))
        self.assertTrue(hasattr(new_gm.pre, "conv1"))

        og_out = self.gm(self.dummy_input)
        new_out = new_gm(self.dummy_input)
        torch.testing.assert_close(og_out, new_out)

    def test_prefix_dict_output(self):
        gm = GraphModulePlus.new_from_trace(DummyModuleWithDictOutput())
        new_graph = prefix_all_nodes(gm.graph, prefix="pre")
        new_gm = GraphModulePlus(root=nn.ModuleDict({"pre": gm}), graph=new_graph)

        dummy_input = torch.randn(4, 2)
        og_out = gm(dummy_input)
        new_out = new_gm(dummy_input)
        torch.testing.assert_close(og_out, new_out)

    def test_prefix_call_module(self):
        dummy_gm = GraphModulePlus.new_from_trace(DummyModuleWithMethodsAndAssertions())
        new_graph = prefix_all_nodes(dummy_gm.graph, prefix="pre")
        new_gm = GraphModulePlus(root=nn.ModuleDict({"pre": dummy_gm}), graph=new_graph)

        dummy_input = torch.ones(4, 3)
        og_out = dummy_gm(dummy_input)
        new_out = new_gm(dummy_input)
        torch.testing.assert_close(og_out, new_out)

    def test_copy_is_not_deep(self):
        param_before = next(iter(self.gm.parameters()))
        param_before.data[:] = 1.0
        torch.testing.assert_close(param_before.data, torch.ones_like(param_before.data))

        gm2 = GraphModulePlus.new_from_copy(self.gm)
        param2 = next(iter(gm2.parameters()))
        param2.data[:] = 2.0

        torch.testing.assert_close(param_before.data, torch.ones_like(param2.data) * 2)

    def test_copy_does_not_delete(self):
        og_num_nodes = len(list(self.gm.graph.nodes))
        og_out = self.gm(self.dummy_input)
        truncated_copy = GraphModulePlus.new_from_copy(self.gm).set_output("add_3")

        # The copy should have fewer nodes...
        self.assertLess(len(list(truncated_copy.graph.nodes)), og_num_nodes)

        # ...but should not have modified the original
        self.assertEqual(len(list(self.gm.graph.nodes)), og_num_nodes)

        # The copy should still be able to run
        trunc_out = truncated_copy(self.dummy_input)
        self.assertEqual(trunc_out.ndim, 4)

        # The og model should still be able to run
        torch.testing.assert_close(og_out.detach(), self.gm(self.dummy_input).detach())

    def test_strip_assert(self):
        good_input = torch.ones(4, 3)
        bad_input = torch.ones(5, 4, 3)
        dummy_gm = GraphModulePlus.new_from_trace(DummyModuleWithMethodsAndAssertions())

        dummy_gm(good_input)
        with self.assertRaises(AssertionError):
            dummy_gm(bad_input)

        dummy_gm.strip_all_where(lambda node: node.is_impure and "assert" in node.name)

        dummy_gm(good_input)
        with self.assertRaises(RuntimeError):
            dummy_gm(bad_input)

    def test_delta_state_dict(self):
        copy_gm = deepcopy(self.gm)
        copy_gm.layer1._modules["0"].conv1.weight.data = torch.ones_like(
            copy_gm.layer1._modules["0"].conv1.weight
        )

        delta = copy_gm.delta_state_dict(self.gm)
        self.assertEqual(len(delta), 1)

        new_resnet = GraphModulePlus.new_from_trace(resnet18())
        new_resnet.load_delta_state_dict(delta, self.gm)

        assert torch.equal(
            new_resnet.layer1._modules["0"].conv2.weight, self.gm.layer1._modules["0"].conv2.weight
        )
        assert torch.equal(
            new_resnet.layer1._modules["0"].conv1.weight,
            torch.ones_like(copy_gm.layer1._modules["0"].conv1.weight),
        )

    def test_insert_noop(self):
        num_nodes1 = len(list(self.gm.graph.nodes))
        out1 = self.gm(self.dummy_input)
        noop_node = self.gm.insert_noop("add_4")
        num_nodes2 = len(list(self.gm.graph.nodes))
        out2 = self.gm(self.dummy_input)
        torch.testing.assert_close(out1, out2)
        self.assertEqual(num_nodes1 + 1, num_nodes2)
        self.assertIn(noop_node, self.gm.graph.nodes)
        self.assertEqual(noop_node.args[0].name, "add_4")
        self.assertGreater(len(list(noop_node.users)), 0)
    
    def test_extract_subgraph_drops_unreachable_inputs(self):
        """A half cut from the middle should take only the activation.

        A traced model may compute something it never uses, such as a shape
        check. That branch keeps the original input alive even after the input
        has been moved to a mid-graph node, leaving the extracted half with a
        parameter it does not need.
        """
        whole = GraphModulePlus.new_from_trace(ShapeCheckedNet().eval()).eval()
        cut = "add"

        self.assertIn(cut, [node.name for node in whole.graph.nodes])

        upper = GraphModulePlus.new_from_copy(whole).extract_subgraph(output=cut).eval()
        lower = GraphModulePlus.new_from_copy(whole).extract_subgraph(inputs=[cut]).eval()

        self.assertEqual([node.name for node in lower.graph.nodes if node.op == "placeholder"], [cut])

        x = torch.randn(2, 8)
        with torch.no_grad():
            self.assertTrue(torch.allclose(whole(x), lower(upper(x)), atol=1e-5))
    
    def test_merge_keeps_inputs_that_cannot_be_shared(self):
        """Merging a model whose forward takes optional arguments should succeed.

        Placeholders that came from the same argument are one input and merge.
        Placeholders from different arguments are not, and are carried through
        as separate inputs rather than refused.
        """
        net = OptionalArgsNet().eval()
        merged = GraphModulePlus.new_from_merge({"a": net}, rewire_inputs={}).eval()

        names = [node.name for node in merged.graph.nodes if node.op == "placeholder"]
        self.assertEqual(names, ["x", "bias", "scale"])

        x = torch.randn(2, 8)
        with torch.no_grad():
            self.assertTrue(torch.allclose(net(x), merged(x), atol=1e-5))

    def test_merge_of_single_input_models_is_unchanged(self):
        """A model with one argument still merges to a single input named x."""
        net = nn.Sequential(nn.Linear(8, 8), nn.ReLU()).eval()
        merged = GraphModulePlus.new_from_merge({"a": net}, rewire_inputs={}).eval()

        names = [node.name for node in merged.graph.nodes if node.op == "placeholder"]
        self.assertEqual(names, ["x"])

        x = torch.randn(2, 8)
        with torch.no_grad():
            self.assertTrue(torch.allclose(net(x), merged(x), atol=1e-5))
    
    def test_extract_subgraph_orders_inputs_so_the_forward_compiles(self):
        """A new input must be placed ahead of any parameter that has a default.

        extract_subgraph turns a mid-graph node into a placeholder, inserting it
        where that node sat, which is after the model's own placeholders. When
        the model's forward takes optional arguments, the regenerated forward
        then has a parameter without a default following ones that have
        defaults, which is a syntax error.
        """
        net = OptionalArgsNet().eval()
        gm = GraphModulePlus.new_from_trace(net).eval()
        cut = "mul"

        self.assertIn(cut, [node.name for node in gm.graph.nodes])

        lower = GraphModulePlus.new_from_copy(gm).extract_subgraph(inputs=[cut]).eval()

        names = [node.name for node in lower.graph.nodes if node.op == "placeholder"]
        self.assertEqual(names[0], cut)


if __name__ == "__main__":
    unittest.main()
