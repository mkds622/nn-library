from typing import Iterable, List, Optional, Any, assert_never, Self, Callable
import inspect
import pydot
import torch
from torch import nn
from torch.fx import symbolic_trace, GraphModule, Graph, Node

from nn_lib.models.graph_utils import prefix_all_nodes, validate_common_placeholder_attrs
from nn_lib.utils import supersedes


@supersedes(GraphModule)
class GraphModulePlus(GraphModule):
    """An extension of torch.fx.GraphModule that provides additional functionality.

    WARNING: We globally inject this class into existing torch.fx code by replacing the
    GraphModule.__new__ class method with the @supersedes decorator. This is a hacky way to
    extend the functionality of torch.fx globally, such that existing methods in torch.fx which
    used to return a GraphModule should now return a GraphModulePlus. We should be extra careful
    when overriding fx.GraphModule methods in this class.
    """

    ###############
    ## Factories ##
    ###############

    @staticmethod
    def new_from_copy(gm: GraphModule, name: Optional[str] = None) -> Self:
        """Create a new GraphModulePlus from a copy of an existing GraphModule. The root module
        will share references to the attributes of the model, but the underlying graphs will be
        independent."""
        # Some special handling of graph-copying is required.
        new_graph = Graph()
        output_node = new_graph.graph_copy(gm.graph, {})
        if output_node is not None:
            new_graph.output(output_node)
        class_name = gm.__class__.__name__ if name is None else name
        return GraphModulePlus(root=gm, graph=new_graph, class_name=class_name)

    @staticmethod
    def new_from_trace(module: nn.Module) -> Self:
        """Create a new GraphModulePlus by symbolic trace of a torch.nn.Module"""
        if isinstance(module, GraphModulePlus):
            return GraphModulePlus.new_from_copy(module)
        else:
            # Note: because GraphModulePlus *supersedes* GraphModule, and the original
            # fx.symbolic_trace returns a GraphModule, we can now just call symbolic_trace and it
            # will return a GraphModulePlus.
            return symbolic_trace(module)

    @staticmethod
    def new_from_merge(
        modules: dict[str, nn.Module],
        rewire_inputs: dict[str, str | Iterable[str]],
        auto_trace: bool = True,
        share_inputs: bool = True,
    ) -> Self:
        """Create a new GraphModulePlus by merging a set of modules together.

        :param modules: a mapping from module names to nn.Modules or GraphModules. The new
            GraphModulePlus object will have access to all of these modules as attributes.
        :param rewire_inputs: a mapping from the name of a node in one module to the name(s)
            of its new input node(s). That is, if rewire_inputs_from[nodeA] = [nodeX, nodeY], then
            the node named nodeA will ahve its args set to (nodeX, nodeY). This is not the same as
            *replacing* nodeX and nodeY with nodeA. Node names must be prefixed. That is, if
            modules={"A": moduleA, "B": moduleB} and moduleA has a node called node1, then the keys
            or values in rewire_inputs_from should be strings like "A_node1".
        :param auto_trace: if True, any modules that are not already GraphModules will be traced
            before being merged. If False, modules that are not GraphModules will be called
            atomically with a new call_module node.
        :param share_inputs: if True, the 'placeholder' nodes will all be replaced with a single
            shared 'x' input node. If False, each module will retain its own (prefixed) input node.
            This is overridden by any rewiring specified in 'rewire_inputs'.
        """

        new_graph = Graph()
        class_name = "Merged" + "".join([m.__class__.__name__ for m in modules.values()])
        new_node_lookup = {}

        # By default, the output of the new module will be the output of the last module in the
        # dict. If this is not the desired behavior, the output can be set manually after.
        new_output_value_node = None
        inputs_per_module: list[list[Node]] = []
        for name, module in modules.items():
            if auto_trace:
                module = GraphModulePlus.new_from_trace(module)

            match module:
                case GraphModule():
                    # If we're merging a GraphModule, copy it by copying the underlying graph
                    # after adding a prefix to all nodes. The prefix is required so that the
                    # nodes' args and targets point to attributes of the root dict.
                    prefixed_graph = prefix_all_nodes(module.graph, name)
                    new_output_value_node = new_graph.graph_copy(
                        prefixed_graph, val_map=new_node_lookup
                    )
                    inputs_per_module.append(
                        [
                            new_node_lookup[node]
                            for node in prefixed_graph.nodes
                            if node.op == "placeholder"
                        ]
                    )
                case nn.Module():
                    # We should only get here if auto_trace=False and module was a plain-old
                    # nn.Module. In this case, we'll add it as a call_module node to the graph.
                    # Its args will be set later.
                    with new_graph.inserting_after():
                        new_node = new_graph.call_module(name, args=(), kwargs={})
                    new_output_value_node = new_node
                    inputs_per_module.append([])
                case _:
                    assert_never(module)

        # If share_inputs=True, replace all placeholder nodes with a single shared 'x' node. This is
        # just a convenience so that we don't have to worry about input node names when merging
        # multiple modules together. If share_inputs=False, we'll keep the original input nodes with
        # their prefixes, which means the user will have to specify the prefixed input node names
        # when rewiring.
        if share_inputs:
            # Group placeholders by position. The first argument of every module is one
            # input, the second of every module the next, and so on. Position is what makes
            # two arguments the same input; the names are whatever each author happened to
            # pick, and nn.Sequential calling its argument "input" while another module
            # calls it "x" does not make them different inputs.
            width = max((len(inputs) for inputs in inputs_per_module), default=0)
            groups = [
                [inputs[index] for inputs in inputs_per_module if index < len(inputs)]
                for index in range(width)
            ] or [[]]

            # The first input keeps the name "x", as before, so nothing that refers to it
            # by name changes. Later ones take the name they had in the module they came
            # from, with the prefix removed.
            prefixes = tuple(f"{name}_" for name in modules)

            def unprefixed(node: Node, fallback: str) -> str:
                """Recover the name a placeholder had before the merge prefixed it."""
                for prefix in prefixes:
                    if node.name.startswith(prefix):
                        return node.name[len(prefix) :]
                return fallback

            named = [
                ("x" if index == 0 else unprefixed(group[0], f"input_{index}"), group)
                for index, group in enumerate(groups)
            ]

            # A parameter without a default cannot follow one that has a default, and fx
            # orders parameters by their position in the graph, so the placeholders without
            # defaults have to be created first.
            resolved = [
                (name, *validate_common_placeholder_attrs(group), group) for name, group in named
            ]
            resolved.sort(key=lambda item: item[2] is not inspect.Signature.empty)

            previous = None
            for name, type_expr, default_value, group in resolved:
                insertion = (
                    new_graph.inserting_before()
                    if previous is None
                    else new_graph.inserting_after(previous)
                )
                with insertion:
                    previous = new_graph.placeholder(name, type_expr, default_value)
                for node in group:
                    node.replace_all_uses_with(previous)
                    new_graph.erase_node(node)

        # Create the new GraphModulePlus object, taking attributes from the modules dict; nodes with
        # targets pointing to "moduleName.attribute" will automatically resolve, since the
        # ModuleDict class allows for attribute-style access of its keys.
        new_module = GraphModulePlus(
            root=nn.ModuleDict(modules),
            graph=new_graph,
            class_name=class_name,
        )

        # Do the rewiring. Wherever map_from_to[key] = value, we'll set the .args attribute of the
        # value node to the key node(s).
        for node, input_nodes in rewire_inputs.items():
            node = new_module._resolve_nodes(node)[0]
            node.args = tuple(new_module._resolve_nodes(input_nodes))

        new_module.set_output(new_output_value_node)

        return new_module

    def extract_subgraph(
        self, inputs: Optional[list[str | Node]] = None, output: Optional[str | Node] = None
    ) -> Self:
        """Extract a subgraph by specifying the input and output nodes. Optionally leave the
        inputs and output unspecified to have them default to the inputs and output of the original
        model.

        Note: The returned GraphModule shares attributes/submodules/parameters with the original,
        but the graph is a new object. This allows for patterns like

            with frozen(module.extract_subgraph(...)):
                ...

        to freeze some subset of the original model."""
        if inputs is None:
            inputs = self.inputs
        if output is None:
            output = self.output_value
        new_module = GraphModulePlus.new_from_copy(self, name="Sub" + self.__class__.__name__)
        new_module.set_inputs_and_output(inputs, output)
        new_module.prune_to_output()
        new_module.recompile()
        new_module.delete_all_unused_submodules()
        return new_module

    def squash_all_conv_batchnorm_pairs(self) -> Self:
        """Squash all conv-batchnorm pairs in this model. Returns a new model with
        parameters/attributes shared with the original model *except* for the new conv layers.
        """
        new_module = GraphModulePlus.new_from_copy(
            self, name="Squashed" + self.__class__.__name__
        ).eval()

        # Find all conv-batchnorm pairs
        # TODO: handle functional calls like F.conv2d; currently we assume all convs and batchnorms
        #  are called as modules.
        conv_bn_pairs = []
        for node in new_module.graph.nodes:
            if node.op == "call_module" and isinstance(
                new_module.get_submodule(node.target), nn.Conv2d
            ):
                for user in node.users:
                    if user.op == "call_module" and isinstance(
                        new_module.get_submodule(user.target), nn.BatchNorm2d
                    ):
                        conv_bn_pairs.append((node, user))

        # Squash each pair
        for conv, bn in conv_bn_pairs:
            # Replace the conv node with a new conv node that has the bn parameters folded in
            common_prefix = []
            for conv_part, bn_part in zip(conv.target.split("."), bn.target.split(".")):
                if conv_part == bn_part:
                    common_prefix.append(conv_part)
                else:
                    break
            conv_unique_name = "_".join(conv.target.split(".")[len(common_prefix) :])
            bn_unique_name = "_".join(bn.target.split(".")[len(common_prefix) :])
            squashed_name = ".".join(common_prefix + [f"{conv_unique_name}_{bn_unique_name}"])
            conv_module = new_module.get_submodule(conv.target)
            bn_module = new_module.get_submodule(bn.target)
            squashed_conv = nn.utils.fuse_conv_bn_eval(conv_module, bn_module)
            new_module.add_submodule(squashed_name, squashed_conv)
            with new_module.graph.inserting_before(conv):
                new_node = new_module.graph.call_module(
                    squashed_name, args=conv.args, kwargs=conv.kwargs
                )
                bn.replace_all_uses_with(new_node)

        # Post-surgery, clean up the graph. This will remove all unused nodes, so any conv/bn
        # nodes that we squashed end up removed but only if they are no longer used in any other
        # part of the graph.
        new_module._clean_up_and_recompile()

        return new_module

    def strip_all_where(self, condition_fn: Callable[[Node], bool]) -> Self:
        """Attempt to remove any nodes in the graph matching a condition. This just calls
        graph.erase_node() for all nodes matching the condition. If those nodes cannot be safely
        removed, graph.erase_node() will throw an error.
        """
        for node in list(reversed(self.graph.nodes)):
            if condition_fn(node):
                self.graph.erase_node(node)
        self._clean_up_and_recompile()

        return self

    def delta_state_dict(self, base_model: nn.Module) -> dict:
        """Get the difference between this module's state_dict and another module's state_dict.
        The other module is treated as the 'base'.

        Does not actually compute a difference in parameters, but rather the subset of parameters
        that are unique to this model. This is useful for training parts of models and saving
        only the unique/changed parameters.
        """
        unique_state_dict = {}
        base_state_dict = base_model.state_dict()
        for name, param in self.state_dict().items():
            base_param = base_state_dict.get(name, torch.empty(1))
            if not torch.equal(base_param, param):
                unique_state_dict[name] = param
        return unique_state_dict

    def load_delta_state_dict(
        self,
        delta_state_dict: dict,
        base_model: nn.Module,
        strict: bool = True,
        assign: bool = False,
    ) -> Any:
        """Inverse of delta_state_dict. Load a delta state dict into this model, using the base model
        params where the delta state dict does not have a value.
        """
        state_dict = base_model.state_dict()
        state_dict.update(delta_state_dict)
        return self.load_state_dict(state_dict, strict=strict, assign=assign)

    #############################
    ## Node management helpers ##
    #############################

    def _resolve_nodes(self, names: str | Node | Iterable[str | Node]) -> list[Node]:
        """Get all nodes in the graph with a given name. Raises a ValueError if any of the names
        are not present in the graph. Returns a list of nodes in the same order as the input names.
        """
        # Start by ensuring the type of 'names' is a list of strings
        if isinstance(names, str) or isinstance(names, Node):
            names = [names]
        names = [n.name if isinstance(n, Node) else n for n in names]

        # Keep track of which nodes are found
        lookup_node_by_name: dict[str, Optional[Node]] = {name: None for name in names}

        for node in self.graph.nodes:
            if node.name in lookup_node_by_name:
                lookup_node_by_name[node.name] = node

        missing_names = [name for name, node in lookup_node_by_name.items() if node is None]
        if missing_names:
            raise ValueError("Not all nodes are present in the graph:", missing_names)

        # This relies on the fact that python dicts are ordered, so the output will be in the same
        # order as the input.
        return list(lookup_node_by_name.values())

    def users_of(self, node: str | Node) -> list[Node]:
        """Get the users of a node in the graph."""
        node = self._resolve_nodes(node)[0]
        return list(node.users)

    def insert_noop(self, node: str | Node) -> Node:
        """Insert a no-op (Identity) node in between the given node and its downstream users."""
        to_replace = self._resolve_nodes(node)[0]
        if "noop" not in self._modules:
            self.add_submodule("noop", torch.nn.Identity())
        with self.graph.inserting_after(to_replace):
            noop = self.graph.call_module("noop", args=(to_replace,))
            # Replace the original node with the noop node. (Tricky note: this also makes the noop
            # take itself as input because it is a user of the original node. This will be fixed
            # in a moment.)
            to_replace.replace_all_uses_with(noop)
            # Now we need to fix the noop node's args. It should take the original node as input.
            noop.replace_input_with(noop.args[0], to_replace)
        return noop

    @property
    def inputs(self) -> List[Node]:
        """Get all input nodes in the graph."""
        return [node for node in self.graph.nodes if node.op == "placeholder"]

    def _rm_output(self) -> Self:
        # Cast to a list so that we're not modifying the graph while iterating over it.
        for node in list(self.graph.nodes):
            if node.op == "output":
                self.graph.erase_node(node)
        return self

    @property
    def output(self) -> Node:
        """Get the output node of a graph (assumes there is exactly one). Note that this is just
        a pointer to the node whose value is output. Use output_value for the actual node whose
        value is returned."""
        return next(node for node in self.graph.nodes if node.op == "output")

    @property
    def output_value(self) -> Node:
        """Get the Node which is returned by the graph, which is distinct from the output node.
        The output node is a pointer to what we call the output_value node.
        """
        return self.output.args[0]

    def _update_all_inplace_ops(self, inplace: bool = False) -> Self:
        """Update any inplace operations (e.g. ReLU(inplace=True)) in a model. Set their
        'inplace' attribute to the given value. Setting inplace=False helps for instance by
        making the functions 'pure' and thus play nicer with torch.fx and torch.func.

        The module is itself modified in-place.

        Warning: this only works if the inplace ops are modules. That is, nn.ReLU will be
        updated, but function calls like F.relu will not be updated!
        """
        for node in self.graph.nodes:
            if node.op == "call_module":
                module = self.get_submodule(node.target)
                if hasattr(module, "inplace"):
                    module.inplace = inplace
        return self

    ########################
    ## Graph manipulation ##
    ########################

    def eliminate_dead(self, remove_unused_inputs: bool = True) -> Self:
        """Call self.graph.eliminate_dead_code() AND remove any lingering unused placeholder nodes,
        unless remove_unused_inputs=False.
        """
        # elminiate_dead_code() works backwards from outputs to inputs (it assumes the graph is
        # topologically sorted), removing any node that doesn't have users.
        self.graph.eliminate_dead_code()

        # For reasons unknown, eliminate_dead_code() doesn't remove unused placeholders. We'll do
        # that manually here.
        if remove_unused_inputs:
            # Cast to a list so that we're not modifying the graph while iterating over it.
            for node in list(self.graph.nodes):
                if node.op == "placeholder" and len(node.users) == 0:
                    self.graph.erase_node(node)

        return self

    def prune_to_output(self) -> Self:
        """Erase every node with no path to the output.

        eliminate_dead() removes nodes with no users, which is not the same
        thing. A node can have users and still be unreachable from the output,
        which happens when a traced model computes something it does not use,
        such as a shape check. Those branches survive eliminate_dead(), and any
        placeholder feeding one survives with them, leaving an extracted
        subgraph with inputs it does not need.

        Walks back from the output node, collects everything reachable, and
        erases the rest in reverse topological order so that no node is removed
        before its users.
        """
        output = next(node for node in self.graph.nodes if node.op == "output")

        reachable, stack = {output}, [output]
        while stack:
            node = stack.pop()
            for parent in node.all_input_nodes:
                if parent not in reachable:
                    reachable.add(parent)
                    stack.append(parent)

        for node in reversed(list(self.graph.nodes)):
            if node not in reachable:
                self.graph.erase_node(node)

        self.graph.lint()
        self.recompile()
        self.delete_all_unused_submodules()

        return self

    def _clean_up_and_recompile(self):
        # The purpose of 'eliminate_dead' is to remove Nodes from the Graph that have no path to
        # any output node.
        self.eliminate_dead()
        # The purpose of 'recompile' is to take the modified Graph recompile it into a
        # self.forward() method. Without recompiling, changes to the graph don't actually affect
        # the module's function.
        self.recompile()
        # The purpose of 'delete_all_unused_submodules' is to remove attributes of self that point
        # to other nn.Modules which are no longer used after the graph has been modified.
        self.delete_all_unused_submodules()

    def set_inputs(self, inputs: list[str | Node]) -> Self:
        """Set the inputs of this graph by finding nodes of the given name(s) and replacing them
        with placeholders. Modifies the graph attribute in-place."""
        # For each named input, erase any existing nodes of the same name and replace them with a
        # new placeholder node.
        dont_remove = []
        for node in self._resolve_nodes(inputs):
            if node.op == "placeholder":
                dont_remove.append(node)
                continue
            else:
                with self.graph.inserting_before(node):
                    new_placeholder = self.graph.placeholder(node.name, node.type)
                    node.replace_all_uses_with(new_placeholder)
                    self.graph.erase_node(node)
                    # Handle potential name collision by reinforcing the new node name now that
                    # the old node is deleted.
                    if new_placeholder.name != node.name:
                        new_placeholder.name = node.name
                    dont_remove.append(new_placeholder)

        self._clean_up_and_recompile()

        return self

    def set_output(self, output: str | Node | Any) -> Self:
        """Remove all preexisting outputs and set the output of a graph to the node of the given
        name."""
        if isinstance(output, str) or isinstance(output, Node):
            # Find the named node to be the arg to a new output node
            node_to_output = self._resolve_nodes(output)[0]
        else:
            # In cases like outputting a dict, the output passed in is itself the thing we want
            # to pass to self.graph.output() below.
            node_to_output = output

        # Remove all preexisting outputs.
        self._rm_output()

        with self.graph.inserting_after():
            self.graph.output(node_to_output)

        self._clean_up_and_recompile()

        return self

    def set_dict_outputs(self, outputs: Iterable[str | Node]) -> Self:
        """Modify the Graph by adding a new node which collects multiple outputs in a dict. This
        new node will then become the output of the graph.
        """
        # Find the named nodes to be the args to a new output node
        nodes_to_output = self._resolve_nodes(outputs)
        names = [node.name for node in nodes_to_output]

        # Remove all preexisting outputs.
        self._rm_output()

        with self.graph.inserting_after():
            # Create a new node which collects the outputs into a dict
            collector_node = self.graph.call_function(
                the_function=dict,
                kwargs={name: node for name, node in zip(names, nodes_to_output)},
            )
            # Set the new 'collector' node as the output of the graph
            self.graph.output(collector_node)

        self._clean_up_and_recompile()

        return self

    def set_inputs_and_output(self, inputs: list[str | Node], output: str | Node) -> Self:
        """Set both the inputs and output of this graph. Modifies the graph attribute in-place."""
        if output in inputs:
            raise ValueError("Output node cannot also be an input node.")
        self.set_output(output)
        self.set_inputs(inputs)

        return self

    ############################################
    ## Debugging and visualization utilities ##
    ############################################

    def to_dot(self) -> pydot.Dot:
        dot = pydot.Dot()
        for node in self.graph.nodes:
            dot.add_node(pydot.Node(node.name))
            for user in node.users:
                dot.add_edge(pydot.Edge(node.name, user.name))
        return dot

    def step_through_call(self, context: Optional[dict] = None, callback=None) -> Any:
        """Line-by-line debuggable model evaluation allowing the graph itself to be debugged.

        The idea is that

             output = model(**context)

        has the same behavior as

            output = model.step_through_call(context)

        but the latter allows for stepping through the graph node-by-node, inspecting the state of
        the model at each step, and callbacks.
        """

        def _get_arg(arg: Any):
            if isinstance(arg, Node):
                return context[arg.name]
            elif isinstance(arg, list):
                return [_get_arg(a) for a in arg]
            elif isinstance(arg, dict):
                return {k: _get_arg(v) for k, v in arg.items()}
            else:
                return arg

        for node in self.graph.nodes:
            match node.op:
                case "placeholder":
                    assert node.name in context, f"Missing input {node.name}"
                case "get_attr":
                    obj = self
                    for part in node.target.split("."):
                        obj = getattr(obj, part)
                    context[node.name] = obj
                case "call_module":
                    module = self.get_submodule(node.target)
                    args = [_get_arg(arg) for arg in node.args]
                    kwargs = {k: _get_arg(v) for k, v in node.kwargs.items()}
                    context[node.name] = module(*args, **kwargs)
                case "call_method":
                    self_obj, *args = [_get_arg(arg) for arg in node.args]
                    kwargs = {k: _get_arg(v) for k, v in node.kwargs.items()}
                    method = getattr(self_obj, node.target)
                    context[node.name] = method(*args, **kwargs)
                case "call_function":
                    the_function = node.target
                    args = [_get_arg(arg) for arg in node.args]
                    kwargs = {k: _get_arg(v) for k, v in node.kwargs.items()}
                    context[node.name] = the_function(*args, **kwargs)
                case "output":
                    args = [_get_arg(arg) for arg in node.args]
                    return args[0]
                case _:
                    assert_never(node.op)

            if callback is not None:
                callback(node, context[node.name])
