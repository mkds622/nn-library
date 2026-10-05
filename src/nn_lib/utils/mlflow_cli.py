"""MLflow helpers: artifact I/O, plus DEPRECATED param-matching query utilities.

Current:
- `open_mlflow_artifact_file`, `save_as_artifact`, `load_artifact`: read/write MLflow run
  artifacts without manual tempfile bookkeeping.

Deprecated (kept because existing projects depend on them; superseded by
`nn_lib.utils.run_registry`):
- `search_runs_by_params`, `search_single_run_by_params`, `run_has_params`, and
  `flatten_params` match runs by stringified params via MLflow filter strings, which is
  fragile (quote escaping, 6000-char truncation, coupling to `str()` serialization). New
  code should use `run_registry.RunIndex` / `hash_spec` instead.
"""

import os
import pickle
import tempfile
try:
    from warnings import deprecated          # Python 3.13+
except ImportError:
    from typing_extensions import deprecated # 3.9 to 3.12
from contextlib import contextmanager
from pathlib import Path
from typing import Union, Optional, Any, Literal, Iterable, Generator

import mlflow
import pandas as pd
import torch
from jsonargparse import Namespace
from mlflow.entities import Run

RunOrURI = Union[pd.Series, Run, str, Path]


class RunExists(Exception):
    def __init__(self, run: Run, *args):
        super().__init__(*args)
        self.run = run


class RunDoesNotExist(Exception):
    pass


def _path_to_mlflow_path(path: Path) -> str | None:
    str_path = str(path)
    if str_path.startswith("."):
        str_path = str_path.lstrip("." + os.sep)
    if str_path == "":
        str_path = None
    return str_path


@contextmanager
def open_mlflow_artifact_file(
    path: Union[str, Path], mode: str = "w", run_id: Optional[str] = None
):
    path = Path(path)
    path, name = path.parent, path.name

    if run_id is None:
        run_id = mlflow.active_run().info.run_id

    artifact_exists = False
    for file in mlflow.artifacts.list_artifacts(run_id=run_id, artifact_path=str(path)):
        if Path(file.path).name == name:
            artifact_exists = True
            break

    with tempfile.TemporaryDirectory() as tmpdir:
        if artifact_exists:
            local_file = mlflow.artifacts.download_artifacts(
                run_id=run_id, artifact_path=str(path / name), dst_path=tmpdir
            )
        else:
            local_file = Path(tmpdir) / name

        with open(local_file, mode) as f:
            yield f

        mlflow.log_artifact(
            str(local_file), artifact_path=_path_to_mlflow_path(path), run_id=run_id
        )


def _quote_value(val: Any):
    val = str(val)
    has_single_quote = "'" in val
    has_double_quote = '"' in val
    if has_single_quote and has_double_quote:
        # Todo: figure out how to escape characters in values. MLFlow docs seem to imply it
        #  should be supported, but I can't get it to work.
        raise ValueError(
            "Parameter value containing both single and double quotes will be a problem "
            "for MLFlow filter strings"
        )
    if has_single_quote:
        return f'"{val}"'
    else:
        return f"'{val}'"


def _build_filter_string(
    params: Optional[Namespace] = None,
    finished_only: bool = True,
    skip_keys: Optional[Iterable[str]] = None,
) -> str:
    query_parts = []
    if params is not None:
        # NB: not flatten_params() to avoid cascading its DeprecationWarning onto callers.
        flattened_params = dict(_iter_params_skip(params, skip_keys))
        query_parts.extend(
            [
                f"params.`{k}` = {_quote_value(v)}"
                for k, v in flattened_params.items()
                if v is not None
            ]
        )
    if finished_only:
        query_parts.append("status = 'FINISHED'")
    return " and ".join(query_parts)


def search_runs_by_params(
    params: Optional[Namespace] = None,
    experiment_name: Optional[str | list[str]] = None,
    tracking_uri: Optional[Union[str, Path]] = None,
    finished_only: bool = True,
    skip_keys: Optional[Iterable[str]] = None,
    output_format: Literal["pandas", "list"] = "pandas",
) -> list[Run] | pd.DataFrame:
    """Query the MLflow server for runs in the specified experiment that match the given
    parameters (which will be flattened if they aren't already). Keys in `skip_keys` will be
    ignored.

    Inteded use of this function is for analysis/plotting. For managing runs before/as they are
    happening, e.g. for deduplication, run_registry utilities are recommended.
    """
    query_string = _build_filter_string(params, finished_only, skip_keys)
    if tracking_uri:
        mlflow.set_tracking_uri(tracking_uri)

    if experiment_name is not None:
        if isinstance(experiment_name, str):
            experiment_name = [experiment_name]
        elif isinstance(experiment_name, list):
            experiment_name = experiment_name
        else:
            raise ValueError("`experiment_name` must be a string or a list of strings")

    return mlflow.search_runs(
        experiment_names=experiment_name, filter_string=query_string, output_format=output_format
    )


def search_single_run_by_params(
    params: Optional[Namespace] = None,
    experiment_name: Optional[str | list[str]] = None,
    tracking_uri: Optional[Union[str, Path]] = None,
    finished_only: bool = True,
    skip_keys: Optional[Iterable[str]] = None,
) -> Run:
    """Query the MLflow server for runs in the specified experiment that match the given parameters.
    If exactly one run is found, return it. If no runs or multiple runs are found, raise an error.
    """
    runs: list[Run] = search_runs_by_params(
        params, experiment_name, tracking_uri, finished_only, skip_keys, output_format="list"
    )
    if len(runs) == 0:
        raise RunDoesNotExist("No runs found with the specified parameters")
    elif len(runs) > 1:
        raise RunExists(runs[0], "Multiple runs found with the specified parameters")
    return runs[0]


@deprecated(
    "run_registry contains better utilities for canonicalizing/serializing parameters"
)
def run_has_params(run: Run, params: Namespace, skip_keys: Optional[Iterable[str]] = None) -> bool:
    """Check if a Run has parameters. This provides an alternative to 'search_runs_by_params' where
    runs can be pre-fetched from the MLflow server and compared in memory to params.
    """
    run_params = run.data.params
    for k, v in _iter_params_skip(params, skip_keys):
        # Internally, mlflow wraps all params in a str() call, so we need to check equality vs
        # the stringified param (which could be any data type). See mlflow.log_params for reference.
        if k not in run_params or run_params[k] != str(v):
            return False
    return True


def save_as_artifact(
    obj: object,
    path: str | Path,
    run_id: Optional[str] = None,
    backend: Literal["torch", "pickle"] = "torch",
):
    """Use torch.save or pickle.dump to save the given object to the given path as an MLflow
    artifact in the given run.
    """
    if isinstance(path, str):
        path = Path(path)
    with tempfile.TemporaryDirectory() as tmpdir:
        local_file = Path(tmpdir) / path.name
        remote_path = str(path.parent) if path.parent != Path() else None
        match backend:
            case "torch":
                torch.save(obj, local_file)
            case "pickle":
                with open(local_file, "wb") as f:
                    pickle.dump(obj, f)
            case _:
                raise ValueError(f"Unknown backend: {backend}")
        mlflow.log_artifact(str(local_file), artifact_path=remote_path, run_id=run_id)


def load_artifact(
    path: str | Path,
    run_id: Optional[str] = None,
    backend: Literal["torch", "pickle"] = "torch",
    **kwargs,
) -> object:
    """Use torch.load or pickle.load to load the given artifact from the specified MLflow run.
    Path is relative to the artifact URI, just like save_as_artifact()
    """
    if isinstance(path, Path):
        path = str(path)
    if run_id is None:
        run_id = mlflow.active_run().info.run_id
    # Note: despite the name, "downloading" artifacts involves no copying of files if we leave the
    # local path unspecified and the artifacts are stored on this file system.
    local_path = mlflow.artifacts.download_artifacts(run_id=run_id, artifact_path=path)
    match backend:
        case "torch":
            return torch.load(local_path, **kwargs)
        case "pickle":
            with open(local_path, "rb") as f:
                return pickle.load(f, **kwargs)
        case _:
            raise ValueError(f"Unknown backend: {backend}")


def _match_nested_key_prefix(key: str, prefix: str) -> bool:
    """Check if a maybe-dot-separated prefix like 'foo' matches a maybe-dot-separated key like
    'foo.bar.baz'. This is used to skip nested keys when flattening a Namespace. Note that dot
    separators are important and 'f' is not considered a prefix of 'foo' or 'foo.bar', but 'f' *is*
    a prefix of 'f.x.y.z'.
    """
    key_parts = key.split(".")
    prefix_parts = prefix.split(".")
    if len(prefix_parts) > len(key_parts):
        return False
    for kp, pp in zip(key_parts, prefix_parts):
        if kp != pp:
            return False
    return True


def _iter_params_skip(
    params: Namespace, skip_keys: Optional[Iterable[str]] = None
) -> Generator[tuple[str, Any], None, None]:
    """Like Namespace.items(), iterate leaf (key, value) options of a possibly-nested Namespace.
    For example, if `params` is like `Namespace(a=1, b=Namespace(c=2, d=3))`, this will yield
    `('a', 1)`, `('b.c', 2)`, and `('b.d', 3)`.

    'skip_keys' can specify fields to skip. For instance if `skip_keys = ['a']` then we just get
    `('b.c', 2)`, and `('b.d', 3)`. If `skip_keys = ['b']` we get just `('a', 1)`. Skipped keys can
    also be nested, so if `skip_keys = ['a', 'b.c']` we'll get back just `('b.d', 3)`. If
    `skip_keys = None` (the default), nothing is skipped and all elements from `params.items()` are
    yielded.
    """
    skip_keys = set() if skip_keys is None else set(skip_keys)
    for nested_key, value in params.items():
        if any(_match_nested_key_prefix(nested_key, skip) for skip in skip_keys):
            continue
        yield nested_key, value


@deprecated("Use flatten() and to_plain() in run_registry instead, if possible")
def flatten_params(params: Namespace, skip_keys: Optional[Iterable[str]] = None) -> dict:
    """Flatten the given parameters, like Namespace.as_flat, but allow some keys to be skipped and
    returning as a dict.
    """
    return dict(_iter_params_skip(params, skip_keys))


__all__ = [
    "RunDoesNotExist",
    "RunExists",
    "load_artifact",
    "open_mlflow_artifact_file",
    "save_as_artifact",
    "search_runs_by_params",
    "search_single_run_by_params",
]
