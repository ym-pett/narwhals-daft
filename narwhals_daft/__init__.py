from __future__ import annotations
from narwhals_daft import dataframe
from narwhals_daft import expr as expr
import daft
from narwhals.utils import Version
from typing import Any
from typing_extensions import TypeIs

def __getattr__(name: str) -> Any:
    if name == "from_native":
        from narwhals_daft.namespace import DaftNamespace

        from narwhals._utils import Version

        return DaftNamespace(version=Version.MAIN).from_native
    msg = f"module {__name__!r} has no attribute {name!r}"
    raise AttributeError(msg)

def from_native(native_object: daft.DataFrame, eager_only: bool, series_only: bool) -> dataframe.DaftLazyFrame:
    if eager_only or series_only:
        raise ValueError("eager_only and series_only options are not supported as daft is lazy-only.")
    return dataframe.DaftLazyFrame(native_object, version=Version.MAIN)

def is_native_object(obj: Any) -> TypeIs[daft.DataFrame]:
    return isinstance(obj, daft.DataFrame)
