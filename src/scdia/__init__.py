"""DIA single-cell proteomics analysis toolkit."""

from importlib import import_module
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from . import io, pl, pp, qc, tl

__version__ = "0.1.0"

__all__ = ["io", "qc", "pp", "tl", "pl"]


def __getattr__(name):
    if name in __all__:
        return import_module(f"{__name__}.{name}")
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
