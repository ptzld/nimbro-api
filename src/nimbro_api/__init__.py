import sys
from functools import wraps
from inspect import getmembers
from importlib.metadata import version, metadata, PackageNotFoundError

from .core import Core
from .test import test
from . import api as _api
from .api import __all__ as api_all
from .client import Client, ClientBase

try:
    __version__ = version("nimbro-api")
    __author__ = metadata("nimbro-api").get("Author") or metadata("nimbro-api").get("Author-email", "")
except PackageNotFoundError:
    pass

__all__ = ["test"]

_core = Core()
_module = sys.modules[__name__]

def make_wrapper(method):
    @wraps(method)
    def wrapper(*args, **kwargs):
        return method(*args, **kwargs)
    return wrapper

for name, attr in getmembers(_core, callable):
    if not name.startswith("_"):
        setattr(_module, name, make_wrapper(attr))
        __all__.append(name)

__all__.extend(["Client", "ClientBase"])

conflicts = set(__all__) & set(api_all)
if conflicts:
    raise ImportError(f"API modules export conflicting name{'' if len(conflicts) == 1 else 's'}: {conflicts}")

for name in api_all:
    setattr(_module, name, getattr(_api, name))

__all__.extend(api_all)
