import sys
from functools import wraps
from inspect import getmembers
from importlib.metadata import metadata, PackageNotFoundError

from .core import Core
from .test import benchmark, test
from . import api as _api
from .api import __all__ as api_all
from .client import Client, ClientBase

# public entry points
benchmark.__module__ = __name__
test.__module__ = __name__

try:
    _metadata = metadata("nimbro-api")
    __version__ = _metadata['Version']
    __author__ = _metadata.get('Author') or _metadata.get('Author-email', "")
except PackageNotFoundError:
    pass

__all__ = ["benchmark", "test"]

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

def __getattr__(name):
    if name in api_all:
        value = getattr(_api, name)
        globals()[name] = value
        return value
    raise AttributeError(f"Module '{__name__}' has no attribute '{name}'.")

def __dir__():
    return sorted(set(globals()) | set(__all__))

__all__.extend(api_all)
