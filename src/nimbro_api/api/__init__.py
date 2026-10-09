import pkgutil
import importlib

__all__ = []

for module_info in pkgutil.iter_modules(__spec__.submodule_search_locations):
    if not module_info.ispkg:
        continue

    name = module_info.name
    __all__.append(name)

def __getattr__(name):
    if name in __all__:
        module = importlib.import_module(f".{name}", package=__name__)
        globals()[name] = module
        return module
    raise AttributeError(f"Module '{__name__}' has no attribute '{name}'.")

def __dir__():
    return sorted(set(globals()) | set(__all__))
