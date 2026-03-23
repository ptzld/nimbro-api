import pkgutil
import importlib

__all__ = []

for module_info in pkgutil.iter_modules(__spec__.submodule_search_locations):
    if not module_info.ispkg:
        continue

    name = module_info.name
    module = importlib.import_module(f".{name}", package=__name__)

    __all__.append(name)
    globals()[name] = module
