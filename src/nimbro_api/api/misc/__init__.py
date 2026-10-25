import importlib

_CLIENTS = {
    'No': "no",
    'SearXNG': "searxng",
    'SystemOne': "system_one",
    'VlmGist': "vlm_gist"
}

__all__ = list(_CLIENTS)

def __getattr__(name):
    if name in _CLIENTS:
        module = importlib.import_module(f".client.{_CLIENTS[name]}", package=__name__)
        value = getattr(module, name)
        globals()[name] = value
        return value
    raise AttributeError(f"Module '{__name__}' has no attribute '{name}'.")

def __dir__():
    return sorted(set(globals()) | set(__all__))
