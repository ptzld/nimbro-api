import importlib

_CLIENTS = {
    'Dam': "dam",
    'Florence2': "florence2",
    'Kosmos2': "kosmos2",
    'MmGroundingDino': "mmgroundingdino",
    'Sam2Realtime': "sam2_realtime"
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
