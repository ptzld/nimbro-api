import types

from .api import get_api_key, validate_endpoint, HttpRequestCancelled, post_request, get_request, http_request

from .io import (
    download_file, read_json, write_json, read_as_b64,
    encode_b64, decode_b64, parse_image_b64, parse_audio_bytes,
    get_cache_location, acquire_lock, release_lock
)

from .logger import Logger

from .misc import (
    UnrecoverableError, assert_type_value, assert_keys, assert_log,update_dict, count_duplicates,
    get_image_dimensions, escape, visible_len, print_lines, format_obj
)

from .string import is_url, is_base64, extract_json

_VISUAL_EXPORTS = {
    "Color",
    "ColorPalette",
    "nimbro_colors",
    "kelly_colors",
    "visualize_detections",
    "draw_rectangle",
    "draw_text",
    "convert_boxes",
}

def __getattr__(name):
    if name in _VISUAL_EXPORTS:
        import importlib
        visual = importlib.import_module(".visual", __name__)
        value = getattr(visual, name)
        globals()[name] = value
        return value
    raise AttributeError(f"Module '{__name__}' has no attribute '{name}'.")

__all__ = [
    name for name, obj in globals().items()
    if not name.startswith("_") and not isinstance(obj, types.ModuleType)
]
