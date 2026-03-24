import os
import time
import json
import threading

import requests

try:
    import orjson
    ORJSON_AVAILABLE = True
except ImportError:
    ORJSON_AVAILABLE = False
try:
    import pybase64
    PYBASE64_AVAILABLE = True
except ImportError:
    import base64
    PYBASE64_AVAILABLE = False

import nimbro_api
from .misc import UnrecoverableError, assert_type_value
from .string import is_url, is_base64

IS_WINDOWS = os.name == "nt"
if IS_WINDOWS:
    import msvcrt
else:
    import fcntl
LOCK_HANDLES = set() # storing locked resources for monitoring/debugging
LOCK_HANDLES_LOCK = threading.Lock() # ensuring thread-safety for LOCK_HANDLES

def download_file(url, *, retry=1, name="file", logger=None):
    """
    Download a file from the internet.

    Args:
        url (str):
            The URL from which to download the image.
        retry (bool | int, optional):
            If `True`, retry until successful.
            If `int` > 0, retry this often before returning a failure. Defaults to 1.
        name (str, optional):
            Descriptive name for logging. Defaults to "file".
        logger (nimbro_api.utility.logger.Logger | None, optional):
            If provided, logs status/error messages. Defaults to `None`.

    Raises:
        UnrecoverableError: If input arguments are invalid.

    Returns:
        tuple[bool, str, bytes | None]: A tuple containing:
            - bool: `True` if the operation succeeded, `False` otherwise.
            - str: A descriptive message about the operation result.
            - bytes | None: File as bytes, or `None` if not successful.

    Notes:
        - This function first queries the cache via `nimbro_api.query_cache()` using the category "download_file" and the 'url' as the identifier.
        - If the download is successful, the cache is updated via 'nimbro_api.update_cache()'.
    """
    # parse arguments
    from nimbro_api.utility.logger import Logger
    assert_type_value(obj=logger, type_or_value=[Logger, None], name="argument 'logger'")
    assert_type_value(obj=url, type_or_value=str, name="argument 'url'")
    assert_type_value(obj=retry, type_or_value=[bool, int], name="argument 'retry'")
    assert_type_value(obj=name, type_or_value=str, name="argument 'name'")

    # query cache
    success, message, cache = nimbro_api.query_cache(category="download_file", identifier=url, age=None, mute=True)
    if logger is not None:
        logger.debug(message)
    if success:
        data = cache['data']
        return True, f"Obtained {name} '{url}' from cache.", data

    # download
    stamp = time.perf_counter()
    message = None
    while True:
        if logger is not None:
            if message is None:
                logger.debug(f"Downloading {name} '{url}'.")
            else:
                logger.warn(f"Retrying download of {name} '{url}' after failure: {message}")
        try:
            headers = {'User-Agent': 'Mozilla/5.0'}
            response = requests.get(url, headers=headers, timeout=60.0)
        except Exception as e:
            message = f"Failed to download {name} from '{url}': {repr(e)}"
            if isinstance(retry, bool) and retry is True:
                continue
            if isinstance(retry, int) and retry > 0:
                retry -= 1
                continue
            return False, message, None
        message = f"Downloaded {name} '{url}' in '{time.perf_counter() - stamp:.3f}s'."
        data = response.content
        # cache response
        _success, _message = nimbro_api.update_cache(category="download_file", identifier=url, data=data, mute=True)
        if logger is not None:
            if _success:
                logger.debug(_message)
            else:
                logger.warn(_message)
            return True, message, data

def read_json(file_path, *, name="file", logger=None):
    """
    Read and decode a JSON file.

    Args:
        file_path (str):
            Path to the JSON file.
        name (str, optional):
            Descriptive name for logging. Defaults to "file".
        logger (nimbro_api.utility.logger.Logger | None, optional):
            If provided, logs status/error messages. Defaults to `None`.

    Raises:
        UnrecoverableError: If input arguments are invalid (excluding invalid file path, error while reading file, or decoding it as JSON).

    Returns:
        tuple[bool, str, any]: A tuple containing:
            - bool: `True` if the operation succeeded, `False` otherwise.
            - str: A descriptive message about the operation result.
            - any: Decoded JSON object if successful, or `None` if not successful.

    Notes:
        - If available, this function uses the faster `orjson` module; otherwise, it falls back to the standard `json` module.
    """
    # parse arguments
    from nimbro_api.utility.logger import Logger
    assert_type_value(obj=logger, type_or_value=[Logger, None], name="argument 'logger'")
    assert_type_value(obj=file_path, type_or_value=str, name="argument 'file_path'")
    assert_type_value(obj=name, type_or_value=str, name="argument 'name'")

    # read and decode file

    tic = time.perf_counter()

    if not os.path.exists(file_path):
        success = False
        message = f"Failed to read {name} '{file_path}': Path does not exist."
        json_object = None
    elif not os.path.isfile(file_path):
        success = False
        message = f"Failed to read {name} '{file_path}': Path is not a file."
        json_object = None
    else:
        success = True

    if success:

        if logger is not None:
            logger.debug(f"Reading {name} '{file_path}'.")

        try:
            if ORJSON_AVAILABLE:
                with open(file_path, "rb") as f:
                    json_object = orjson.loads(f.read())
            else:
                if logger is not None:
                    logger.debug(f"Using slow 'json' module to read {name}. Install 'orjson' (pip install orjson) to speed this up!", once=True)
                with open(file_path, 'r', encoding='utf-8') as f:
                    json_object = json.load(f)
        except Exception as e:
            success = False
            message = f"Failed to read or decode {name} '{file_path}': {repr(e)}"
            json_object = None
        else:
            message = f"Read {name} '{file_path}' in '{time.perf_counter() - tic:.3f}s'."

    return success, message, json_object

def write_json(file_path, json_object, *, indent=True, name="file", logger=None):
    """
    Encode and write a JSON object.

    Args:
        file_path (str):
            Destination path for the JSON file.
        json_object (any):
            Object to serialize to JSON.
        indent (bool, optional):
            Pretty-print with indentation (2 spaces) when `True`. Defaults to `True`.
        name (str, optional):
            Descriptive name for logging. Defaults to "file".
        logger (nimbro_api.utility.logger.Logger | None, optional):
            If provided, logs status/error messages. Defaults to `None`.

    Raises:
        UnrecoverableError: If input arguments are invalid (excluding invalid file path, error while writing file, or encoding it as JSON).

    Returns:
        tuple[bool, str]: A tuple containing:
            - bool: `True` if the operation succeeded, `False` otherwise.
            - str: A descriptive message about the operation result.

    Notes:
        - If available, this function uses the faster `orjson` module; otherwise, it falls back to the standard `json` module.
    """
    # parse arguments
    from nimbro_api.utility.logger import Logger
    assert_type_value(obj=logger, type_or_value=[Logger, None], name="argument 'logger'")
    assert_type_value(obj=file_path, type_or_value=str, name="argument 'file_path'")
    assert_type_value(obj=indent, type_or_value=bool, name="argument 'indent'")
    assert_type_value(obj=name, type_or_value=str, name="argument 'name'")

    # encode and write file

    tic = time.perf_counter()
    file_path = os.path.abspath(file_path)

    if logger is not None:
        logger.debug(f"Writing {name} '{file_path}'.")

    target_folder = os.path.dirname(file_path)
    if not os.path.exists(target_folder):
        if logger is not None:
            logger.debug(f"Creating directory '{target_folder}'.")
        try:
            os.makedirs(target_folder)
        except Exception as e:
            success = False
            message = f"Failed to create directory '{target_folder}': {repr(e)}"
            return success, message
    elif not os.path.isdir(target_folder):
        message = f"Expected path '{target_folder}' to either not exist or be a directory."
        raise UnrecoverableError(message)

    try:
        if ORJSON_AVAILABLE:
            with open(file_path, "wb") as f:
                try:
                    if indent:
                        f.write(orjson.dumps(json_object, option=orjson.OPT_INDENT_2))
                    else:
                        f.write(orjson.dumps(json_object))
                except Exception as e:
                    if str(e) == 'Integer exceeds 64-bit range':
                        if logger is not None:
                            logger.debug(f"Using slow 'json' module to write {name} after 'orjson' error: {repr(e)}")
                        with open(file_path, 'w', encoding='utf-8') as f:
                            json.dump(json_object, f, indent=2 if indent else None)
                    else:
                        raise e
        else:
            if logger is not None:
                logger.debug(f"Using slow 'json' module to write {name}. Install 'orjson' (pip install orjson) to speed this up!", once=True)
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(json_object, f, indent=2 if indent else None)
    except Exception as e:
        success = False
        message = f"Failed to write or encode {name} '{file_path}': {repr(e)}"
    else:
        success = True
        message = f"Written {name} '{file_path}' in '{time.perf_counter() - tic:.3f}s'."

    return success, message

def read_as_b64(file_path, *, name="file", logger=None):
    """
    Read a file as a Base64 encoded ASCII string.

    Args:
        file_path (str):
            Path to the file to read.
        name (str, optional):
            Descriptive name for logging. Defaults to "file".
        logger (nimbro_api.utility.logger.Logger | None, optional):
            If provided, logs status/error messages. Defaults to `None`.

    Raises:
        UnrecoverableError: If input arguments are invalid (excluding invalid file path or error while loading file).

    Returns:
        tuple[bool, str]: A tuple containing:
            - bool: `True` if the operation succeeded, `False` otherwise.
            - str: A descriptive message about the operation result.
    """
    # parse arguments
    from nimbro_api.utility.logger import Logger
    assert_type_value(obj=logger, type_or_value=[Logger, None], name="argument 'logger'")
    assert_type_value(obj=file_path, type_or_value=str, name="argument 'file_path'")
    assert_type_value(obj=name, type_or_value=str, name="argument 'name'")

    # read and encode file
    if not os.path.exists(file_path):
        success = False
        message = f"Failed to read {name} '{file_path}': Path does not exist."
        obj_str = None
    elif not os.path.isfile(file_path):
        success = False
        message = f"Failed to read {name} '{file_path}': Path is not a file."
        obj_str = None
    else:
        if logger is not None:
            logger.debug(f"Reading {name} '{file_path}'.")
        tic = time.perf_counter()
        try:
            with open(file_path, "rb") as f:
                success, message, obj_str = encode_b64(
                    obj=f.read(),
                    name=name,
                    logger=logger
                )
        except Exception as e:
            success = False
            message = f"Failed to read {name}: {repr(e)}"
            obj_str = None
        else:
            if success:
                message = f"Read and encoded {name} '{file_path}' as Base64 in '{time.perf_counter() - tic:.3f}s'."

    return success, message, obj_str

def encode_b64(obj, *, name="object", logger=None):
    """
    Encode a bytes-like object as a Base64 ASCII string.

    Args:
        obj (bytes):
            Object to be encoded as bytes.
        name (str, optional):
            Descriptive name for logging. Defaults to "object".
        logger (nimbro_api.utility.logger.Logger | None, optional):
            If provided, logs status/error messages. Defaults to `None`.

    Raises:
        UnrecoverableError: If input arguments are invalid.

    Returns:
        tuple[bool, str, str | None]: A tuple containing:
            - bool: `True` if the operation succeeded, `False` otherwise.
            - str: A descriptive message about the operation result.
            - str | None: Object encoded as Base64 string, or `None` if not successful.

    Notes:
        - If available, this function uses the faster `pybase64` module; otherwise, it falls back to the standard `base64` module.
    """
    # parse arguments
    from nimbro_api.utility.logger import Logger
    assert_type_value(obj=logger, type_or_value=[Logger, None], name="argument 'logger'")
    assert_type_value(obj=obj, type_or_value=bytes, name="argument 'obj'")
    assert_type_value(obj=name, type_or_value=str, name="argument 'name'")

    # encode object

    tic = time.perf_counter()

    if logger is not None:
        logger.debug(f"Encoding {name} as Base64.")

    try:
        if PYBASE64_AVAILABLE:
            obj_str = pybase64.b64encode(obj).decode('ascii')
        else:
            if logger is not None:
                logger.debug(f"Using slow 'base64' module to encode {name}. Install 'pybase64' (pip install pybase64) to speed this up!", once=True)
            obj_str = base64.b64encode(obj).decode('ascii')
    except Exception as e:
        success = False
        message = f"Failed to encode {name} as Base64: {repr(e)}"
        obj_str = None
    else:
        success = True
        message = f"Encoded {name} as Base64 in '{time.perf_counter() - tic:.3f}s'."

    return success, message, obj_str

def decode_b64(string, *, name="object", logger=None):
    """
    Decode a Base64 encoded ASCII string to bytes.

    Args:
        string (str):
            Base64 encoded ASCII string.
        name (str, optional):
            Descriptive name for logging. Defaults to "object".
        logger (nimbro_api.utility.logger.Logger | None, optional):
            If provided, logs status/error messages. Defaults to `None`.

    Raises:
        UnrecoverableError: If input arguments are invalid (excluding invalid Base64 string or decoding error).

    Returns:
        tuple[bool, str, bytes | None]: A tuple containing:
            - bool: `True` if the operation succeeded, `False` otherwise.
            - str: A descriptive message about the operation result.
            - bytes | None: Decoded object as bytes, or `None` if not successful.

    Notes:
        - If available, this function uses the faster `pybase64` module; otherwise, it falls back to the standard `base64` module.
    """
    # parse arguments
    from nimbro_api.utility.logger import Logger
    assert_type_value(obj=logger, type_or_value=[Logger, None], name="argument 'logger'")
    assert_type_value(obj=string, type_or_value=str, name="argument 'string'")
    assert_type_value(obj=name, type_or_value=str, name="argument 'name'")

    # decode object

    tic = time.perf_counter()

    if logger is not None:
        logger.debug(f"Decoding {name} from Base64.")

    try:
        if PYBASE64_AVAILABLE:
            obj_bytes = pybase64.b64decode(string, altchars=None, validate=True)
        else:
            if logger is not None:
                logger.debug(f"Using slow 'base64' module to decode {name}. Install 'pybase64' (pip install pybase64) to speed this up!", once=True)
            obj_bytes = base64.b64decode(string, altchars=None, validate=True)
    except Exception as e:
        success = False
        message = f"Failed to decode {name} from Base64: {repr(e)}"
        obj_bytes = None
    else:
        success = True
        message = f"Decoded {name} from Base64 in '{time.perf_counter() - tic:.3f}s'."

    return success, message, obj_bytes

def parse_image_b64(image, *, logger=None):
    """
    Parse an image from various formats into a Base64 encoded ASCII string.

    Args:
        image (str | bytes):
            The image to parse. Can be raw bytes, a Base64 encoded string, a URL, or a local file path.
        logger (nimbro_api.utility.logger.Logger | None, optional):
            If provided, logs status/error messages. Defaults to `None`.

    Raises:
        UnrecoverableError: If input arguments are invalid.

    Returns:
        tuple[bool, str, str, str | None]: A tuple containing:
            - bool: `True` if the operation succeeded, `False` otherwise.
            - str: A descriptive message about the operation result.
            - str: Image as Base64 encoded ASCII string, or `None` if not successful.
            - str | None: Original path or URL of the image if applicable and successful, otherwise `None`.
    """
    # parse arguments
    from nimbro_api.utility.logger import Logger
    assert_type_value(obj=logger, type_or_value=[Logger, None], name="argument 'logger'")
    assert_type_value(obj=image, type_or_value=[str, bytes], name="argument 'image'", logger=logger)

    # parse image
    image_path = None
    if isinstance(image, bytes):
        success, message, image_file = encode_b64(obj=image, name="image", logger=logger)
        if logger is not None and success:
            logger.debug(message)
    elif is_base64(image):
        success = True
        message = "Provided image is Base64-encoded."
        image_file = image
        if logger is not None:
            logger.debug(message)
    elif is_url(image):
        if logger is not None:
            logger.debug("Provided image is a valid URL.")
        success, message, image_file = download_file(url=image, name="image", retry=1, logger=logger)
        if success:
            if logger is not None:
                logger.info(message)
            success, message, image_file = encode_b64(obj=image_file, name="image", logger=logger)
            if success:
                if logger is not None:
                    logger.debug(message)
                image_path = image
    elif os.path.exists(image):
        success, message, image_file = read_as_b64(file_path=image, name="image", logger=logger)
        if success:
            if logger is not None:
                logger.debug(message)
            image_path = image
    else:
        success = False
        message = f"Provided image '{image}' is neither Base64-encoded, a valid local path, or a web URL."
        image_file = image

    return success, message, image_file, image_path

def parse_audio_bytes(audio, *, logger=None):
    """
    Parse audio from various formats into raw bytes.

    Args:
        audio (str | bytes):
            The audio to parse. Can be raw bytes, a Base64 encoded string, a URL, or a local file path.
        logger (nimbro_api.utility.logger.Logger | None, optional):
            If provided, logs status/error messages. Defaults to `None`.

    Raises:
        UnrecoverableError: If input arguments are invalid.

    Returns:
        tuple[bool, str, bytes | None, str | None]: A tuple containing:
            - bool: `True` if the operation succeeded, `False` otherwise.
            - str: A descriptive message about the operation result.
            - bytes | None: Audio data as bytes, or `None` if not successful.
            - str | None: Original path, URL, or Base64 string of the audio if applicable and successful, otherwise `None`.
    """
    # parse arguments
    from nimbro_api.utility.logger import Logger
    assert_type_value(obj=logger, type_or_value=[Logger, None], name="argument 'logger'")
    assert_type_value(obj=audio, type_or_value=[str, bytes], name="argument 'audio'")

    audio_file, audio_path = None, None
    if isinstance(audio, str):
        if is_base64(audio):
            if logger is not None:
                logger.debug("Provided audio is Base64-encoded.")
            success, message, audio_file = decode_b64(string=audio, name="audio", logger=logger)
            if success:
                audio_path = audio
                if logger is not None:
                    logger.debug(message)
        elif is_url(audio):
            if logger is not None:
                logger.debug("Provided audio is a valid URL.")
            success, message, audio_file = download_file(url=audio, name="audio", retry=1, logger=logger)
            if success:
                audio_path = audio
                if logger is not None:
                    logger.info(message)
        elif os.path.exists(audio):
            if os.path.isfile(audio):
                try:
                    with open(audio, "rb") as f:
                        audio_file = f.read()
                except Exception as e:
                    success = False
                    message = f"Failed to read audio-file '{audio}': {repr(e)}"
                    audio_file = None
                else:
                    success = True
                    message = f"Read audio-file '{audio}'."
                    audio_path = os.path.abspath(audio)
            else:
                success = False
                message = f"Provided path '{audio}' is not a valid file."
        else:
            success = False
            message = f"Provided audio '{audio}' is neither Base64-encoded, a valid local path, or a web URL."
    else:
        success = True
        message = "Provided audio as bytes."
        audio_file = audio

    return success, message, audio_file, audio_path

def get_cache_location():
    """
    Get the directory path used for caching by this package.

    Returns:
        str: The absolute path to the cache directory.
            If set the environment variable "NIMBRO_API_HOME" is used.
            Otherwise "~/.cache/nimbro_api" is used.
    """
    path = os.environ.get("NIMBRO_API_HOME")
    if path is None:
        path = os.path.join("~", ".cache", "nimbro_api")
        path = os.path.expanduser(path)
    return path

def acquire_lock(path):
    """
    Acquire an exclusive, blocking file lock.

    Args:
        path (str):
            The filesystem path to the lock file.

    Returns:
        io.IOBase | int: An opaque resource handle that must be passed to 'release_lock()'.
            On Windows, this is a file object; on Unix-like systems, this is a file descriptor.

    Notes:
        - This function is cross-platform, using `msvcrt` on Windows and `fcntl` on Unix.
        - The function blocks until the lock is acquired.
        - Active handles are tracked in the global 'LOCK_HANDLES' set for monitoring.
    """
    # parse arguments
    assert_type_value(obj=path, type_or_value=str, name="argument 'path'")

    os.makedirs(os.path.dirname(path), exist_ok=True)

    if IS_WINDOWS:
        f = open(path, "a+b")
        f.seek(0)
        fd = f.fileno()
        while True:
            try:
                msvcrt.locking(fd, msvcrt.LK_LOCK, 1)
                break
            except OSError:
                time.sleep(0.05)
        resource = f
    else:
        fd = os.open(path, os.O_RDWR | os.O_CREAT, 0o666)
        fcntl.flock(fd, fcntl.LOCK_EX)
        resource = fd

    with LOCK_HANDLES_LOCK:
        LOCK_HANDLES.add(resource)

    return resource

def release_lock(resource):
    """
    Release an exclusive file lock and close the resource.

    Args:
        resource (io.IOBase | int):
            The opaque resource handle previously acquired via 'acquire_lock()'.

    Raises:
        UnrecoverableError: If the 'resource' was not found in the active locks registry or has already been released.

    Notes:
        - This function is cross-platform, using `msvcrt` on Windows and `fcntl` on Unix.
        - The 'resource' is automatically removed from the global 'LOCK_HANDLES' set upon release.
    """
    # parse arguments
    assert_type_value(obj=resource, type_or_value=list(LOCK_HANDLES), name="argument 'resource'")

    with LOCK_HANDLES_LOCK:
        try:
            LOCK_HANDLES.remove(resource)
        except KeyError as e:
            raise UnrecoverableError(f"Resource key '{resource}' was not acquired or already released.") from e

    if IS_WINDOWS:
        f = resource
        fd = f.fileno()
        msvcrt.locking(fd, msvcrt.LK_UNLCK, 1)
        f.close()
    else:
        fd = resource
        fcntl.flock(fd, fcntl.LOCK_UN)
        os.close(fd)
