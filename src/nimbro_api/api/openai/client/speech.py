import os

from nimbro_api.client import Client
from nimbro_api.utility.io import get_cache_location
from ..base.speech_base import SpeechBase

default_settings = {
    'logger_severity': None,
    'logger_name': "Speech",
    'endpoints': {
        'OpenAI': {
            'api_url': "https://api.openai.com/v1/audio/speech",
            'models_url': "https://api.openai.com/v1/models",
            'key_type': "environment",
            'key_value': "OPENAI_API_KEY"
        },
        'AIS': {
            'api_url': "https://api-code.ais.uni-bonn.de/v1/audio/speech",
            'models_url': "https://api-code.ais.uni-bonn.de/v1/models",
            'key_type': "environment",
            'key_value': "AIS_API_KEY"
        },
        'vLLM': {
            'api_url': "http://localhost:8000/v1/audio/speech",
            'models_url': "http://localhost:8000/v1/models",
            'key_type': "environment",
            'key_value': "VLLM_API_KEY"
        }
    },
    'endpoint': "OpenAI",
    'model': "gpt-4o-mini-tts",
    'validate_model': 3600,
    'voice_presets': os.path.join(os.path.dirname(os.path.dirname(os.path.realpath(__file__))), "voice_presets.json"),
    'voice': "alloy",
    'instructions': "",
    'speed': 1.0,
    'timeout_connect': 2.0,
    'timeout_read': 5.0,
    'return_path': True,
    'return_encoding': "base64",
    'cache_folder': os.path.join(get_cache_location(), "speech"),
    'cache_file': "cache_speech.json",
    'cache_read': 30.0,
    'cache_write': True,
    'retry': 2
}

class Speech(Client):
    """
    This is an implementation of OpenAI's v1 Speech API (https://platform.openai.com/docs/api-reference/audio/createSpeech),
    with sensible default settings and behaviors throughout, extensive capabilities for configuring endpoints and models,
    managing connections, converting audio encodings, caching responses, and logging.
    """

    def __init__(self, settings=None, **kwargs):
        """
        Create an Client implementing OpenAI's v1 Speech API (https://platform.openai.com/docs/api-reference/audio/createSpeech).
        """
        super().__init__(client_base=SpeechBase, settings=settings, default_settings=default_settings, **kwargs)

    def get_settings(self, name=None):
        """
        Obtain all settings or a specific one.

        Args:
            name (str | None, optional):
                If provided, the one setting with this name is returned directly.
                Use `None` to return all settings as a dictionary. Defaults to `None`.

        Settings:
            logger_severity (str | None):
                Logger severity in ["debug", "info", "warn", "error", "fatal", "off"] (str) or `None` to adopt global process-wide severity.
            logger_name (str | None):
                Logger name shown in each log identifying this object.
            endpoints (dict[dict]):
                Endpoint definitions pointing to providers of the v1 Chat Completions API.
                - Each endpoint must be a dictionary (`dict`), with the required keys "api_url",
                  "key_type", and "key_value", and the optional key "models_url".
                - The value of 'key_type' must be either "environment" or "plain".
                - All values must be non-empty strings (`str`).
            endpoint (str | dict):
                Name of the endpoint to be used from the list of defined 'endpoints'.
                Pass an endpoint definition (`dict`) to automatically add/update the definition and select it.
            model (str):
                Name of the model used (see `get_models()`).
            validate_model (float | int | bool):
                Validate the set 'model' is available with the Models API of the set 'endpoint' when using `get_speech()`:
                - Use `float` or `int` to set the time in seconds permitted for cached responses to be reused before requesting a new one.
                - Use `True` to force validation, or `False` to deactivate it.
            voice_presets (str | None):
                Path of a file containing voice presets. Use `None` to point to default voice presets.
            voice (str):
                Voice used to generate speech. Use a supported voice directly or the name of a voice preset.
            instructions (str):
                Supply additional instructions to shape the tonality of the generated speech.
                Use an empty string (`str`) for no special instructions, or the name of a voice preset.
            speed (float):
                Speed of the generated speech in interval (from 0.25 to 4.0).
            timeout_connect (float | int | None):
                Time in seconds waited for connecting to the 'endpoint', or `None` to wait indefinitely.
            timeout_read (float | int | None):
                Time in seconds waited for receiving a response from the 'endpoint', or `None` to wait indefinitely.
            return_path (bool):
                Return speech as file path (`str`) instead of returning the file directly according to 'return_encoding' (`str` or `bytes`).
            return_encoding (str):
                File encoding when not using 'return_path'. Use "bytes" for returning `bytes` or "base64" for a Base64 encoding (`str`).
            cache_folder (str | None):
                Folder path used to store all caching-related files. Use `None` to select the default location.
            cache_file (str):
                File name used as cache-index.
            cache_read (bool | float | int):
                Attempt to retrieve requested speech from either the process-wide cache, or from disk using 'cache_folder' and 'cache_file':
                - Use `float` or `int` to set the time in seconds permitted for responses
                  cached in the process-wide cache to be reused before attempting to read cache from disk.
                - Use `True` to force reading cache from disk, or `False` to deactivate reading cache in general.
            cache_write (bool):
                Cache generated speech to disk using 'cache_folder' and 'cache_file'.
            retry (bool | int):
                Defines retry behavior in failure cases, if the cause is eligible for retry.
                If `True`, retries indefinitely. If `False`, failure is returned immediately.
                Use a positive `int` to permit a specific number of retry attempts.

        Raises:
            UnrecoverableError: If 'name' is provided and does not refer to an existing setting.

        Returns:
            any: A deep copy of the current settings (`dict`) or a single setting when providing 'name' (`any`).

        Notes:
            - See the global dictionary 'default_settings' on top of this file for defaults.
        """
        return self._base.get_settings(name)

    def set_settings(self, settings=None, **kwargs):
        """
        Configure all settings or a subset of them.

        Args:
            settings (dict | None, optional):
                New settings to apply. Settings not contained are kept.
                See the documentation of `get_settings()` for a comprehensive list of all available settings.
                Use `None` to reset all settings to their initial values. Defaults to `None`.
            **kwargs:
                All settings (see `get_settings()`) can also be configured via keyword arguments.
                When doing so, 'settings' must be `None` or an empty `dict`.

        Returns:
            tuple[bool, str]: A tuple containing:
                - bool: `True` if the operation succeeded, `False` otherwise.
                - str: A descriptive message about the operation result.
        """
        return self._base.wrap(0, self._base.set_settings, settings, **kwargs)

    def get_api_key(self, **kwargs):
        """
        Obtain the API key for the 'endpoint' currently set.

        Args:
            **kwargs:
                All settings (see `get_settings()`) can also be configured via keyword arguments from here.
                Additionally, special keyword arguments can be passes to `wrap()`:
                    persist (bool):
                        If `True`, settings applied via keyword arguments are not reverted after termination. Defaults to `False`.
                    mute (bool):
                        If `True`, all logs emitted by this function are muted. Defaults to `False`.

        Returns:
            tuple[bool, str, str | None]: A tuple containing:
                - bool: `True` if the operation succeeded, `False` otherwise.
                - str: A descriptive message about the operation result.
                - str | None: The API key for the 'endpoint' currently set, or `None` if not successful.
        """
        return self._base.wrap(1, self._base.get_api_key, **kwargs)

    def get_models(self, age=0, **kwargs):
        """
        Request all available models from the Models API of the set 'endpoint'.

        Args:
            age (float | int | None, optional):
                The time in seconds permitted for cached responses to this request being reused instead of sending a new one.
                Use `None` to always reuse cached responses regardless of age, or 0 to force a new request. Defaults to 0,
            **kwargs:
                All settings (see `get_settings()`) can also be configured via keyword arguments from here.
                Additionally, special keyword arguments can be passes to `wrap()`:
                    persist (bool):
                        If `True`, settings applied via keyword arguments are not reverted after termination. Defaults to `False`.
                    mute (bool):
                        If `True`, all logs emitted by this function are muted. Defaults to `False`.

        Returns:
            tuple[bool, str, list[str] | None]: A tuple containing:
                - bool: `True` if the operation succeeded, `False` otherwise.
                - str: A descriptive message about the operation result.
                - list[str] | None: A list of all model names served by the current 'endpoint', or `None` if not successful.

        Notes:
            - Responses by the Models API are cached for the duration of this process:
              `nimbro_api.query_cache(category="models", identifier=settings['endpoints'][settings['endpoint']]['models_url'])`
        """
        return self._base.wrap(1, self._base.get_models, age, **kwargs)

    def get_speech(self, text, **kwargs):
        """
        Request a text-to-speech generation.

        Args:
            text (str):
                The text to generate speech for.
            **kwargs:
                All settings (see `get_settings()`) can also be configured via keyword arguments from here.
                Additionally, special keyword arguments can be passes to `wrap()`:
                    persist (bool):
                        If `True`, settings applied via keyword arguments are not reverted after termination. Defaults to `False`.
                    mute (bool):
                        If `True`, all logs emitted by this function are muted. Defaults to `False`.

        Returns:
            tuple[bool, str, str | bytes | None]: A tuple containing:
                - bool: `True` if the operation succeeded, `False` otherwise.
                - str: A descriptive message about the operation result.
                - str | bytes | None: The audio-file formatted according to the settings 'return_path' and 'return_encoding', or `None` if not successful.
        """
        return self._base.wrap(1, self._base.get_speech, text, **kwargs)
