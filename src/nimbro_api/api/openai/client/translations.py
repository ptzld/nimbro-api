from nimbro_api.client import Client
from ..base.translations_base import TranslationsBase

default_settings = {
    'logger_severity': None,
    'logger_name': "Translations",
    'endpoints': {
        'OpenAI': {
            'api_url': "https://api.openai.com/v1/audio/translations",
            'models_url': "https://api.openai.com/v1/models",
            'key_type': "environment",
            'key_value': "OPENAI_API_KEY"
        },
        'AIS': {
            'api_url': "https://api-code.ais.uni-bonn.de/v1/audio/translations",
            'models_url': "https://api-code.ais.uni-bonn.de/v1/models",
            'key_type': "environment",
            'key_value': "AIS_API_KEY"
        },
        'vLLM': {
            'api_url': "http://localhost:8000/v1/audio/translations",
            'models_url': "http://localhost:8000/v1/models",
            'key_type': "environment",
            'key_value': "VLLM_API_KEY"
        }
    },
    'endpoint': "OpenAI",
    'model': "whisper-1",
    'validate_model': 3600,
    'temperature': 0.0,
    'prompt': "",
    'response_format': "json",
    'timeout_connect': 2.0,
    'timeout_read': 5.0,
    'retry': 2
}

class Translations(Client):
    """
    This is an implementation of OpenAI's v1 Translations API (https://platform.openai.com/docs/api-reference/audio/createTranslation),
    with sensible default settings and behaviors throughout, extensive capabilities for configuring endpoints and models,
    managing connections, converting audio encodings, caching responses, and logging.
    """

    def __init__(self, settings=None, **kwargs):
        """
        Create an Client implementing OpenAI's v1 Translations API (https://platform.openai.com/docs/api-reference/audio/createTranslation).
        """
        super().__init__(client_base=TranslationsBase, settings=settings, default_settings=default_settings, **kwargs)

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
                Validate the set 'model' is available with the Models API of the set 'endpoint' when using `get_translation()`:
                - Use `float` or `int` to set the time in seconds permitted for cached responses to be reused before requesting a new one.
                - Use `True` to force validation, or `False` to deactivate it.
            temperature (float | int):
                Sampling temperature for the transcription process (from 0.0 to 1.0).
            prompt (str):
                Guide the model's style in English, or use an empty string for default behavior.
            response_format (str):
                Response format in ["json", "verbose_json", "text", "srt", "vtt"].
            timeout_connect (float | int | None):
                Time in seconds waited for connecting to the 'endpoint', or `None` to wait indefinitely.
            timeout_read (float | int | None):
                Time in seconds waited for receiving a response from the 'endpoint', or `None` to wait indefinitely.
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

    def get_translation(self, audio, **kwargs):
        """
        Request an audio translation to English.

        Args:
            audio (str | bytes):
                The audio file to be transcribed as a local path, URL, Base64 encoding (all `str`), or raw `bytes`.
                Supported types: [flac, mp3, mp4, mpeg, mpga, m4a, ogg, wav, webm].
            **kwargs:
                All settings (see `get_settings()`) can also be configured via keyword arguments from here.
                Additionally, special keyword arguments can be passes to `wrap()`:
                    persist (bool):
                        If `True`, settings applied via keyword arguments are not reverted after termination. Defaults to `False`.
                    mute (bool):
                        If `True`, all logs emitted by this function are muted. Defaults to `False`.

        Returns:
            tuple[bool, str, dict | str | None]: A tuple containing:
                - bool: `True` if the operation succeeded, `False` otherwise.
                - str: A descriptive message about the operation result.
                - dict | str | None: The translation result according to the set 'response_format', or `None` if not successful.
        """
        return self._base.wrap(1, self._base.get_translation, audio, **kwargs)
