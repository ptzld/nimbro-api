import os

from nimbro_api.client import Client
from nimbro_api.utility.io import get_cache_location
from ..base.embeddings_base import EmbeddingsBase

default_settings = {
    'logger_severity': None,
    'logger_name': "Embeddings",
    'logger_info_requests': False,
    'logger_info_progress': True,
    'logger_info_cutoff': 100,
    'endpoints': {
        'OpenRouter': {
            'api_url': "https://openrouter.ai/api/v1/embeddings",
            'key_type': "environment",
            'key_value': "OPENROUTER_API_KEY"
        },
        'OpenAI': {
            'api_url': "https://api.openai.com/v1/embeddings",
            'models_url': "https://api.openai.com/v1/models",
            'key_type': "environment",
            'key_value': "OPENAI_API_KEY"
        },
        'Mistral AI': {
            'api_url': "https://api.mistral.ai/v1/embeddings",
            'models_url': "https://api.mistral.ai/v1/models",
            'key_type': "environment",
            'key_value': "MISTRAL_API_KEY"
        },
        'AIS': {
            'api_url': "https://api-code.ais.uni-bonn.de/v1/embeddings",
            'models_url': "https://api-code.ais.uni-bonn.de/v1/models",
            'key_type': "environment",
            'key_value': "AIS_API_KEY"
        },
        'vLLM': {
            'api_url': "http://localhost:8000/v1/embeddings",
            'models_url': "http://localhost:8000/v1/models",
            'key_type': "environment",
            'key_value': "VLLM_API_KEY"
        }
    },
    'endpoint': "OpenRouter",
    'model': "google/gemini-embedding-001",
    'validate_model': 3600,
    'max_batch_size': 100,
    'timeout_connect': 2.0,
    'timeout_read': 60.0,
    'cache_folder': os.path.join(get_cache_location(), "embeddings"),
    'cache_file': "cache_embeddings_index.json",
    'cache_read': 30.0,
    'cache_write': True,
    'retry': 2
}

class Embeddings(Client):
    """
    This is an implementation of OpenAI's v1 Embeddings API (https://platform.openai.com/docs/api-reference/embeddings),
    with sensible default settings and behaviors throughout, extensive capabilities for configuring endpoints and models,
    managing connections, batching large requests, caching responses, and logging.
    """

    def __init__(self, settings=None, **kwargs):
        """
        Create an Client implementing OpenAI's v1 Embeddings API (https://platform.openai.com/docs/api-reference/embeddings).
        """
        super().__init__(client_base=EmbeddingsBase, settings=settings, default_settings=default_settings, **kwargs)

    def get_settings(self, name=None):
        """
        Obtain all settings or a specific one.

        Args:
            name (str | None, optional):
                If provided, the one setting with this name is returned directly.
                Use `None` to return all settings as a dictionary. Defaults to `None`.

        Settings:
            logger_severity (int):
                Logger severity in [10, 20, 30, 40, 50] (DEBUG, INFO, WARN, ERROR, FATAL).
            logger_name (str | None):
                Logger name shown in each log identifying this object.
            logger_info_requests (bool):
                Emit an INFO log for every request sent within `get_embedding()`.
            logger_info_progress (bool):
                Emit an INFO log for every response received within `get_embedding()`.
            logger_info_cutoff (int):
                Maximum number of characters after which logs within `get_embedding()` are cutoff.
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
                Validate the set 'model' is available with the Models API of the set 'endpoint' when using `get_embedding()`:
                - Use `float` or `int` to set the time in seconds permitted for cached responses to be reused before requesting a new one.
                - Use `True` to force validation, or `False` to deactivate it.
            max_batch_size (int | None):
                Maximum batch-size for a single request to the Embeddings API. Use `None` to deactivate batching.
            timeout_connect (float | int | None):
                Time in seconds waited for connecting to the 'endpoint', or `None` to wait indefinitely.
            timeout_read (float | int | None):
                Time in seconds waited for receiving a response from the 'endpoint', or `None` to wait indefinitely.
            cache_folder (str | None):
                Folder path used to store all caching-related files. Use `None` to select the default location.
            cache_file (str):
                File name used as cache-index.
            cache_read (bool | float | int):
                Attempt to retrieve requested embeddings from either the process-wide cache, or from disk using 'cache_folder' and 'cache_file':
                - Use `float` or `int` to set the time in seconds permitted for responses
                  cached in the process-wide cache to be reused before attempting to read cache from disk.
                - Use `True` to force reading cache from disk, or `False` to deactivate reading cache in general.
            cache_write (bool):
                Cache generated embeddings to disk using 'cache_folder' and 'cache_file'.
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

    def get_embedding(self, text, **kwargs):
        """
        Request one or multiple text embeddings.

        Args:
            text (str | list[str]):
                Use `str` to generate an embedding for a single texts.
                Use `list[str]` to generate embeddings for multiple texts.
            **kwargs:
                All settings (see `get_settings()`) can also be configured via keyword arguments from here.
                Additionally, special keyword arguments can be passes to `wrap()`:
                    persist (bool):
                        If `True`, settings applied via keyword arguments are not reverted after termination. Defaults to `False`.
                    mute (bool):
                        If `True`, all logs emitted by this function are muted. Defaults to `False`.

        Returns:
            tuple[bool, str, list[float] | list[list[float]] | None]: A tuple containing:
                - bool: `True` if the operation succeeded, `False` otherwise.
                - str: A descriptive message about the operation result.
                - list[float] | list[list[float]] | None: One or multiple embeddings as `list` of `float` (`text` was `str`),
                  as `list` of `list` of `float` (`text` was `list`), or `None` if not successful.
        """
        return self._base.wrap(1, self._base.get_embedding, text, **kwargs)
