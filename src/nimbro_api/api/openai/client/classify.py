from nimbro_api.client import Client
from ..base.classify_base import ClassifyBase

default_settings = {
    'logger_severity': None,
    'logger_name': "Classify",
    'message_results': False,
    'endpoints': {
        'AIS': {
            'api_url': "https://api-code.ais.uni-bonn.de/classify",
            'models_url': "https://api-code.ais.uni-bonn.de/v1/models",
            'key_type': "environment",
            'key_value': "AIS_API_KEY"
        },
        'vLLM': {
            'api_url': "http://localhost:8000/classify",
            'models_url': "http://localhost:8000/v1/models",
            'key_type': "environment",
            'key_value': "VLLM_API_KEY"
        }
    },
    'endpoint': "AIS",
    'mode': "messages",
    'model': None,
    'validate_model': 3600,
    'truncate_prompt_tokens': None,
    'truncation_side': None,
    'request_id': None,
    'priority': 0,
    'mm_processor_kwargs': None,
    'cache_salt': None,
    'use_activation': None,
    'add_special_tokens': False,
    'add_generation_prompt': False,
    'continue_final_message': False,
    'chat_template': None,
    'chat_template_kwargs': None,
    'media_io_kwargs': None,
    'timeout_connect': 2.0,
    'timeout_read': 5.0,
    'retry': 2
}

class Classify(Client):
    """
    This is an implementation of vLLM's Classify API (https://docs.vllm.ai/en/latest/models/pooling_models/classify),
    with sensible default settings and behaviors throughout, extensive capabilities for configuring endpoints and models,
    managing connections, converting file encodings, and logging.
    """

    def __init__(self, settings=None, **kwargs):
        """
        Create an Client implementing vLLM's Classify API (https://docs.vllm.ai/en/latest/models/pooling_models/classify).

        Args:
            settings (dict | None, optional):
                Settings initializing the object. Settings not contained are initialized to their default values.
                See the documentation of `get_settings()` for a comprehensive list of all available settings.
                Nested settings can be specified using dot-separated keys (e.g., "a.b.c" is equivalent to {"a": {"b": {"c": ...}}}).
                Use `None` to initialize with default settings. Defaults to `None`.
            **kwargs:
                All settings (see `get_settings()`) can also be initialized via keyword arguments.
                When doing so, 'settings' must be `None` or an empty `dict`.
        """
        super().__init__(client_base=ClassifyBase, settings=settings, default_settings=default_settings, **kwargs)

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
            endpoints (dict):
                Endpoint definitions mapping names (`str`) to endpoints/providers (`dict`) of the targeted API.
                - Each endpoint must be a dictionary (`dict`), with the required keys 'api_flavor',
                  'api_url', 'key_type', and 'key_value', and the optional key 'models_url'.
                - The value of 'key_type' must be either "environment" or "plain".
                - All values must be non-empty strings (`str`), except 'key_value', which may be empty.
            endpoint (str | dict):
                Name of the defined endpoint to be used from the list of keys in setting 'endpoints'.
                Pass an endpoint definition (`dict`) with the addional key 'name' to automatically add/update the setting 'endpoints' and select it.
            mode (str):
                Determines the mode in which the API is used ("input" or "messages").
                Accordingly, the prompt passed to `classify()` is forwarded to the "input" or "messages" field in the request.
            model (str, None):
                Name of the model used (see `get_models()`). If not provided, server-side model selection applies.
            validate_model (float | int | bool):
                Validate the set 'model' is available with the Models API of the set 'endpoint' when using `get_transcription()`:
                - Use `float` or `int` to set the time in seconds permitted for cached responses to be reused before requesting a new one.
                - Use `True` to force validation, or `False` to deactivate it.
                Only applicable when 'model' is provided.
            truncate_prompt_tokens (int, None):
                Truncation applied after tokenization:
                - Use -1 to truncate to the models maximum input length.
                - Use 0 to keep zero prompt tokens, i.e. an empty prompt token sequence.
                - Use a positive integer to set the maximum number of prompt tokens kept.
                - Use `None` to to not apply truncation.
            truncation_side (str, None):
                Truncation style if provided together with 'truncate_prompt_tokens' and the prompt is longer than the resolved truncation length:
                - Use "right" to keep the first tokens and drops tokens from the end.
                - Use "left" to keep the last tokens and drop tokens from the start.
                - Use `None` to apply default truncation behavior.
            request_id (str, None):
                Request identifier used by vLLM during inference. If not provided, a random UUID will be generated and used.
            priority (int):
                Configures priority scheduling, where lower values are scheduled earlier. Use 0 for highest priority.
            mm_processor_kwargs (dict, None):
                If provided, passed to the model’s Hugging Face multimodal processor.
            cache_salt (str, None):
                If provided, the prefix cache will be salted with the provided string to prevent an attacker to guess prompts in multi-user environments.
            use_activation (bool, None):
                Use `True` to apply the configured classifier/pooler activation and `False` to return unactivated classifier/pooler outputs. Use `None` to adopt the pooler default.
            add_special_tokens (bool):
                Controls whether tokenizer special tokens such as BOS/EOS may be added to the prompt.
            add_generation_prompt (bool):
                If `True`, the generation prompt will be added to the chat template. Only applicable when passing messages to `classify()`.
            continue_final_message (bool):
                If `True`, the chat will be formatted so that the final message in the chat is open-ended, without any EOS tokens.
                The model will continue this message rather than starting a new one. This allows for prefilling part of the model's response.
                Cannot be used at the same time as 'add_generation_prompt'. Only applicable when passing messages to `classify()`.
            chat_template (str, None):
                If provided, a Jinja template overriding the model/tokenizer chat template. Only applicable when passing messages to `classify()`.
            chat_template_kwargs (dict, None):
                If provided, passed to the model's template renderer. Only applicable when passing messages to `classify()`.
            media_io_kwargs (dict, None):
                If provided, passed to to model's media IO connectors. Only applicable when passing messages to `classify()`.
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
                Nested settings can be specified using dot-separated keys (e.g., "a.b.c" is equivalent to {"a": {"b": {"c": ...}}}).
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

    def classify(self, prompt, **kwargs):
        """
        Request a classification.

        Args:
            prompt (list[any] | list[str] | list[int] | list[list[int]] | str):
                Either input or messages (see setting 'mode') to be classified.
                - Messages must be provided as a list (`list[any]`) of valid message objects according to the OpenAI standard.
                - Input must be either a single text string (`str`), a list of strings strings (`list[str]`), a token-ID sequence (`list[int]`), or a list of token-ID sequences (`list[list[int]]`).
            **kwargs:
                All settings (see `get_settings()`) can also be configured via keyword arguments from here.
                Additionally, special keyword arguments can be passes to `wrap()`:
                    persist (bool):
                        If `True`, settings applied via keyword arguments are not reverted after termination. Defaults to `False`.
                    mute (bool):
                        If `True`, all logs emitted by this function are muted. Defaults to `False`.

        Returns:
            tuple[bool, str, dict | None]: A tuple containing:
                - bool: `True` if the operation succeeded, `False` otherwise.
                - str: A descriptive message about the operation result.
                - dict | None: The classification result (`dict`), where class probabilities are typically found under `result['data'][0]['probs']`, or `None` if not successful.
        """
        return self._base.wrap(1, self._base.classify, prompt, **kwargs)
