from nimbro_api.client import Client
from ..base.chat_completions_base import ChatCompletionsBase

default_settings = {
    'logger_severity': None,
    'logger_name': "ChatCompletions",
    'logger_info_prompt': True,
    'logger_info_prompt_threshold': 500,
    'logger_info_prompt_cutoff': 0,
    'logger_info_completion': True,
    'logger_info_completion_cutoff': 1000,
    'logger_debug_chunks': False,
    'endpoints': {
        'OpenRouter': {
            'api_flavor': "openrouter",
            'api_url': "https://openrouter.ai/api/v1/chat/completions",
            'models_url': "https://openrouter.ai/api/v1/models",
            'key_type': "environment",
            'key_value': "OPENROUTER_API_KEY"
        },
        'OpenAI': {
            'api_flavor': "openai",
            'api_url': "https://api.openai.com/v1/chat/completions",
            'models_url': "https://api.openai.com/v1/models",
            'key_type': "environment",
            'key_value': "OPENAI_API_KEY"
        },
        'Mistral': {
            'api_flavor': "mistral",
            'api_url': "https://api.mistral.ai/v1/chat/completions",
            'models_url': "https://api.mistral.ai/v1/models",
            'key_type': "environment",
            'key_value': "MISTRAL_API_KEY"
        },
        'AIS': {
            'api_flavor': "vllm",
            'api_url': "https://api-code.ais.uni-bonn.de/v1/chat/completions",
            'models_url': "https://api-code.ais.uni-bonn.de/v1/models",
            'key_type': "environment",
            'key_value': "AIS_API_KEY"
        },
        'vLLM': {
            'api_flavor': "vllm",
            'api_url': "http://localhost:8000/v1/chat/completions",
            'models_url': "http://localhost:8000/v1/models",
            'key_type': "environment",
            'key_value': "VLLM_API_KEY"
        }
    },
    'endpoint': "OpenRouter",
    'model': "google/gemini-3-flash-preview",
    'validate_model': 3600,
    'temperature': 1.0,
    'top_p': 1.0,
    'max_tokens': 5000,
    'presence_penalty': 0.0,
    'frequency_penalty': 0.0,
    'reasoning_effort': "none",
    'download_image': False,
    'download_audio': True,
    'download_video': False,
    'download_file': True,
    'stream': False,
    'max_tool_calls': 1,
    'correction': True,
    'timeout_connect': 10.0,
    'timeout_read': 20.0,
    'timeout_chunk_first': 10.0,
    'timeout_chunk_next': 5.0,
    'timeout_completion': 30.0,
    'request_safeguard': True,
    'parser': ["string_strip.py"],
    'retry': 2
}

class ChatCompletions(Client):
    """
    This is an implementation of OpenAI's v1 Chat Completions API (https://platform.openai.com/docs/api-reference/chat),
    with generalization across API providers deviating from the standard (OpenRouter, OpenAI, Mistral, vLLM), sensible
    default settings and behaviors throughout, and extensive capabilities for configuring endpoints, models, context
    window, and tool definitions, as well as managing connections, correcting errors, parsing responses, and logging.
    """

    def __init__(self, settings=None, **kwargs):
        """
        Create an Client implementing OpenAI's v1 Chat Completions API (https://platform.openai.com/docs/api-reference/chat).
        """
        super().__init__(client_base=ChatCompletionsBase, settings=settings, default_settings=default_settings, **kwargs)

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
            logger_info_prompt (bool | int):
                Logger behavior regarding an INFO log with each call of `prompt()`´:
                - Use `False` to deactivate this log.
                - Use `True` to log general info and all messages in context.
                - Use 0 to log general info without any context messages.
                - Use a positive `int` to restrict the number of context messages shown starting with the first.
                - Use a negative `int` to restrict the number of context messages shown ending with the last.
            logger_info_prompt_threshold (int):
                Number of characters above which logs of context messages related to 'logger_info_prompt' are cutoff.
            logger_info_prompt_cutoff (int):
                Number of characters shown of context messages exceeding 'logger_info_prompt_threshold'.
            logger_info_completion (bool):
                Emit an INFO log showing the contents of each received assistant response.
            logger_debug_chunks (bool):
                Emit DEBUG logs showing the raw contents received from the API.
            endpoints (dict[dict]):
                Endpoint definitions pointing to providers of the v1 Chat Completions API.
                - Each endpoint must be a dictionary (`dict`), with the required keys "api_flavor",
                  "api_url", "key_type", and "key_value", and the optional key "models_url".
                - The value of 'key_type' must be either "environment" or "plain".
                - All values must be non-empty strings (`str`).
            endpoint (str | dict):
                Name of the endpoint to be used from the list of defined 'endpoints'.
                Pass an endpoint definition (`dict`) to automatically add/update the definition and select it.
            model (str):
                Name of the model used (see `get_models()`).
            validate_model (float | int | bool):
                Validate the set 'model' is available with the Models API of the set 'endpoint' when using `prompt()`:
                - Use `float` or `int` to set the time in seconds permitted for cached responses to be reused before requesting a new one.
                - Use `True` to force validation, or `False` to deactivate it.
            temperature (float | int):
                Model temperature parameter (from 0.0 to 1.5).
            top_p (float | int):
                Model top_p parameter (from 0.0 to 2.0).
            max_tokens (int):
                Maximum number of tokens generated per assistant response.
            presence_penalty (float | int):
                Model presence_penalty parameter (from -2.0 to +2.0).
            frequency_penalty (float | int):
                Model frequency_penalty parameter (from -2.0 to +2.0).
            reasoning_effort (str):
                Model reasoning effort in ["", "none", "minimal", "low", "medium", "high"].
            download_image (bool):
                Download all images in context messages provided as URLs when using `set_context()` or `prompt()`.
            download_audio (bool):
                Download all audios in context messages provided as URLs when using `set_context()` or `prompt()`.
            download_video (bool):
                Download all videos in context messages provided as URLs when using `set_context()` or `prompt()`.
            download_file (bool):
                Download all files in context messages provided as URLs when using `set_context()` or `prompt()`.
            stream (bool):
                Stream responses.
            max_tool_calls (int | None):
                Maximum number of tool calls permitted in a single assistant response. Use `None` to put no restriction.
            correction (bool):
                Attempt automatic correction routines after invalid assistant responses before returning a failure:
                - Invalid assistant messages and correction prompts are only added to context temporarily and removed afterwards.
            timeout_connect (float | int | None):
                Time in seconds waited for connecting to the 'endpoint', or `None` to wait indefinitely.
            timeout_read (float | int | None):
                Time in seconds waited for receiving a response from the 'endpoint', or `None` to wait indefinitely.
            timeout_chunk_first (float | int | None):
                Time in seconds waited until the first response chunk is received when using 'stream', or `None` to wait indefinitely.
            timeout_chunk_next (float | int | None):
                Time in seconds waited until the next response chunk is received when using 'stream', or `None` to wait indefinitely.
            timeout_completion (float | int | None):
                Time in seconds waited until the complete response is received, or `None` to wait indefinitely.
            request_safeguard (bool):
                Wait until the thread posting the API request terminated before terminating prompt().
            parser (list | str):
                List of parsers executed sequentially:
                - See template for using or creating parsers: nimbro_api/api/openai/parser/template.py
                - Parsers can be referenced either as path or as a name of a default parser in "nimbro_api/api/openai/parser".
                - Use `str` to automatically set up a list with one completion parser.
            retry (bool | int):
                Defines retry behavior in failure cases, if the cause is eligible for retry:
                - If `True`, retries indefinitely. If `False`, failure is returned immediately.
                - Use a positive integer (`int`) to permit a specific number of retry attempts.

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
        return self._base.wrap(1, self._base.get_api_key)

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

    def prompt(self, text=None, reset_context=False, response_type="auto", **kwargs):
        """
        Request an assistant response.

        Args:
            text (str | dict | list[str|dict] | None, optional):
                Content added to the context window:
                - Use `str` to automatically format a context message as either user text or a tool response,
                  depending on whether a tool response is awaited (see `get_awaited_tools()`).
                - Use `dict` to pass a correctly formatted message directly (see `set_context()`).
                - Use `list` to pass multiple messages as `str` (auto-formatted) or `dict`.
                - Use `None` to not add anything to the context. Defaults to `None`.
            reset_context (bool, optional):
                Reset the context window before adding `text` to it. Defaults to `False`.
            response_type (str, optional):
                Determine if and which type of assistant response is being triggered:
                - "none": Apply `text` and `reset_context` without triggering an assistant response.
                - "auto": The assistant response is not restricted.
                - "text": The assistant must respond with text only.
                - "json": The assistant must respond with valid JSON, which gets encoded (`any`) in the 'text' response.
                - "always": The assistant must respond with tool call(s) only.
                - The name of a defined tool: The assistant must respond with tool call of this tool.
                Defaults to `auto`.
            **kwargs:
                All settings (see `get_settings()`) can also be configured via keyword arguments from here.
                Additionally, special keyword arguments can be passes to `wrap()`:
                    persist (bool):
                        If `True`, settings applied via keyword arguments are not reverted after termination. Defaults to `False`.
                    mute (bool):
                        If `True`, all logs emitted by this function are muted. Defaults to `False`.

        Returns:
            tuple[bool, str, dict | None]:
                A tuple containing:
                - bool: `True` if the operation succeeded, `False` otherwise.
                - str: A descriptive message about the operation result.
                - dict | None: A dictionary containing the assistant response ('text', 'tools', 'reasoning'),
                  as well as 'logs', 'usage' and possibly other items, or `None` if not successful.

        Notes:
            - Any valid assistant response is automatically added to the context window, allowing `prompt()` alone to implement a chat-style client.
        """
        return self._base.wrap(1, self._base.prompt, text, reset_context, response_type, **kwargs)

    def interrupt(self):
        """
        Interrupt an ongoing assistant response.

        Args:
            **kwargs:
                All settings (see `get_settings()`) can also be configured via keyword arguments from here.
                Additionally, special keyword arguments can be passes to `wrap()`:
                    persist (bool):
                        If `True`, settings applied via keyword arguments are not reverted after termination. Defaults to `False`.
                    mute (bool):
                        If `True`, all logs emitted by this function are muted. Defaults to `False`.

        Returns:
            tuple[bool, str]: A tuple containing:
                - bool: `True` if the operation succeeded, `False` otherwise.
                - str: A descriptive message about the operation result.
        """
        return self._base.interrupt()

    def get_context(self, **kwargs):
        """
        Obtain the context window.

        Args:
            **kwargs:
                All settings (see `get_settings()`) can also be configured via keyword arguments from here.
                Additionally, special keyword arguments can be passes to `wrap()`:
                    persist (bool):
                        If `True`, settings applied via keyword arguments are not reverted after termination. Defaults to `False`.
                    mute (bool):
                        If `True`, all logs emitted by this function are muted. Defaults to `False`.

        Returns:
            tuple[bool, str, list[dict] | None]: A tuple containing:
                - bool: `True` if the operation succeeded, `False` otherwise.
                - str: A descriptive message about the operation result.
                - list[dict] | None: A list of all messages in the context window, or `None` if not successful.

        Notes:
            - The `context` response (or a subset of it) can be fed directly to `set_context(messages=context)`.
        """
        return self._base.wrap(1, self._base.get_context, **kwargs)

    def set_context(self, mode="reset", messages=None, index=0, reverse_indexing=True, **kwargs):
        """
        Configure the context window.

        Args:
            mode (str, optional):
                Determines how the context is edited:
                - "reset": Clear the entire context and replace it by `messages`, if provided.
                - "insert": Insert `messages` at the specified `index`. Use `index` 0 with `indexing_last_to_first` to append.
                - "replace": Replace context message(s) starting at `index` by `messages`, where exceeding messages are appended.
                - "remove": Remove context message at `index` from context.
                Defaults to `reset`.
            messages (list[dict] | dict | None, optional):
                New message(s) added to context according to `mode`. Defaults to `None`.
            index (int, optional):
                Points at a specific point/message in the context window according to `mode`.
            reverse_indexing (bool, optional):
                Specifies whether `index` 0 points to the last (newest) or first (oldest) message in context.
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

        Notes:
            - System message: {'role': "system", 'content': "You are a helpful assistant."}
            - User text: {'role': "user", 'content': [{'type': "text", 'text': "Tell me a fun fact!"}]}
            - User image: {'role': "user", 'content': [{'type': "image_url", 'image_url': {'url': "file path or URL or valid base64 encoding", 'detail': "high"}}]}
            - User audio: {'role': "user", 'content': [{'type': "input_audio", 'input_audio': {'data': "file path or URL or valid base64 encoding", 'format': "wav"}}]}
            - User video: {'role': "user", 'content': [{'type': "video_url", 'video_url': {'url': "file path or URL or valid base64 encoding"}}]}
            - User file: {'role': "user", 'content': [{'type': "file", 'file': {'filename': "document.pdf", 'file_data': "file path or URL or valid base64 encoding"}}]}
            - Assistant text response: {'role': "assistant", 'content': "How can I help you?"}
            - Assistant tool call: {'role': "assistant", 'content': None, 'tool_calls': [{'type': "function",
                'id': "tool_get_weather_m95WJakYZ9Hju00eWscH", 'function': {'name': "get_weather", 'arguments': r"{}"}}]}
            - Tool response: {'role': "tool", 'tool_call_id': "tool_get_weather_m95WJakYZ9Hju00eWscH", 'content': "It is rainbows and unicorns!"}
            - Official message specifications: https://platform.openai.com/docs/api-reference/chat/create?locale=en
        """
        return self._base.wrap(0, self._base.set_context, mode, messages, index, reverse_indexing, **kwargs)

    def get_tools(self, **kwargs):
        """
        Obtain all tool definitions.

        Args:
            **kwargs:
                All settings (see `get_settings()`) can also be configured via keyword arguments from here.
                Additionally, special keyword arguments can be passes to `wrap()`:
                    persist (bool):
                        If `True`, settings applied via keyword arguments are not reverted after termination. Defaults to `False`.
                    mute (bool):
                        If `True`, all logs emitted by this function are muted. Defaults to `False`.

        Returns:
            tuple[bool, str, list[dict] | None]: A tuple containing:
                - bool: `True` if the operation succeeded, `False` otherwise.
                - str: A descriptive message about the operation result.
                - list[dict] | None: A list of all tool definitions as passed to `set_tools()`, or `None` if not successful.

        Notes:
            - The `tools` response (or a subset of it) can be fed directly to `set_tools(tools=tools)`.
        """
        return self._base.wrap(1, self._base.get_tools, **kwargs)

    def set_tools(self, tools=None, **kwargs):
        """
        Configure all tool definitions.

        Args:
            tools (list[dict] | dict | None, optional):
                A list of tool definitions, where each one is a `dict` containing a 'function'.
                Passing a `dict` is interpreted as passing a `list` with a single item.
                Passing `None` is interpreted as passing an empty `list`. Defaults to `None`.
            **kwargs:
                All settings (see `get_settings()`) can also be configured via keyword arguments from here.
                Additionally, special keyword arguments can be passes to `wrap()`:
                    persist (bool):
                        If `True`, settings applied via keyword arguments are not reverted after termination. Defaults to `False`.
                    mute (bool):
                        If `True`, all logs emitted by this function are muted. Defaults to `False`.

        Returns:
            tuple[bool, str]: A tuple containing:
                - bool: `True` if the operation succeeded, `False` otherwise.
                - str: A descriptive message about the operation result.

        Notes:
            - Official tool specifications: https://platform.openai.com/docs/guides/function-calling?api-mode=chat
            - Example:
            {
                'type': "function",
                'function': {
                    'name': "get_weather",
                    'description': "Get the current weather at the users location",
                    'parameters': {
                        'type': "object",
                        'properties': {},
                        'additionalProperties': False
                    },
                    'strict': True
                }
            }
        """
        return self._base.wrap(0, self._base.set_tools, tools, **kwargs)

    def get_awaited_tools(self, **kwargs):
        """
        Obtain the IDs of all unanswered tool calls in the context window.

        Args:
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
                - list[str] | None: A list of tool call IDs that have not been responded to, or `None` if not successful.
        """
        return self._base.wrap(1, self._base.get_awaited_tools, **kwargs)
