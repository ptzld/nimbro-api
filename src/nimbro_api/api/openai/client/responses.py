from nimbro_api.client import Client
from ..base.responses_base import ResponsesBase

default_settings = {
    'logger_name': "Responses",
    'logger_info_prompt': True,
    'logger_info_prompt_threshold': 500,
    'logger_info_prompt_cutoff': 0,
    'logger_info_response': True,
    'logger_debug_chunks': False,
    'endpoints': {
        'OpenAI': {
            'api_flavor': "openai",
            'api_url': "https://api.openai.com/v1/responses",
            'models_url': "https://api.openai.com/v1/models",
            'key_type': "environment",
            'key_value': "OPENAI_API_KEY"
        },
        'OpenRouter': {
            'api_flavor': "openrouter",
            'api_url': "https://openrouter.ai/api/v1/responses",
            'models_url': "https://openrouter.ai/api/v1/models",
            'key_type': "environment",
            'key_value': "OPENROUTER_API_KEY"
        },
        'vLLM': {
            'api_flavor': "vllm",
            'api_url': "http://localhost:8000/v1/responses",
            'models_url': "http://localhost:8000/v1/models",
            'key_type': "environment",
            'key_value': "VLLM_API_KEY"
        }
    },
    'endpoint': "OpenAI",
    'model': "gpt-6-luna",
    'validate_model': 3600,
    'service_tier': None,
    'instructions': None,
    'parallel_tool_calls': True,
    'reasoning_effort': "",
    'reasoning_summary': None,
    'send_reasoning': True,
    'verbosity': None,
    'max_tokens': 10000,
    'temperature': None,
    'top_p': None,
    'presence_penalty': None,
    'frequency_penalty': None,
    'top_k': None,
    'stop': None,
    'truncation': "disabled",
    'include': ["reasoning.encrypted_content"],
    'download_image': False,
    'download_audio': True,
    'download_video': False,
    'download_file': True,
    'stream': False,
    'timeout_connect': 10.0,
    'timeout_read': 20.0,
    'timeout_chunk_first': 10.0,
    'timeout_chunk_next': 5.0,
    'timeout_response': 30.0
}

class Responses(Client):
    """
    Generate responses with conversation context, tools, and multimodal input.

    Supports streaming, reasoning, and structured output.
    Context is a list of Responses input and output dictionaries stored by this client instance.
    `prompt()` updates it after success.

    Notes:
        - API references: https://developers.openai.com/api/reference/resources/responses/methods/create,
          https://openrouter.ai/docs/api/reference/responses/overview,
          https://docs.vllm.ai/en/latest/serving/online_serving/, and https://www.openresponses.org/reference.
        - Endpoint flavors select provider-specific routing and service tiers.
        - Requests disable server-side response storage. The client sends full local history on each prompt.
          Select 'endpoint' as "OpenRouter" with a model ID such as "openai/gpt-6-luna".
        - For vLLM, select 'endpoint' as "vLLM" with the served model ID and set 'VLLM_API_KEY'.
          For an unauthenticated server, supply an endpoint with 'key_type' set to "plain" and 'key_value' set to "".
        - Open Responses is a specification, not a hosted provider. Supply an endpoint definition for your server
          with 'api_flavor' set to "openresponses", its 'api_url', 'name', 'key_type', and 'key_value'.
        - Generation preserves output dictionaries, including encrypted reasoning, tool calls, and assistant 'phase'.
        - Hosted tools run on the provider. Application tools must be executed by the caller.
          Configure tools with `set_tools()`. Availability depends on the provider and model.
        - Reasoning effort support depends on the model. Values are forwarded without translation.
          The summary preference is omitted for vLLM. Other flavors receive it unchanged.
        - Switching models or endpoints preserves context, including native reasoning items and their summaries.
          Outgoing reasoning is adapted on a best-effort basis. See 'send_reasoning' in `get_settings()`
          for compatibility adjustments and disabling reasoning replay for unsupported destinations.
    """

    def __init__(self, settings=None, **kwargs):
        """
        Create a client implementing the Responses API.

        Args:
            settings (dict | None, optional):
                Settings initializing the object. Settings not contained are initialized to their default values.
                See the documentation of `get_settings()` for a comprehensive list of all available settings.
                Nested settings can be specified using dot-separated keys (e.g., `{'a.b.c': ...}` is equivalent to `{'a': {'b': {'c': ...}}}`).
                Use `None` to initialize with default settings. Defaults to `None`.
            **kwargs:
                All settings (see `get_settings()`) can also be initialized via keyword arguments.
                When doing so, 'settings' must be `None` or an empty `dict`.

        Raises:
            UnrecoverableError: If initialization fails, including invalid settings.
        """
        super().__init__(client_base=ResponsesBase, settings=settings, default_settings=default_settings, **kwargs)

    def get_settings(self, name=None):
        """
        Obtain all settings or a specific one.

        Args:
            name (str | None, optional):
                If provided, the one setting with this name is returned directly.
                Use `None` to return all settings as a dictionary. Defaults to `None`.

        Settings:
            logger_info_prompt (bool | int):
                Logger behavior regarding an INFO log with each call of `prompt()`:
                - Use `False` to deactivate this log.
                - Use `True` to log general info and all messages in context.
                - Use 0 to log general info without any context messages.
                - Use a positive `int` to restrict the number of context messages shown ending with the last.
                - Use a negative `int` to restrict the number of context messages shown starting with the first.
            logger_info_prompt_threshold (int):
                Number of characters above which content is shortened in prompt logs. Must be positive.
                Text is shortened to this length, followed by an ellipsis. Media URLs/data use 'logger_info_prompt_cutoff' instead.
            logger_info_prompt_cutoff (int):
                Number of characters shown for media URLs/data exceeding 'logger_info_prompt_threshold', followed by an ellipsis. Must be non-negative.
                Use `0` to show only an ellipsis. Does not affect text content.
            logger_info_response (bool):
                Emit an INFO log showing the contents of each received assistant response.
            logger_debug_chunks (bool):
                Emit DEBUG logs showing the raw contents received from the API.
            endpoints (dict):
                Endpoint definitions mapping names (`str`) to endpoints/providers (`dict`) of the targeted API.
                - The mapping must contain at least one endpoint and all names must contain a non-whitespace character.
                - Each endpoint must be a dictionary (`dict`), with the required keys 'api_flavor',
                  'api_url', 'key_type', and 'key_value', and the optional key 'models_url'.
                - The value of 'key_type' must be either "environment" or "plain".
                - All values must be non-empty strings (`str`), except 'key_value', which may be empty.
                - Use 'key_type'="plain" and 'key_value'="" for an unauthenticated server.
            endpoint (str | dict):
                Name of the defined endpoint to be used from the list of keys in setting 'endpoints'.
                Pass an endpoint definition (`dict`) with the additional key 'name' (a string containing a non-whitespace character) to automatically add/update the setting 'endpoints' and select it.
            model (str):
                Name of the model used (see `get_models()`).
            validate_model (float | int | bool):
                Numeric cache ages must be non-negative. Use `0` to force a fresh check.
                Checks are skipped when the selected endpoint has no 'models_url'.
                Validate the set 'model' is available with the Models API of the set 'endpoint' when using `prompt()`:
                - Use `float` or `int` to set the time in seconds permitted for cached responses to be reused before requesting a new one.
                - Use `True` to force validation, or `False` to deactivate it.
            service_tier (str | None):
                Choose a service preference:
                - `None`: Keep the endpoint's defaults without sending a service preference.
                - "price": Prefer lower cost.
                - "latency": Prefer a faster initial response.
                - "throughput": Prefer faster token generation.
                Support depends on the provider, model, and account. vLLM ignores this setting.
                Speed preferences opt into premium tiers where available and may cost more.
                The distinction between "latency" and "throughput" applies only to OpenRouter.
            instructions (str | None):
                System or developer instructions sent on every generation and compaction request.
                Use `None` to omit the field or "" to send empty instructions.
                Sent separately from context. System/developer context messages may also be supplied and are sent alongside this field.
                Avoid duplicating or conflicting instructions across the two mechanisms.
            parallel_tool_calls (bool):
                Allow multiple tool calls in one response. Use `False` to disable.
                Always forwarded to the provider. Does not execute tools or enforce a client-side tool-call limit.
            reasoning_effort (str):
                Model reasoning effort in ["", "none", "minimal", "low", "medium", "high", "xhigh", "max"].
                Use an empty string to not pass this parameter explicitly and let the API choose a default instead.
            reasoning_summary (str | None):
                Request a readable reasoning summary in ["auto", "concise", "detailed"]. Use `None` to omit the preference.
                Returned summaries remain part of native reasoning items in both the result and replayed context.
            send_reasoning (bool):
                Send reasoning history with generation and compaction requests, applying compatibility adjustments.
                Use `False` to omit reasoning items from requests without removing them from local history or results.
                Does not disable the model's own reasoning or change 'reasoning_effort' or 'reasoning_summary'.
            verbosity (str | None):
                Output detail level in ["low", "medium", "high"]. Use `None` to omit the preference.
            max_tokens (int):
                Maximum number of tokens generated per assistant response. Must be positive.
            temperature (float | int | None):
                Model temperature parameter (from 0.0 to 2.0 inclusive). Use `None` to omit the parameter.
            top_p (float | int | None):
                Model top_p parameter (from 0.0 to 1.0 inclusive). Use `None` to omit the parameter.
            presence_penalty (float | int | None):
                Model presence_penalty parameter (from -2.0 to +2.0 inclusive). Use `None` to omit the parameter.
            frequency_penalty (float | int | None):
                Model frequency_penalty parameter (from -2.0 to +2.0 inclusive). Use `None` to omit the parameter.
            top_k (int | None):
                Sampling candidate limit. Use a positive integer to limit, `0` to disable,
                or `None` to omit.
            stop (str | list[str] | None):
                OpenRouter/vLLM stop sequences, with at most four sequences.
                Use `None` to omit or `[]` for no stop sequences. Strings must be nonempty.
            truncation (str):
                Use "auto" to permit server-side context truncation or "disabled" to fail on oversized input.
            include (list[str] | None):
                Extra output fields requested during generation. Accepted values:
                    "reasoning.encrypted_content": Encrypted reasoning state for subsequent requests.
                    "file_search_call.results": Retrieved document passages and search-result metadata.
                    "code_interpreter_call.outputs": Provider-run code outputs, such as logs and images.
                    "web_search_call.results": Web-search result details.
                    "web_search_call.action.sources": Sources consulted during web search.
                    "output.encrypted_content": Provider-specific encrypted output.
                Use `None` to omit or `[]` to request no extra fields. Unsupported fields are omitted per flavor
                without changing this setting. Returned output items, including their extra fields, remain in context.
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
            timeout_connect (float | int | None):
                Time in seconds waited for connecting to the 'endpoint', or `None` to wait indefinitely.
                Numeric values must be positive and finite.
            timeout_read (float | int | None):
                Time in seconds waited for receiving a response from the 'endpoint', or `None` to wait indefinitely.
                Numeric values must be positive and finite.
            timeout_chunk_first (float | int | None):
                Time in seconds waited until the first response chunk is received when using 'stream', or `None` to wait indefinitely.
                Numeric values must be positive and finite.
            timeout_chunk_next (float | int | None):
                Time in seconds waited until the next response chunk is received when using 'stream', or `None` to wait indefinitely.
                Numeric values must be positive and finite.
            timeout_response (float | int | None):
                Time in seconds waited until the complete response is received, or `None` to wait indefinitely.
                Numeric values must be positive and finite.

        Raises:
            UnrecoverableError: If 'name' is provided and does not refer to an existing setting.

        Returns:
            object: A deep copy of the current settings (`dict`) or a single setting when providing 'name'.

        Notes:
            - See `nimbro_api.Client.get_settings()` for inherited settings.
            - See the global dictionary 'default_settings' on top of this file for client-specific defaults.
            - 'max_tokens' includes reasoning tokens and is sent as 'max_output_tokens'.
            - Endpoint 'api_flavor' is "openai", "openrouter", "vllm", or "openresponses".
              Model IDs must contain a non-whitespace character. Supported models and options depend on the endpoint.
            - Numeric model-validation cache ages must be finite.
              Model catalogs do not guarantee support for the Responses API or individual options.
            - 'verbosity' is forwarded for OpenAI, OpenRouter, and Open Responses, and omitted for vLLM.
            - 'top_k' is forwarded for OpenRouter and vLLM, and omitted for other flavors.
            - 'reasoning_summary' is omitted for vLLM.
            - With 'send_reasoning' enabled, OpenRouter-marked reasoning sent to OpenAI retains readable
              summaries and converts readable reasoning text into summary entries. Foreign identifiers,
              signatures, ciphertext, and metadata are omitted. Items without readable text are omitted.
              Encrypted reasoning generated by this client on OpenAI receives an OpenAI format marker when sent to
              OpenRouter. Native local history and results remain unchanged. Origin tracking lasts while
              the item remains in this client's history. Arbitrary imported ciphertext is not identified
              automatically. Use 'send_reasoning=False' if a destination rejects imported reasoning.
            - All flavors accept "reasoning.encrypted_content" in 'include'. OpenAI, OpenRouter, and vLLM
              also accept file-search results and code-interpreter outputs.
              Web-search results and sources are OpenAI-only. "output.encrypted_content" is OpenRouter-only.
              Schema support does not guarantee availability for every model or server deployment.
            - 'logger_debug_chunks' logs SSE event dictionaries when streaming and full response contents otherwise.
            - Streaming generation requests disable event obfuscation padding for OpenAI and Open Responses.
            - 'timeout_response' starts with each generation request after credential and model validation.
              It also limits each compaction request. Response and chunk timeouts permit retries.
            - Media download settings also apply to `compact()`. Disabled downloads forward URLs.
              Image paths, bytes, and Base64 in 'input_image.image_url' are always embedded as data URLs.
              Paths and bytes in 'input_audio.input_audio.data' are Base64 encoded. Supply 'format', such as "wav" or "mp3".
              No audio transcoding is performed. Video paths and bytes in 'input_video.video_url' support MP4 and WebM.
              File paths and bytes in 'input_file.file_data' are Base64 encoded. Supply 'filename' for file bytes.
              Audio/video parts and forwarded URLs require support from the provider and model.
            - 'presence_penalty' and 'frequency_penalty' are omitted from OpenAI requests.
            - A full settings snapshot consolidates the selected endpoint into 'endpoint', including its 'name', and omits 'endpoints'.
            - Optional request fields set to `None` are omitted rather than sent as JSON null.
            - Streaming chunks are complete SSE event dictionaries. Keepalive comments and partial event bytes
              do not reset chunk timeouts. The first-chunk timer includes connection and response-header waits.
              Read and overall response timeouts remain active while streaming. The earliest limit wins.
            - Wrapper retries start a new response deadline per attempt. Media preparation is outside that deadline.
        """
        return self._base.get_settings(name)

    def set_settings(self, settings=None, **kwargs):
        """
        Configure all settings or a subset of them.

        Args:
            settings (dict | None, optional):
                New settings to apply. Settings not contained are kept.
                See the documentation of `get_settings()` for a comprehensive list of all available settings.
                Nested settings can be specified using dot-separated keys (e.g., `{'a.b.c': ...}` is equivalent to `{'a': {'b': {'c': ...}}}`).
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
                Additionally, special keyword arguments can be passed to `wrap()`:
                    persist (bool):
                        If `True`, settings applied via keyword arguments are not reverted after termination. Defaults to `False`.
                    mute (bool):
                        If `True`, all logs emitted by this function are muted. Defaults to `False`.

        Returns:
            tuple[bool, str, str | None]: A tuple containing:
                - bool: `True` if the operation succeeded, `False` otherwise.
                - str: A descriptive message about the operation result.
                - str | None: The selected endpoint's API key, or `None` if not successful.
        """
        return self._base.wrap(1, self._base.get_api_key, **kwargs)

    def get_models(self, age=0, **kwargs):
        """
        Request available model IDs from the selected endpoint's Models API.

        Args:
            age (float | int | None, optional):
                Maximum cache age in seconds. Numeric values must be non-negative.
                Use `0` to force a request, a positive value to reuse sufficiently recent cached responses, or `None` to allow any cache age.
                A cache miss still requires a request. Defaults to `0`.
            **kwargs:
                All settings (see `get_settings()`) can also be configured via keyword arguments from here.
                Additionally, special keyword arguments can be passed to `wrap()`:
                    persist (bool):
                        If `True`, settings applied via keyword arguments are not reverted after termination. Defaults to `False`.
                    mute (bool):
                        If `True`, all logs emitted by this function are muted. Defaults to `False`.

        Returns:
            tuple[bool, str, list[str] | None]: A tuple containing:
                - bool: `True` if the operation succeeded, `False` otherwise.
                - str: A descriptive message about the operation result.
                - list[str] | None: Model IDs, or `None` on failure, including an endpoint without 'models_url'.

        Notes:
            - Results use the process-wide "models" cache keyed by 'models_url'.
        """
        return self._base.wrap(1, self._base.get_models, age, **kwargs)

    def prompt(self, text=None, reset_context=False, response_type="auto", **kwargs):
        """
        Request an assistant response using this client's context.

        Args:
            text (str | dict | list[str | dict] | None, optional):
                User text, a Responses input dictionary, or multiple such items. Strings become
                outputs for pending function/custom calls in order, otherwise user messages.
                Computer/shell/patch results and approval decisions require output dictionaries.
                Multimodal messages use content parts with 'type' set to "input_text", "input_image", or "input_file".
                Media accepts provider URLs and file IDs, and local paths/bytes as described by the download settings.
                Image objects are not accepted.
                Use `None` to add nothing. Defaults to `None`.
            reset_context (bool, optional):
                Start with empty local history. Defaults to `False`.
            response_type (str | dict, optional):
                Select the response format or which configured tools the model may use. Defaults to "auto".

                String values:
                    "auto": Allow ordinary text and calls to any configured tool.
                    "none": Update context without requesting a model response.
                    "text": Request plain text without tool calls.
                    "json": Request JSON without tool calls. Return the parsed value in `result['text']`.
                    "always": Require at least one call to a configured tool.
                    Tool identifier: Require the selected configured tool.

                Function and custom tools use their configured 'name'. Other tools use their configured 'type',
                such as "web_search" or "code_interpreter". MCP tools use "mcp/<server_label>" to select a
                configured MCP server, or "mcp/<server_label>/<tool_name>" to select a tool on that server.
                Identifiers must resolve unambiguously. Reserved string values cannot be used as tool identifiers.

                Tool-subset dictionary:
                    `{'tools': ["drive", "look"], 'mode': "auto"}`
                    'tools' is a nonempty list of identifiers using the same rules as string selection.
                    The tools must already be configured through `set_tools()`.
                    'mode' is "auto" or "required". "auto" allows ordinary text or calls within the subset.
                    "required" requires at least one call within the subset. Defaults to "auto".
                    Tools outside the subset cannot be selected. Configured tool definitions remain unchanged.

                JSON-schema dictionary:
                    `{'schema': {...}, 'name': "answer", 'strict': True}`
                    'schema' is a JSON Schema dictionary describing the requested output.
                    'name' identifies the schema and defaults to "response".
                    'strict' requests strict schema adherence and defaults to `True`.
                    Optional 'description' is a string describing the requested output.
                    Request schema-constrained JSON without tool calls and return the parsed value in
                    `result['text']`. The client does not independently validate the value against the schema.

                Dictionaries must specify either 'tools' or 'schema', not both. Other keys are rejected.
                Provider request fields are derived from this configuration, not forwarded unchanged.
                Supply all pending synchronous tool results before generating again.
                Calls with 'async' set to `True` remain pending but do not block generation.
            **kwargs:
                All settings (see `get_settings()`) can also be configured via keyword arguments from here.
                Additionally, special keyword arguments can be passed to `wrap()`:
                    persist (bool):
                        If `True`, settings applied via keyword arguments are not reverted after termination. Defaults to `False`.
                    mute (bool):
                        If `True`, all logs emitted by this function are muted. Defaults to `False`.

        Returns:
            tuple[bool, str, dict | None]: A tuple containing:
                - bool: `True` if successful, `False` otherwise.
                - str: A descriptive result message.
                - dict | None: 'text' (concatenated output text, or the parsed JSON value for JSON/schema output),
                    'refusal' (always "" on success), 'reasoning' (reasoning dictionaries),
                    'tools' (complete, unanswered tool calls from this response), 'usage' (token counts), and 'response' (complete provider response).
                    Unfinished tool calls remain in 'response' and context but are not offered for execution.
                    For incomplete JSON output that cannot be parsed, 'text' is `None`. Original output remains in 'response'.
                    Returns `None` on failure or when 'response_type' is "none".
                    The 'usage' dictionary contains non-negative integer counts or `None` when unavailable:
                        - 'tokens_input': Total input tokens, including cached input.
                        - 'tokens_input_cached': Cached subset of input tokens.
                        - 'tokens_input_uncached': Input minus cached tokens, assuming zero cached tokens when unavailable.
                        - 'tokens_output': Total output tokens, including reasoning.
                        - 'tokens_output_reasoning': Reasoning subset of output tokens.
                        - 'tokens_total': Provider-reported total tokens.
                    Reported zeros are preserved. Missing, invalid, or inconsistent subset counts emit warnings
                    and remain `None`. Subsets must not be added to totals again. Original usage remains in 'response'.

        Notes:
            - Saves input and all output dictionaries in context only when the call succeeds.
              Failures leave context unchanged. Incomplete responses are successful and retained.
            - Explicit refusal output returns a failure with the refusal reason in the message.
              Refusals follow the configured retry policy, subject to interruption retry restrictions.
            - Function arguments remain JSON strings. Function execution and subsequent requests are caller-controlled.
            - Retries can repeat generation charges or hosted-tool actions after an ambiguous failure.
              No retry occurs after interruption.
            - Basic tool selection uses the same wire values across supported flavors. "always" sends "required".
              Tool availability, explicit selection, subsets, and JSON Schema support depend on the endpoint and model.
            - Changing endpoints does not clear history. Reset explicitly if the new provider cannot
              consume previous reasoning, tool, or extension items.
            - Function tools use top-level 'name' and 'parameters'. Results identify calls by 'call_id'.
              Programmatic function outputs may instead use 'name', 'namespace', and 'caller'.
              These resolve a pending call only when exactly one call matches.
              All original provider data is in 'response'.
        """
        return self._base.wrap(1, self._base.prompt, text, reset_context, response_type, **kwargs)

    def interrupt(self):
        """
        Request interruption of generation or compaction without acquiring the API lock.

        Returns:
            tuple[bool, str]: A tuple containing:
                - bool: `True` if interruption was requested or no operation was active, `False` otherwise.
                - str: A descriptive message about the operation result.

        Notes:
            - Does not wait for the active operation to finish.
              The interrupted operation returns a failure tuple and leaves local history unchanged.
            - Model discovery finishes or times out before interruption takes effect.
        """
        return self._base.interrupt()

    def get_context(self, **kwargs):
        """
        Obtain the input and output dictionaries stored in this client's context.

        Args:
            **kwargs:
                All settings (see `get_settings()`) can also be configured via keyword arguments from here.
                Additionally, special keyword arguments can be passed to `wrap()`:
                    persist (bool):
                        If `True`, settings applied via keyword arguments are not reverted after termination. Defaults to `False`.
                    mute (bool):
                        If `True`, all logs emitted by this function are muted. Defaults to `False`.

        Returns:
            tuple[bool, str, list[dict] | None]: A tuple containing:
                - bool: `True` if successful, `False` otherwise.
                - str: A descriptive result message.
                - list[dict] | None: Defensive copy of all context items, or `None` on failure.
        """
        return self._base.wrap(1, self._base.get_context, **kwargs)

    def set_context(self, mode="reset", messages=None, index=0, reverse_indexing=True, **kwargs):
        """
        Edit this client's context, preserving it if validation fails.

        Args:
            mode (str, optional):
                Use "reset", "insert", "replace", or "remove". Defaults to "reset".
                "replace" replaces as many items as supplied. "remove" removes one item.
            messages (dict | list[dict] | None, optional):
                Responses input or output dictionaries, including reasoning, tool results, or compacted output.
                Media is converted as in `prompt()`. Use `None` to clear on reset or remove an item.
                Defaults to `None`. Insert/replace require at least one item.
            index (int, optional):
                Non-negative item index, or insertion boundary. Defaults to `0`.
                Must remain non-negative for "reset", although no position is selected in that mode.
            reverse_indexing (bool, optional):
                Count from the end. Index zero appends on insert and selects the last item otherwise.
                Use `False` to count from the start. Defaults to `True`.
            **kwargs:
                All settings (see `get_settings()`) can also be configured via keyword arguments from here.
                Additionally, special keyword arguments can be passed to `wrap()`:
                    persist (bool):
                        If `True`, settings applied via keyword arguments are not reverted after termination. Defaults to `False`.
                    mute (bool):
                        If `True`, all logs emitted by this function are muted. Defaults to `False`.

        Returns:
            tuple[bool, str]: A tuple containing:
                - bool: `True` if successful, `False` otherwise.
                - str: A descriptive result message.
        """
        return self._base.wrap(0, self._base.set_context, mode, messages, index, reverse_indexing, **kwargs)

    def get_tools(self, **kwargs):
        """
        Obtain the configured tool definitions.

        Args:
            **kwargs:
                All settings (see `get_settings()`) can also be configured via keyword arguments from here.
                Additionally, special keyword arguments can be passed to `wrap()`:
                    persist (bool):
                        If `True`, settings applied via keyword arguments are not reverted after termination. Defaults to `False`.
                    mute (bool):
                        If `True`, all logs emitted by this function are muted. Defaults to `False`.

        Returns:
            tuple[bool, str, list[dict] | None]: A tuple containing:
                - bool: `True` if successful, `False` otherwise.
                - str: A descriptive result message.
                - list[dict] | None: Defensive copy of tool definitions, or `None` on failure.
        """
        return self._base.wrap(1, self._base.get_tools, **kwargs)

    def set_tools(self, tools=None, **kwargs):
        """
        Replace the tool definitions used by both generation methods.

        Args:
            tools (list[dict] | dict | None, optional):
                Responses tool dictionaries with 'type'. Function tools use top-level 'name',
                'parameters', 'description', and 'strict'.
                Extension fields are preserved. The server validates tool-specific schemas.
                Passing a `dict` is interpreted as a list containing one tool definition.
                Use `None` or `[]` to clear definitions. Defaults to `None`.
            **kwargs:
                All settings (see `get_settings()`) can also be configured via keyword arguments from here.
                Additionally, special keyword arguments can be passed to `wrap()`:
                    persist (bool):
                        If `True`, settings applied via keyword arguments are not reverted after termination. Defaults to `False`.
                    mute (bool):
                        If `True`, all logs emitted by this function are muted. Defaults to `False`.

        Returns:
            tuple[bool, str]: A tuple containing:
                - bool: `True` if successful, `False` otherwise.
                - str: A descriptive result message.

        Notes:
            - Tool availability depends on the provider and model. OpenAI supports web/file search,
              code interpreter, image generation, computer use, and MCP. OpenRouter supports "openrouter:web_search".
            - Define application tools, then return their results:

                import nimbro_api

                client = nimbro_api.openai.Responses()
                success, message = client.set_tools([{
                    'type': "function",
                    'name': "get_number",
                    'description': "Return the secret number.",
                    'parameters': {
                        'type': "object",
                        'properties': {},
                        'additionalProperties': False
                    },
                    'strict': True
                }])
                success, message, response = client.prompt("Call get_number.", reset_context=True, response_type="always")
                if success and response['tools']:
                    outputs = [{
                        'type': "function_call_output",
                        'call_id': call['call_id'],
                        'output': "372"
                    } for call in response['tools'] if call['name'] == "get_number"]
                    success, message, response = client.prompt(outputs, response_type="auto")

            - Use "auto" for tool-result continuation and instruct the model to report the supplied
              results. Using 'response_type' set to "text" disables tools. Some models then decline
              the original tool request instead of reporting results already supplied in history.
        """
        return self._base.wrap(0, self._base.set_tools, tools, **kwargs)

    def get_awaited_tools(self, **kwargs):
        """
        Obtain the IDs of unanswered application tool calls in local history.

        Args:
            **kwargs:
                All settings (see `get_settings()`) can also be configured via keyword arguments from here.
                Additionally, special keyword arguments can be passed to `wrap()`:
                    persist (bool):
                        If `True`, settings applied via keyword arguments are not reverted after termination. Defaults to `False`.
                    mute (bool):
                        If `True`, all logs emitted by this function are muted. Defaults to `False`.

        Returns:
            tuple[bool, str, list[str] | None]: A tuple containing:
                - bool: `True` if successful, `False` otherwise.
                - str: A descriptive result message.
                - list[str] | None: The 'call_id' values of unanswered function/custom/computer/shell/patch calls in order.
                    Returns `[]` if none are pending or `None` on failure. Matching output types resolve calls.
                    Async calls are included. Incomplete calls and hosted search/code-interpreter calls are excluded.
        """
        return self._base.wrap(1, self._base.get_awaited_tools, **kwargs)

    def compact(self, input_items, **kwargs):
        """
        Return a compacted context without changing this client's context.

        Args:
            input_items (str | list[dict]):
                History to compact, with media conversion as in `prompt()`.
            **kwargs:
                All settings (see `get_settings()`) can also be configured via keyword arguments from here.
                Additionally, special keyword arguments can be passed to `wrap()`:
                    persist (bool):
                        If `True`, settings applied via keyword arguments are not reverted after termination. Defaults to `False`.
                    mute (bool):
                        If `True`, all logs emitted by this function are muted. Defaults to `False`.

        Returns:
            tuple[bool, str, dict | None]: A tuple containing:
                - bool: `True` if successful, `False` otherwise.
                - str: A descriptive result message.
                - dict | None: Compaction response dictionary including compacted 'output', or `None` on failure.

        Notes:
            - Supported by OpenAI and compatible Open Responses servers.
            - Sends 'model', optional 'instructions', and the OpenAI service preference.
              Adopt the returned 'output' explicitly with `set_context(messages=result['output'])` when desired.
        """
        return self._base.wrap(1, self._base.compact, input_items, **kwargs)
