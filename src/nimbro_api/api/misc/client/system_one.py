from nimbro_api.client import Client
from ..base.system_one_base import SystemOneBase

default_settings = {
    'logger_severity': None,
    'logger_name': "SystemOne",
    'message_results': True,
    'endpoints': {
        'OpenRouter': {
            'api_url': "https://openrouter.ai/api/v1/systemone",
            'key_type': "environment",
            'key_value': "OPENROUTER_API_KEY"
        },
        'TypeSafe': {
            'api_url': "https://api.typesafe.ai/v1/systemone",
            'key_type': "environment",
            'key_value': "TYPESAFE_API_KEY"
        },
        'vLLM': {
            'api_url': "http://localhost:8000/v1/systemone",
            'key_type': "environment",
            'key_value': "VLLM_API_KEY"
        }
    },
    'endpoint': "OpenRouter",
    'model': "jev-latest",
    'timeout_connect': 2.0,
    'timeout_read': 10.0,
    'retry': 2
}

class SystemOne(Client):
    """
    This is an implementation of the System One API (https://openrouter.ai/docs/api/api-reference/systemone/submit-a-system-one-request), with sensible default
    settings and behaviors throughout, extensive capabilities for configuring endpoints and models, managing connections, and logging.

    Compatible with OpenRouter, TypeSafe, and local servers implementing the System One request and response format.
    The vLLM endpoint requires a System One wrapper or plugin, such as vllm-jev; stock vLLM does not provide this endpoint.
    """

    def __init__(self, settings=None, **kwargs):
        """
        Create a client implementing the System One API (https://openrouter.ai/docs/api/api-reference/systemone/submit-a-system-one-request).

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
        super().__init__(client_base=SystemOneBase, settings=settings, default_settings=default_settings, **kwargs)

    def get_settings(self, name=None):
        """
        Obtain all settings or a specific one.

        Args:
            name (str | None, optional):
                If provided, the one setting with this name is returned directly.
                Use `None` to return all settings as a dictionary. Defaults to `None`.

        Settings:
            logger_severity (str | None):
                Logger severity in ["debug", "info", "warn", "error", "fatal", "off"] (`str`) or `None` to adopt global process-wide severity.
            logger_name (str | None):
                Logger name shown in each log identifying this object.
            message_results (bool):
                Include the state, formatted questions, and answers in the operation result message, with labeled options
                and probabilities when available. Highlighted options are marked with '*': the selected Choice,
                the more likely Noul answer (unless tied), or the highest-probability Score levels when probabilities are available.
                Exact weighted scores and confidence values remain available in the response dictionary, but are not displayed.
            endpoints (dict):
                Endpoint definitions mapping names (`str`) to endpoints/providers (`dict`) of the targeted API.
                - Each endpoint must be a dictionary (`dict`), with the required keys 'api_url', 'key_type', and 'key_value'.
                - The value of 'key_type' must be either "environment" or "plain".
                - All values must be non-empty strings (`str`), except 'key_value', which may be empty.
                - Use 'key_type'="plain" and 'key_value'="" for an unauthenticated local server.
            endpoint (str | dict):
                Name of the defined endpoint to be used from the list of keys in setting 'endpoints'.
                Pass an endpoint definition (`dict`) with the additional key 'name' to automatically add/update the setting 'endpoints' and select it.
            model (str | None):
                Name of the model used. OpenRouter and TypeSafe accept the default alias "jev-latest".
                Set the served model name when using a local server, or use `None` to omit the field for server-side selection.
                Model availability is not probed, since model catalogs differ and may omit valid aliases or versioned models.
            timeout_connect (float | int | None):
                Time in seconds waited for connecting to the 'endpoint', or `None` to wait indefinitely.
            timeout_read (float | int | None):
                Time in seconds waited for receiving a response from the 'endpoint', or `None` to wait indefinitely.
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
                Additionally, special keyword arguments can be passed to `wrap()`:
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

    def evaluate(self, state, questions, **kwargs):
        """
        Evaluate typed questions against a shared state.

        Args:
            state (str | dict | list):
                Text or structured JSON context shared by all questions. Forwarded without conversion.
                Media objects in structured state are only supported if the selected server accepts their format.
            questions (dict):
                Non-empty map of question identifiers (`str`) to question objects (`dict`).
                Each question requires 'type' ("noul", "choice", or "score") and 'instructions' (`str`, `dict`, or `list`).
                - Noul: a yes/no question, optionally with 'criteria' mapping "true" and/or "false" to descriptions.
                - Choice: requires 'criteria' mapping option labels (`str`) to descriptions (`str`, `dict`, `list`, or `None`).
                - Score: requires 'criteria' as an ordered list of level descriptions (`str`, `dict`, or `list`), from low to high.
                Question counts, option counts, and context limits depend on the selected server and model.
            **kwargs:
                All settings (see `get_settings()`) can also be configured via keyword arguments from here.
                Additionally, special keyword arguments can be passed to `wrap()`:
                    persist (bool):
                        If `True`, settings applied via keyword arguments are not reverted after termination. Defaults to `False`.
                    mute (bool):
                        If `True`, all logs emitted by this function are muted. Defaults to `False`.

        Returns:
            tuple[bool, str, dict | None]: A tuple containing:
                - bool: `True` if the operation succeeded, `False` otherwise.
                - str: A descriptive message about the operation result.
                - dict | None: The complete API response, or `None` if not successful.
                  The 'answers' map uses the supplied question identifiers. Noul returns 'noul' (probability of yes),
                  Choice returns 'choice' (selected label), and Score returns 'score' (probability-weighted zero-based level).
                  Additional fields, such as probabilities, confidence, legend, usage, model, and provider metadata, are preserved.

        Notes:
            - Confidence semantics vary across implementations. Values are returned without normalization or thresholding.
            - The default endpoint is OpenRouter's TypeSafe-compatible System One endpoint, not its alpha Decisions endpoint.
        """
        return self._base.wrap(1, self._base.evaluate, state, questions, **kwargs)
