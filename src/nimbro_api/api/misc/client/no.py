from nimbro_api.client import Client
from ..base.no_base import NoBase

default_settings = {
    'logger_severity': None,
    'logger_name': "no-as-service",
    'logger_wrap_cutoff': None,
    'logger_wrap_failure_prefix': True,
    'message_response': True,
    'endpoints': {
        'official': {
            'api_url': "https://naas.isalman.dev/no"
        },
        'localhost': {
            'api_url': "http://localhost:3000/no"
        }
    },
    'endpoint': "official",
    'timeout_connect': 1.0,
    'timeout_read': 2.0,
    'retry': 2
}

class No(Client):
    """
    This is an implementation of the no-as-a-service API (https://github.com/hotheadhacker/no-as-a-service), with sensible default
    settings and behaviors throughout, capabilities for configuring endpoints, managing connections, and logging.
    """

    def __init__(self, settings=None, **kwargs):
        """
        Create a client implementing the no-as-a-service API (https://github.com/hotheadhacker/no-as-a-service).

        Args:
            settings (dict | None, optional):
                Settings initializing the object. Settings not contained are initialized to their default values.
                See the documentation of `get_settings()` for a comprehensive list of all available settings.
                Nested settings can be specified using dot-separated keys (e.g., "a.b.c" is equivalent to {"a": {"b": {"c": ...}}}).
                Use `None` to initialize with default settings. Defaults to `None`.
            **kwargs:
                All settings (see `get_settings()`) can also be initialized via keyword arguments.
                When doing so, 'settings' must be `None` or an empty `dict`.

        Raises:
            UnrecoverableError: If initialization fails, including invalid settings.
        """
        super().__init__(client_base=NoBase, settings=settings, default_settings=default_settings, **kwargs)

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
            logger_wrap_cutoff (int | None):
                Maximum visible message body length in wrapper result and retry logs, excluding prefixes. Longer bodies end with "...".
                Use `0` to suppress result logs and retain only retry prefixes. Use `None` to apply no restriction.
            logger_wrap_failure_prefix (bool):
                Include function and attempt information before ordinary failure messages logged by the wrapper.
                Use `False` to log only the operation message.
            message_response (bool):
                Include the returned reason in successful response messages when using `no()`.
            endpoints (dict):
                Endpoint definitions mapping names (`str`) to endpoints/providers (`dict`) of the targeted API.
                - The mapping must contain at least one endpoint and all names must contain a non-whitespace character.
                - Each endpoint must be a dictionary (`dict`), with the required key 'api_url'.
                - All values must be non-empty strings (`str`).
            endpoint (str | dict):
                Name of the defined endpoint to be used from the list of keys in setting 'endpoints'.
                Pass an endpoint definition (`dict`) with the additional key 'name' (a string containing a non-whitespace character) to automatically add/update the setting 'endpoints' and select it.
            timeout_connect (float | int | None):
                Time in seconds waited for connecting to the 'endpoint', or `None` to wait indefinitely.
                Numeric values must be positive and finite.
            timeout_read (float | int | None):
                Time in seconds waited for receiving a response from the 'endpoint', or `None` to wait indefinitely.
                Numeric values must be positive and finite.
            retry (bool | int):
                Defines retry behavior in failure cases, if the cause is eligible for retry:
                - If `True`, retries indefinitely. If `False` or `0`, failure is returned immediately.
                - Use a positive integer (`int`) to permit a specific number of additional retry attempts.

        Raises:
            UnrecoverableError: If 'name' is provided and does not refer to an existing setting.

        Returns:
            object: A deep copy of the current settings (`dict`) or a single setting when providing 'name'.

        Notes:
            - See the global dictionary 'default_settings' on top of this file for defaults.
            - A full settings snapshot consolidates the selected endpoint into 'endpoint', including its 'name', and omits 'endpoints'.
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

    def no(self, **kwargs):
        """
        Get a no response.

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
                - str | None: A no response (`str`), or `None` if not successful.
        """
        return self._base.wrap(1, self._base.no, **kwargs)
