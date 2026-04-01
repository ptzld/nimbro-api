from nimbro_api.client import Client
from ..base.no_base import NoBase

default_settings = {
    'logger_severity': None,
    'logger_name': "no-as-service",
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
    settings and behaviors throughout, extensive capabilities for configuring endpoints and models, managing connections, and logging.
    """

    def __init__(self, settings=None, **kwargs):
        """
        Create an Client implementing the no-as-a-service API (https://github.com/hotheadhacker/no-as-a-service).
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
                Logger severity in ["debug", "info", "warn", "error", "fatal", "off"] (str) or `None` to adopt global process-wide severity.
            logger_name (str | None):
                Logger name shown in each log identifying this object.
            endpoints (dict[dict]):
                Endpoint definitions pointing to providers of the v1 Chat Completions API.
                - Each endpoint must be a dictionary (`dict`), with the required keys "api_url", "key_type", and "key_value".
                - The value of 'key_type' must be either "environment" or "plain".
            endpoint (str | dict):
                Name of the endpoint to be used from the list of defined 'endpoints'.
                Pass an endpoint definition (`dict`) to automatically add/update the definition and select it.
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
                Additionally, special keyword arguments can be passes to `wrap()`:
                    persist (bool):
                        If `True`, settings applied via keyword arguments are not reverted after termination. Defaults to `False`.
                    mute (bool):
                        If `True`, all logs emitted by this function are muted. Defaults to `False`.

        Returns:
            tuple[bool, str, list[str] | None]: A tuple containing:
                - bool: `True` if the operation succeeded, `False` otherwise.
                - str: A descriptive message about the operation result.
                - str | None: A no response (`str`), or `None` if not successful.
        """
        return self._base.wrap(1, self._base.no, **kwargs)
