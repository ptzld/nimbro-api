from .client_base import ClientBase
from ..utility.misc import assert_type_value, assert_log

class Client:
    """
    This class serves as a container for all user-exposed API implementations
    of nimbro_api, providing common design principles and standardization.

    To be used in conjunction with `nimbro_api.client.ClientBase`.
    """

    def __init__(self, client_base, *, settings=None, default_settings=None, **kwargs):
        """
        Initialize the Client and its underlying implementation.

        Args:
            client_base (type):
                The class providing the API implementation. Must be a subclass of `ClientBase`.
            settings (dict | None, optional):
                Settings initializing the client. Missing settings are drawn from 'default_settings'.
                See the documentation of `get_settings()` for of comprehensive list of all available settings.
                Use `None` to use default settings. Defaults to `None`.
            default_settings (dict | None, optional):
                The default settings of this client.
                Use `None` to use default settings. Defaults to `None`.
            **kwargs:
                The initial settings (see `get_settings()`) can also be configured via keyword arguments.
                When doing so, 'settings' must be `None` or an empty `dict`.

        Raises:
            UnrecoverableError: If 'client_base' is not a subclass of `ClientBase` or if initializing it fails.
        """
        assert_type_value(obj=client_base, type_or_value=type, name="argument 'client_base'")
        assert_log(expression=issubclass(client_base, ClientBase), message="Expected value of argument 'ClientBase' to be a subclass of 'ClientBase'.")
        if default_settings is None:
            default_settings = {
                'logger_severity': None,
                'logger_name': "Client",
                'retry': 2
            }
        self._base = client_base(settings=settings, default_settings=default_settings, **kwargs)

    def get_settings(self, name=None):
        """
        Retrieve all current settings or a specific one.

        Args:
            name (str | None, optional):
                If provided, the one setting with this name is returned directly.
                Use `None` to return all settings as a dictionary. Defaults to `None`.

        Settings: See the global dictionary for defaults.
            logger_severity (int | None):
                Logger severity in [10, 20, 30, 40, 50] or None to adopt global severity.
            logger_name (str | None):
                Logger name shown in each log for identifying this object.
            retry (bool | int):
                Defines retry behavior in failure cases, if the cause is eligible for retry.
                If `True`, retries indefinitely. If `False`, failure is returned immediately.
                Use a positive `int` to permit a specific number of retry attempts.

        Raises:
            UnrecoverableError: If 'name' is provided and does not refer to an existing setting.

        Returns:
            any: A deep copy of the current settings (`dict`) or a single setting when providing 'name' (`any`).

        Notes:
            - Overwrite this function in the class inheriting from `Client` to provide documentation for all settings.
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

        Notes:
            - Overwrite this function in the class inheriting from `Client` to expose all public functions directly.
        """
        return self._base.wrap(0, self._base.set_settings, settings, **kwargs)

    # def example_function(self, number, **kwargs):
    #     """
    #     This API inverts a non-negative number.

    #     Args:
    #         number (int):
    #             A non-negative number sent to the API.
    #         **kwargs:
    #             All settings (see `get_settings()`) can also be configured via keyword arguments from here.
    #             Additionally, special keyword arguments can be passes to `wrap()`:
    #                 persist (bool):
    #                     If `True`, settings applied via keyword arguments are not reverted after termination. Defaults to `False`.
    #                 mute (bool):
    #                     If `True`, all logs emitted by this function are muted. Defaults to `False`.

    #     Raises:
    #         UnrecoverableError: If input arguments are invalid.

    #     Returns:
    #         tuple[bool, str, dict | str | None]: A tuple containing:
    #             - bool: `True` if the operation succeeded, `False` otherwise.
    #             - str: A descriptive message about the operation result.
    #             - str | None: The inverted number, or `None` if not successful.
    #
    #     Notes:
    #         - This serves as an example function for the class inheriting from `Client`.
    #         - It exposes, documents and wraps it's actual implementation within `self._base`.
    #         - The return format of `self._base.example_function` must be success[bool], message[str], payload_a[any], payload_b[any], ...
    #         - The first positional argument of `wrap()` must specify the number of payload results returned by `self._base.example_function`.
    #     """
    #     return self._base.wrap(1, self._base.example_function, number, **kwargs)
