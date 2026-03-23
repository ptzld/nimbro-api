from nimbro_api.client import Client
from ..base.kosmos2_base import Kosmos2Base

default_settings = {
    'logger_severity': None,
    'logger_name': "Kosmos2",
    'message_results': True,
    'endpoints': {
        'AIS': {
            'api_url': "https://api-code.ais.uni-bonn.de/v1/vision/kosmos2",
            'key_type': "environment",
            'key_value': "AIS_API_KEY"
        },
        'localhost': {
            'api_url': "http://localhost:9039",
            'key_type': "environment",
            'key_value': "NIMBRO_VISION_API_KEY"
        }
    },
    'endpoint': "AIS",
    'validate_health': 60,
    'validate_flavor': True,
    'flavor': "patch14-224",
    'prompt': "<grounding> Describe this image in detail:",
    'num_beams': 3,
    'max_new_tokens': 1024,
    'max_batch_size': 6,
    'timeout_connect': 2.0,
    'timeout_read': 5.0,
    'timeout_read_load': 5.0,
    'timeout_read_infer': 5.0,
    'retry': 2
}

class Kosmos2(Client):
    """
    This is an implementation of the NimbRo Vision Servers API for Kosmos-2 (https://github.com/AIS-Bonn/nimbro_vision_servers/tree/main/models/kosmos2),
    with sensible default settings and behaviors throughout, extensive capabilities for configuring endpoints and models,
    managing connections, converting image encodings, and logging.
    """

    def __init__(self, settings=None, **kwargs):
        """
        Create an Client implementing the NimbRo Vision Servers API for Kosmos-2 (https://github.com/AIS-Bonn/nimbro_vision_servers/tree/main/models/kosmos2).
        """
        super().__init__(client_base=Kosmos2Base, settings=settings, default_settings=default_settings, **kwargs)

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
            message_results (bool):
                Include results in successful response messages when using `get_response()`.
            endpoints (dict[dict]):
                Endpoint definitions pointing to providers of the NimbRo Vision Servers API for Kosmos-2.
                - Each endpoint must be a dictionary (`dict`), with the required keys "api_url", "key_type", and "key_value".
                - The value of 'key_type' must be either "environment" or "plain".
                - All values must be non-empty strings (`str`).
            endpoint (str | dict):
                Name of the endpoint to be used from the list of defined 'endpoints'.
                Pass an endpoint definition (`dict`) to automatically add/update the definition and select it.
            validate_health (float | int | bool):
                Validate the set 'endpoint' is healthy when using `get_response()`:
                - Use `float` or `int` to set the time in seconds permitted for cached responses to be reused before requesting a new one.
                - Use `True` to force validation, or `False` to deactivate it.
            validate_flavor (float | int | bool):
                Validate the set 'flavor' is available with the set 'endpoint' when using `load()` or `get_response()`:
                - Use `float` or `int` to set the time in seconds permitted for cached responses to be reused before requesting a new one.
                - Use `True` to force validation, or `False` to deactivate it.
            flavor (str):
                Name of the model flavor used (see `get_flavors()`).
            prompt (str):
                Model prompt parameter providing the task.
            num_beams (int):
                Model num_beams parameter (> 0).
            max_new_tokens (int):
                Model max_new_tokens parameter (> 0).
            max_batch_size (int):
                Model max_batch_size parameter (> 0).
            timeout_connect (float | int | None):
                Time in seconds waited for connecting to the 'endpoint', or `None` to wait indefinitely.
            timeout_read (float | int | None):
                Time in seconds waited for receiving a response from the 'endpoint' (except "load" and "infer"), or `None` to wait indefinitely.
            timeout_read_load (float | int | None):
                Time in seconds waited for receiving a "load" response from the 'endpoint', or `None` to wait indefinitely.
            timeout_read_infer (float | int | None):
                Time in seconds waited for receiving an "infer" response from the 'endpoint', or `None` to wait indefinitely.
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

    def get_status(self, age=0, **kwargs):
        """
        Request the status of the set 'endpoint'.

        Args:
            age (float | int | None, optional):
                The time in seconds permitted for cached responses to this request to be reused instead requesting a new one.
                Use `None` to always reuse cached responses regardless of age. Defaults to 0,
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
                - str | None: The name of the model flavor loaded by the 'endpoint', or `None` if either not successful, or no model is loaded.
        """
        return self._base.wrap(1, self._base.get_status, age, **kwargs)

    def get_health(self, age=0, **kwargs):
        """
        Request the health of the set 'endpoint'.

        Args:
            age (float | int | None, optional):
                The time in seconds permitted for cached responses to this request to be reused instead requesting a new one.
                Use `None` to always reuse cached responses regardless of age. Defaults to 0,
            **kwargs:
                All settings (see `get_settings()`) can also be configured via keyword arguments from here.
                Additionally, special keyword arguments can be passes to `wrap()`:
                    persist (bool):
                        If `True`, settings applied via keyword arguments are not reverted after termination. Defaults to `False`.
                    mute (bool):
                        If `True`, all logs emitted by this function are muted. Defaults to `False`.

        Returns:
            tuple[bool, str, bool | None]: A tuple containing:
                - bool: `True` if the operation succeeded, `False` otherwise.
                - str: A descriptive message about the operation result.
                - bool | None: `True` if the 'endpoint' reports being healthy, `False` if is not, or `None` if either not successful.
        """
        return self._base.wrap(1, self._base.get_health, age, **kwargs)

    def get_flavors(self, age=0, **kwargs):
        """
        Request the available model flavors of the set 'endpoint'.

        Args:
            age (float | int | None, optional):
                The time in seconds permitted for cached responses to this request to be reused instead requesting a new one.
                Use `None` to always reuse cached responses regardless of age. Defaults to 0,
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
                - list[str] | None: A list of model flavors available with the 'endpoint', or `None` if not successful.
        """
        return self._base.wrap(1, self._base.get_flavors, age, **kwargs)

    def load(self, **kwargs):
        """
        Request the set 'endpoint' to load the set 'model'.

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
        return self._base.wrap(0, self._base.load, **kwargs)

    def unload(self, **kwargs):
        """
        Request the set 'endpoint' to unload any loaded model.

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
        return self._base.wrap(0, self._base.unload, **kwargs)

    def get_response(self, image, **kwargs):
        """
        Request a model response to the set task for an image.

        Args:
            image (str | bytes):
                The image file to be transcribed as a local path, URL, Base64 encoding (all `str`), or raw `bytes`.
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
                - dict | None: A dictionary (`dict`) containing the model response according to the set 'prompt', or `None` if not successful.
        """
        return self._base.wrap(1, self._base.get_response, image, **kwargs)
