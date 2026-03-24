from .core_base import CoreBase
from ..client import Client

default_settings = {
    'logger_severity': 20,
    'logger_name': "nimbro_api",
    'logger_mute': False,
    'logger_line_length': None,
    'logger_multi_line_prefix': False,
    'keys_hide': True,
    'keys_cache': True,
    'defer_delay': 1.0,
    'retry': False
}

class Core(Client):
    """
    This class is instantiated automatically when this package is imported for the first time within a process.
    It serves as a central hub for managing process-wide settings, API keys, and data caches.
    All its methods can be accessed directly from the package level, e.g. `nimbro_api.get_settings()`.
    """

    def __init__(self):
        """
        This object is created automatically when this package is imported for the first time within a process.
        It serves as a central hub for managing process-wide settings, API keys, and data caches.
        All its methods can be accessed directly from the package level, e.g. `nimbro_api.get_settings()`.

        Do not instantiate this class manually.
        """
        super().__init__(client_base=CoreBase, default_settings=default_settings)

    # settings

    def get_settings(self, name=None):
        """
        Retrieve all current settings or a specific one.

        Args:
            name (str | None, optional):
                If provided, the one setting with this name is returned directly.
                Use `None` to return all settings as a dictionary. Defaults to `None`.

        Settings: See the global dictionary for defaults.
            logger_severity (int):
                Logger severity in [10, 20, 30, 40, 50] (DEBUG, INFO, WARN, ERROR, FATAL).
            logger_name (str | None):
                Logger name shown in each log identifying this object.
            logger_mute (bool):
                Mute all logs emitted within this process regardless of any individual 'logger_severity' setting.
            logger_line_length (int):
                Total number of characters before line-wrapping for all logs emitted within this process.
            logger_multi_line_prefix (bool):
                Fully prefix multi-line logs instead of indenting with whitespace only.
            keys_hide (bool):
                Do not show values of API keys in logs.
            keys_cache (bool):
                Cache keys to minimize reading environment variables.
            defer_delay (int | float):
                Time in seconds cache jobs are deferred after each registration of a new cache job.
            retry (bool | int):
                Defines retry behavior in failure cases, if the cause is eligible for retry.
                If `True`, retries indefinitely. If `False`, failure is returned immediately.
                Use a positive `int` to permit a specific number of retry attempts.

        Raises:
            UnrecoverableError: If 'name' is provided and does not refer to an existing setting.

        Returns:
            any: A deep copy of the current settings (`dict`) or a single setting when providing 'name' (`any`).
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

    # API keys

    def get_api_key(self, name=None, **kwargs):
        """
        Retrieve all cached API keys or a specific (un-/cached) one.

        Args:
            name (str, optional):
                Name of the API key to be retrieved.
                If the API key is not cached, it is read from environment variables.
                Use `None` to retrieve all cached API keys. Defaults to `None`.
            **kwargs:
                All settings (see `get_settings()`) can also be configured via keyword arguments from here.
                Additionally, special keyword arguments can be passes to `wrap()`:
                    persist (bool):
                        If `True`, settings applied via keyword arguments are not reverted after termination. Defaults to `False`.
                    mute (bool):
                        If `True`, all logs emitted by this function are muted. Defaults to `False`.

        Returns:
            tuple[bool, str, dict[str] | str | None]: A tuple containing:
                - bool: `True` if the operation succeeded, `False` otherwise.
                - str: A descriptive message about the operation result.
                - dict[str] | str | None: All cached API keys (`dict[str]`) mapping names (`str`)
                  to keys (`str`), the API key for the requested `name` (`str`), or `None` if not successful.
        """
        return self._base.wrap(1, self._base.get_api_key, name, **kwargs)

    def set_api_key(self, name, key, **kwargs):
        """
        Set an API key as an environment variable and, if 'keys_cache' is set, also cache it.

        Args:
            name (str):
                The (non-empty) name of the API key.
            key (str):
                The actual (non-empty) API key.
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
        return self._base.wrap(0, self._base.set_api_key, name, key, **kwargs)

    # caching

    def query_cache(self, category=None, identifier=None, age=None, **kwargs):
        """
        Query the process-wide cache by category, identifier, and age.

        Args:
            category (str | None, optional):
                Specifies the category under which the requested data is cached.
                Use `None` to obtain the entire cache as a dictionary. Defaults to `None`.
            identifier (str | None, optional):
                Specifies the identifier of the requested data within 'category'.
                Must be `None` if 'category' is `None`. Defaults to `None`.
            age (float | int | None, optional):
                If provided, the requested data is only considered for
                retrieval if it was updated no more than 'age' seconds ago.
                Must be `None` if 'identifier' is `None`. Defaults to `None`.
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
                - any: The cached data corresponding to the query, or `None` if not successful.
        """
        return self._base.wrap(1, self._base.query_cache, category, identifier, age, **kwargs)

    def update_cache(self, category, identifier, data, **kwargs):
        """
        Update the process-wide cache by category and identifier.

        Args:
            category (str):
                Specifies the category under which the provided data is cached.
            identifier (str):
                Specifies the identifier under which the provided data is cached.
            data (any):
                The serializable data to be cached.
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
        return self._base.wrap(0, self._base.update_cache, category, identifier, data, **kwargs)

    def clear_cache(self, category=None, identifier=None, age=None, **kwargs):
        """
        Clear the process-wide cache by category, identifier, and age.

        Args:
            category (str | None, optional):
                If provided, restricts the data deleted from cache to a specific category.
                Use `None` to consider deleting data from all categories. Defaults to `None`.
            identifier (str | None, optional):
                if provided, restricts the data deleted from cache to a specific identifier.
                Use `None` to consider deleting data from all identifiers. Defaults to `None`.
            age (float | int | None, optional):
                If provided, restricts the data deleted from cache to data updated more than 'age' seconds ago.
                Use `None` to consider deleting data regardless of age. Defaults to `None`.
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
        return self._base.wrap(0, self._base.query_cache, category, identifier, age, **kwargs)

    # deferred jobs

    def register_deferred_job(self, job, **kwargs):
        """
        Register a job to be executed when no other jobs have been registered within the timeframe defined by the "defer_delay" setting.

        Args:
            job (tuple | list):
                The job to be executed as a tuple containing a `callable` function (first element)
                and a `dict` containing the keyword arguments to be passed to the function (second element).
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
        return self._base.wrap(0, self._base.register_deferred_job, job, **kwargs)

    def execute_deferred_jobs(self, **kwargs):
        """
        Execute all threads that have been registered via `register_deferred_job()` but have not yet been run.

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
        return self._base.wrap(0, self._base.execute_deferred_jobs, **kwargs)
