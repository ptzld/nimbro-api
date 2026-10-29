from nimbro_api.client import Client
from ..base.searxng_base import SearXNGBase

default_settings = {
    'logger_severity': None,
    'logger_name': "SearXNG",
    'logger_wrap_cutoff': None,
    'logger_wrap_failure_prefix': True,
    'message_results': True,
    'endpoints': {
        'localhost': {
            'api_url': "http://localhost:8080/search",
            'key_type': "plain",
            'key_value': ""
        }
    },
    'endpoint': "localhost",
    'language': None,
    'safesearch': None,
    'time_range': None,
    'timeout_connect': 5.0,
    'timeout_read': 30.0,
    'retry': 2
}

class SearXNG(Client):
    """
    This is an implementation of the SearXNG search API (https://docs.searxng.org/dev/search_api.html),
    with configurable endpoints, search filters, pagination, connection management, and logging.

    Compatible with public and local servers that enable "json" in 'search.formats'.
    Public instances may restrict API access; the default endpoint targets a local server.
    Search results contain snippets and links rather than the full contents of linked pages.
    """

    def __init__(self, settings=None, **kwargs):
        """
        Create a client implementing the SearXNG search API (https://docs.searxng.org/dev/search_api.html).

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
            UnrecoverableError: If initialization settings are invalid.
        """
        super().__init__(client_base=SearXNGBase, settings=settings, default_settings=default_settings, **kwargs)

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
            message_results (bool):
                Include formatted search results in successful response messages when using `search()`.
                Does not change the returned response dictionary.
            endpoints (dict):
                Endpoint definitions mapping names (`str`) to endpoints/providers (`dict`) of the targeted API.
                - The mapping must contain at least one endpoint and all names must contain a non-whitespace character.
                - Each endpoint must be a dictionary (`dict`), with the required keys 'api_url', 'key_type', and 'key_value'.
                - The value of 'key_type' must be either "environment" or "plain".
                - All values must be non-empty strings (`str`), except 'key_value', which may be empty.
                - Use 'key_type'="plain" and 'key_value'="" for an unauthenticated server.
            endpoint (str | dict):
                Name of the defined endpoint to be used from the list of keys in setting 'endpoints'.
                Pass an endpoint definition (`dict`) with the additional key 'name' (a string containing a non-whitespace character) to automatically add/update the setting 'endpoints' and select it.
            language (str | None):
                Nonempty SearXNG language code, such as "en" or "de-DE".
                Use `None` to omit the parameter and use the server default.
            safesearch (int | None):
                Filtering level: `0` for off, `1` for moderate, or `2` for strict.
                Use `None` to omit the parameter and use the server default.
                Support depends on the underlying search engine.
            time_range (str | None):
                "day", "month", or "year". Use `None` to omit the time filter.
                Support depends on the underlying search engine.
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
            UnrecoverableError: If 'name' does not refer to an existing setting.

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

    def search(self, query, page=1, **kwargs):
        """
        Retrieve one page of search results.

        Args:
            query (str):
                Nonempty search query. SearXNG search syntax is passed through unchanged.
            page (int, optional):
                Positive page number, subject to engine pagination support. Defaults to `1`.
            **kwargs:
                All settings (see `get_settings()`) can also be configured via keyword arguments from here.
                Additionally, special keyword arguments can be passed to `wrap()`:
                    persist (bool):
                        If `True`, settings applied via keyword arguments are not reverted after termination. Defaults to `False`.
                    mute (bool):
                        If `True`, all logs emitted by this function are muted. Defaults to `False`.

        Returns:
            tuple[bool, str, dict | None]: A tuple containing:
                - bool: `True` if a valid JSON search response was received, `False` otherwise.
                - str: A summary with result and unresponsive-engine counts and elapsed time,
                  including formatted results when 'message_results' is enabled, or a failure message.
                - dict | None: The full server response, including 'results', a list of
                  result dictionaries, or `None` on failure. Result fields typically
                  include 'title', 'url', and 'content' (a snippet). Additional metadata,
                  such as 'answers', 'suggestions', and 'unresponsive_engines', is retained.

        Notes:
            - Empty results and responses reporting unresponsive engines remain successful. Inspect 'unresponsive_engines' for incomplete coverage, including when no results are returned.
            - HTTP failures and invalid JSON responses return failure tuples.
            - No linked pages are fetched and no additional pages are requested automatically.
        """
        return self._base.wrap(1, self._base.search, query, page, **kwargs)
