import math
import time

import httpx

from nimbro_api.client import ClientBase
from nimbro_api.utility.api import get_api_key, validate_endpoint, get_request
from nimbro_api.utility.misc import UnrecoverableError, assert_type_value, assert_keys, assert_log

class SearXNGBase(ClientBase):

    def __init__(self, settings, default_settings, **kwargs):
        super().__init__(settings=settings, default_settings=default_settings, **kwargs)
        self.get_api_key = get_api_key.__get__(self)
        self._logger.debug(f"Initialized '{type(self).__name__}' object.")
        self._initialized = True

    def set_settings(self, settings, mode="set"):
        settings = self._introduce_settings(settings=settings, mode=mode)

        # message_results
        assert_type_value(obj=settings['message_results'], type_or_value=bool, name="setting 'message_results'")

        # endpoints
        assert_type_value(obj=settings['endpoints'], type_or_value=dict, name="setting 'endpoints'")
        assert_log(expression=len(settings['endpoints']) > 0, message="Expected setting 'endpoints' to define at least one endpoint.")
        for endpoint in settings['endpoints']:
            assert_type_value(obj=endpoint, type_or_value=str, name="endpoint name in setting 'endpoints'")
            assert_log(expression=len(endpoint.strip()) > 0, message=f"Expected a nonempty, non-whitespace endpoint name in setting 'endpoints' but got '{endpoint}'.")
            validate_endpoint(endpoint=settings['endpoints'][endpoint], flavors=None, require_key=True, require_name=False, allow_models=False, setting_name=f"endpoint '{endpoint}' in setting 'endpoints'")

        # endpoint
        if isinstance(settings['endpoint'], dict):
            validate_endpoint(endpoint=settings['endpoint'], flavors=None, require_key=True, require_name=True, allow_models=False, setting_name="endpoint provided through setting 'endpoint'")
            settings['endpoints'][settings['endpoint']['name']] = settings['endpoint']
            settings['endpoint'] = settings['endpoint']['name']
            del settings['endpoints'][settings['endpoint']]['name']
        else:
            assert_type_value(obj=settings['endpoint'], type_or_value=list(settings['endpoints'].keys()), name="setting 'endpoint'")

        # language
        assert_type_value(obj=settings['language'], type_or_value=[str, None], name="setting 'language'")
        if settings['language'] is not None:
            assert_log(expression=len(settings['language'].strip()) > 0, message=f"Expected setting 'language' to contain a non-whitespace character or be None but got '{settings['language']}'.")

        # safesearch
        assert_type_value(obj=settings['safesearch'], type_or_value=[int, None], name="setting 'safesearch'")
        assert_log(expression=settings['safesearch'] in [None, 0, 1, 2], message=f"Expected setting 'safesearch' to be None, 0, 1, or 2 but got '{settings['safesearch']}'.")

        # time_range
        assert_type_value(obj=settings['time_range'], type_or_value=["day", "month", "year", None], name="setting 'time_range'")

        # timeout_connect
        assert_type_value(obj=settings['timeout_connect'], type_or_value=[float, int, None], name="setting 'timeout_connect'")
        if settings['timeout_connect'] is not None:
            assert_log(
                expression=math.isfinite(settings['timeout_connect']) and settings['timeout_connect'] > 0.0,
                message=f"Expected setting 'timeout_connect' to be None or positive and finite but got '{settings['timeout_connect']}'."
            )

        # timeout_read
        assert_type_value(obj=settings['timeout_read'], type_or_value=[float, int, None], name="setting 'timeout_read'")
        if settings['timeout_read'] is not None:
            assert_log(
                expression=math.isfinite(settings['timeout_read']) and settings['timeout_read'] > 0.0,
                message=f"Expected setting 'timeout_read' to be None or positive and finite but got '{settings['timeout_read']}'."
            )

        # apply settings
        self._endpoint = settings['endpoints'][settings['endpoint']]
        return self._apply_settings(settings, mode)

    def search(self, query, page):
        stamp = time.perf_counter()

        # parse arguments
        assert_type_value(obj=query, type_or_value=str, name="argument 'query'")
        assert_log(expression=len(query.strip()) > 0, message=f"Expected argument 'query' to contain a non-whitespace character but got '{query}'.")
        assert_type_value(obj=page, type_or_value=int, name="argument 'page'")
        assert_log(expression=page > 0, message=f"Expected argument 'page' to be positive but got '{page}'.")

        # retrieve API key
        success, message, api_key = self.get_api_key()
        if not success:
            raise UnrecoverableError(message)

        # construct payload
        headers = {
            'Accept': "application/json",
            'Content-Type': "application/json",
            'Authorization': f"Bearer {api_key}",
            'HTTP-Referer': "https://github.com/ptzld/nimbro-api",
            'X-Title': "NimbRo API"
        }
        if api_key == "":
            del headers['Authorization']
        data = {
            'q': query,
            'pageno': page,
            'format': "json"
        }
        for name in ['language', 'safesearch', 'time_range']:
            if self._settings[name] is not None:
                data[name] = self._settings[name]
        api_url = str(httpx.URL(self._endpoint['api_url']).copy_merge_params(data))

        # use API
        success, message, response = get_request(
            api_name="SearXNG API",
            api_url=api_url,
            headers=headers,
            timeout=(self._settings['timeout_connect'], self._settings['timeout_read']),
            logger=self._logger
        )
        if success:
            # parse API response
            try:
                response = response.json()
                assert_type_value(obj=response, type_or_value=dict, name="API response")
                assert_keys(obj=response, keys=['results'], mode="required", name="API response")
                assert_type_value(obj=response['results'], type_or_value=list, name="key 'results' in API response")
                for index, item in enumerate(response['results']):
                    assert_type_value(obj=item, type_or_value=dict, name=f"result at index '{index}' in API response")
                assert_type_value(obj=response.get('unresponsive_engines', []), type_or_value=list, name="key 'unresponsive_engines' in API response")
            except Exception as error:
                success = False
                message = f"Failed to parse response from SearXNG API '{self._endpoint['api_url']}': {repr(error)}."

        # finalize response
        if success:
            count = len(response['results'])
            failed = len(response.get('unresponsive_engines', []))
            message = f"Retrieved '{count}' search result{'' if count == 1 else 's'} with '{failed}' unresponsive engine{'' if failed == 1 else 's'} in '{time.perf_counter() - stamp:.3f}s'."
            if self._settings['message_results']:
                message = f"{message[:-1]}: {response['results']}."
        else:
            response = None
        return success, message, response
