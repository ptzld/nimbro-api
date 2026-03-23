from nimbro_api.client import ClientBase
from nimbro_api.utility.api import validate_endpoint, get_request
from nimbro_api.utility.misc import assert_type_value, assert_log

class NoBase(ClientBase):

    def __init__(self, settings, default_settings, **kwargs):
        super().__init__(settings=settings, default_settings=default_settings, **kwargs)
        self._logger.debug(f"Initialized '{type(self).__name__}' object.")
        self._initialized = True

    def set_settings(self, settings, mode="set"):
        settings = self._introduce_settings(settings=settings, mode=mode)

        # message_response
        assert_type_value(obj=settings['message_response'], type_or_value=bool, name="setting 'message_response'")

        # endpoints
        assert_type_value(obj=settings['endpoints'], type_or_value=dict, name="setting 'endpoints'")
        assert_log(expression=len(settings['endpoints']) > 0, message="Expected setting 'endpoints' to define at least one endpoint.")
        for endpoint in settings['endpoints']:
            assert_type_value(obj=endpoint, type_or_value=str, name="all endpoint names in setting 'endpoints'")
            assert_log(expression=len(endpoint) > 0, message="Expected all endpoint names in setting 'endpoints' to be non-empty.")
            validate_endpoint(endpoint=settings['endpoints'][endpoint], flavors=None, require_key=False, require_name=False, setting_name=f"endpoint '{endpoint}' in setting 'endpoints'")

        # endpoint
        if isinstance(settings['endpoint'], dict):
            validate_endpoint(endpoint=settings['endpoint'], flavors=None, require_key=False, require_name=True, setting_name="endpoint provided through setting 'endpoint'")
            settings['endpoints'][settings['endpoint']['name']] = settings['endpoint']
            settings['endpoint'] = settings['endpoint']['name']
            del settings['endpoints'][settings['endpoint']]['name']
        else:
            assert_type_value(obj=settings['endpoint'], type_or_value=list(settings['endpoints'].keys()), name="setting 'endpoint'")

        # timeout_connect
        assert_type_value(obj=settings['timeout_connect'], type_or_value=[float, int, None], name="setting 'timeout_connect'")
        if settings['timeout_connect'] is not None:
            assert_log(
                expression=settings['timeout_connect'] > 0.0,
                message=f"Expected setting 'timeout_connect' to be None or greater zero but got '{settings['timeout_connect']}'."
            )

        # timeout_read
        assert_type_value(obj=settings['timeout_read'], type_or_value=[float, int, None], name="setting 'timeout_read'")
        if settings['timeout_read'] is not None:
            assert_log(
                expression=settings['timeout_read'] > 0.0,
                message=f"Expected setting 'timeout_read' to be None or greater zero but got '{settings['timeout_read']}'."
            )

        # apply settings
        self._endpoint = settings['endpoints'][settings['endpoint']]
        return self._apply_settings(settings, mode)

    def no(self):
        # use API
        success, message, response = get_request(
            api_name="no-as-a-service API",
            api_url=f"{self._endpoint['api_url']}",
            headers={},
            timeout=(self._settings['timeout_connect'], self._settings['timeout_read']),
            logger=self._logger
        )

        if success:
            # parse response
            response = response.json()
            assert_type_value(obj=response, type_or_value=dict, name="API response")
            assert_log(expression='reason' in response, message="Expected API response to contain the key 'reason'.")
            response = response['reason']
            assert_type_value(obj=response, type_or_value=str, name="value of key 'reason' in API response")
            if self._settings['message_response']:
                message = f"{message[:-1]}: {response}"
        else:
            response = None

        return success, message, response
