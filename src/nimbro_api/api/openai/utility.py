import datetime

import nimbro_api
from nimbro_api.utility.api import get_request
from nimbro_api.utility.misc import UnrecoverableError, assert_type_value, assert_log

def validate_connection(self, api_key):
    if self._settings['validate_model'] is False:
        success = True
        message = "Not probing Models API."
    elif 'models_url' not in self._endpoint:
        success = True
        message = "Not probing Models API for endpoint that does not contain a Models URL."
    else:
        if isinstance(self._settings['validate_model'], (int, float)):
            # query cache
            success, message, cache = nimbro_api.query_cache(
                category="models",
                identifier=self._endpoint['models_url'],
                age=self._settings['validate_model'],
                mute=True
            )
            self._logger.debug(message)
        else:
            success = False
        if success:
            models = cache['data']
        else:
            # use Models API
            success, message, models = self.get_models(age=0, api_key=api_key)
            if success:
                if len(models) == 0:
                    self._logger.warn(message)
                else:
                    self._logger.debug(message)

        if success:
            # evaluate
            if self._settings['model'] in models or self._settings['model'].split(":", 1)[0] in models:
                message = f"Model '{self._settings['model']}' is served under Models API '{self._endpoint['models_url']}'."
            elif self._settings['model'] in models:
                message = f"Model '{self._settings['model']}' is served under Models API '{self._endpoint['models_url']}'."
            else:
                raise UnrecoverableError(f"Model '{self._settings['model']}' is not served under Models API '{self._endpoint['models_url']}': {models}")

    if success:
        self._logger.debug(message)

    return success, message

def get_models(self, age, api_key=None):
    # parse arguments
    assert_type_value(obj=age, type_or_value=[int, float, None], name="argument 'age'")
    if isinstance(age, (int, float)):
        assert_log(expression=age >= 0, message=f"Expected argument 'age' provided as '{type(age).__name__}' to be non-negative but got '{age}'.")

    # validate endpoint
    if 'models_url' not in self._endpoint:
        message = f"Endpoint '{self._settings['endpoint']}' does not specify a URL for the Models API."
        return False, message, None

    if age is None or age > 0:
        # query cache
        success, message, cache = nimbro_api.query_cache(
            category="models",
            identifier=self._endpoint['models_url'],
            age=age,
            mute=True
        )
        if success:
            cached_stamp = datetime.datetime.fromisoformat(cache['stamp'])
            now_stamp = datetime.datetime.now(datetime.timezone.utc)
            delta = (now_stamp - cached_stamp).total_seconds()
            models = cache['data']
            if len(models) == 0:
                message = f"Models API '{self._endpoint['models_url']}' did not serve any models '{delta:.3f}s' ago."
            else:
                message = f"Models served '{delta:.3f}s' ago under Models API '{self._endpoint['models_url']}': {models}"
            return True, message, models
        self._logger.debug(message)

    if api_key is None:
        # retrieve API key
        api_key = self.get_api_key()[2]

    # use Models API
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
    }
    success, message, response = get_request(
        api_name="Models API",
        api_url=self._endpoint['models_url'],
        headers=headers,
        timeout=(self._settings['timeout_connect'], self._settings['timeout_read']),
        logger=self._logger
    )
    if not success:
        return False, message, None
    self._logger.debug(message)

    # parse response
    try:
        response = response.json()
        # self._logger.warn(f"Keys: {response.keys()}")
        # self._logger.warn(f"object: {response['object']}")
        # self._logger.warn(f"len(data): {len(response['data'])}")
        models = [m['id'] for m in response['data']]
    except Exception as e:
        success = False
        message = f"Failed to parse response from Models API '{self._endpoint['api_url']}': {repr(e)}"
        models = None
    else:
        if len(models) == 0:
            message = f"There are no models served under Models API '{self._endpoint['models_url']}'."
        else:
            message = f"Models served under Models API '{self._endpoint['models_url']}': {models}"

    # update cache
    _success, _message = nimbro_api.update_cache(
        category="models",
        identifier=self._endpoint['models_url'],
        data=models,
        mute=True
    )
    if _success:
        self._logger.debug(_message)
    else:
        self._logger.warn(_message)

    return success, message, models
