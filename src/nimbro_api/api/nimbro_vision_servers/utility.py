import datetime

import nimbro_api
from nimbro_api.utility.api import get_request, post_request
from nimbro_api.utility.misc import assert_type_value, assert_log

def get_status(self, age):
    # parse arguments
    assert_type_value(obj=age, type_or_value=[float, int, None], name="argument 'age'")
    if isinstance(age, (float, int)):
        assert_log(expression=age >= 0, message=f"Expected argument 'age' provided as '{type(age).__name__}' to be non-negative but got '{age}'.")

    if age is None or age > 0:
        # query cache
        success, message, cache = nimbro_api.query_cache(
            category="nimbro_vision_servers",
            identifier=f"{self._endpoint['api_url']}/status",
            age=age,
            mute=True
        )
        if success:
            cached_stamp = datetime.datetime.fromisoformat(cache['stamp'])
            now_stamp = datetime.datetime.now(datetime.timezone.utc)
            delta = (now_stamp - cached_stamp).total_seconds()
            flavor, init = None, None
            if cache['data']['model_family'] != self._model_family:
                success = False
                message = f"Model reported '{delta:.3f}s' ago, that is belongs to model family '{cache['data']['model_family']}' instead of '{self._model_family}'."
            elif cache['data']['status'] is None:
                message = f"Model was not loaded '{delta:.3f}s' ago."
            else:
                flavor = cache['data']['status']['flavor']
                if self._model_family == "sam2_realtime":
                    init = cache['data']['status']['tracker_initialized']
                    message = f"Model flavor '{flavor}' was loaded and {'' if init else 'not '}initialized '{delta:.3f}s' ago."
                else:
                    message = f"Model flavor '{flavor}' was loaded '{delta:.3f}s' ago."
            if self._model_family == "sam2_realtime":
                return success, message, flavor, init
            return success, message, flavor
        self._logger.debug(message)

    # retrieve API key
    api_key = self.get_api_key()[2]

    # use status API
    headers = {
        'Authorization': f"Bearer {api_key}"
    }
    success, message, response = get_request(
        api_name="NimbRo-Vision-Servers API",
        api_url=f"{self._endpoint['api_url']}/status",
        headers=headers,
        timeout=(self._settings['timeout_connect'], self._settings['timeout_read']),
        logger=self._logger
    )
    # [2025-11-19T16:32:56.147994]| {
    # [2025-11-19T16:32:56.147994]|   "model_family": "mmgroundingdino",
    # [2025-11-19T16:32:56.147994]|   "status": {
    # [2025-11-19T16:32:56.147994]|     "flavor": "large"
    # [2025-11-19T16:32:56.147994]|   }
    # [2025-11-19T16:32:56.147994]| }
    # [2025-11-19T17:17:42.954889]| {
    # [2025-11-19T17:17:42.954889]|   "model_family": "mmgroundingdino",
    # [2025-11-19T17:17:42.954889]|   "status": null
    # [2025-11-19T17:17:42.954889]| }

    flavor, init = None, None
    if success:
        # parse response
        response = response.json()
        if response['model_family'] != self._model_family:
            success = False
            message = f"Model reports being from model family '{response['model_family']}' instead of '{self._model_family}'."
        elif response['status'] is None:
            message = "Model is not loaded."
        else:
            flavor = response['status']['flavor']
            if self._model_family == "sam2_realtime":
                init = response['status']['tracker_initialized']
                message = f"Model flavor '{flavor}' is loaded and {'' if init else 'not '}initialized."
            else:
                message = f"Model flavor '{flavor}' is loaded."

        # update cache
        _success, _message = nimbro_api.update_cache(
            category="nimbro_vision_servers",
            identifier=f"{self._endpoint['api_url']}/status",
            data=response,
            mute=True
        )
        if _success:
            self._logger.debug(_message)
        else:
            self._logger.warn(_message)

    if self._model_family == "sam2_realtime":
        return success, message, flavor, init
    return success, message, flavor

def get_health(self, age):
    # parse arguments
    assert_type_value(obj=age, type_or_value=[float, int, None], name="argument 'age'")
    if isinstance(age, (float, int)):
        assert_log(expression=age >= 0, message=f"Expected argument 'age' provided as '{type(age).__name__}' to be non-negative but got '{age}'.")

    if age is None or age > 0:
        # query cache
        success, message, cache = nimbro_api.query_cache(
            category="nimbro_vision_servers",
            identifier=f"{self._endpoint['api_url']}/health",
            age=age,
            mute=True
        )
        if success:
            cached_stamp = datetime.datetime.fromisoformat(cache['stamp'])
            now_stamp = datetime.datetime.now(datetime.timezone.utc)
            delta = (now_stamp - cached_stamp).total_seconds()
            if cache['data']['status'] != "ok":
                is_healthy = False
                message = f"Model reported abnormal status '{cache['data']['status']}' (instead of 'ok') '{delta:.3f}s' ago."
            elif not cache['data']['cuda_available']:
                is_healthy = False
                message = f"Model reported that CUDA is unavailable '{delta:.3f}s' ago."
            else:
                is_healthy = True
                message = f"Model reported being healthy '{delta:.3f}s' ago."
            return True, message, is_healthy
        self._logger.debug(message)

    # retrieve API key
    api_key = self.get_api_key()[2]

    # use health API
    headers = {
        'Authorization': f"Bearer {api_key}"
    }
    success, message, response = get_request(
        api_name="NimbRo-Vision-Servers API",
        api_url=f"{self._endpoint['api_url']}/health",
        headers=headers,
        timeout=(self._settings['timeout_connect'], self._settings['timeout_read']),
        logger=self._logger
    )
    # [2025-11-19T16:32:56.067982]| {
    # [2025-11-19T16:32:56.067982]|   "status": "ok",
    # [2025-11-19T16:32:56.067982]|   "cuda_available": true
    # [2025-11-19T16:32:56.067982]| }

    if success:
        # parse response
        response = response.json()
        if response['status'] != "ok":
            is_healthy = False
            message = f"Model reports abnormal status '{response['status']}' (instead of 'ok')."
        elif not response['cuda_available']:
            is_healthy = False
            message = "Model reports that CUDA is unavailable."
        else:
            is_healthy = True
            message = "Model reports being healthy."

        # update cache
        _success, _message = nimbro_api.update_cache(
            category="nimbro_vision_servers",
            identifier=f"{self._endpoint['api_url']}/health",
            data=response,
            mute=True
        )
        if _success:
            self._logger.debug(_message)
        else:
            self._logger.warn(_message)
    else:
        is_healthy = None

    return success, message, is_healthy

def get_flavors(self, age):
    # parse arguments
    assert_type_value(obj=age, type_or_value=[float, int, None], name="argument 'age'")
    if isinstance(age, (float, int)):
        assert_log(expression=age >= 0, message=f"Expected argument 'age' provided as '{type(age).__name__}' to be non-negative but got '{age}'.")

    if age is None or age > 0:
        # query cache
        success, message, cache = nimbro_api.query_cache(
            category="nimbro_vision_servers",
            identifier=f"{self._endpoint['api_url']}/model_flavors",
            age=age,
            mute=True
        )
        if success:
            cached_stamp = datetime.datetime.fromisoformat(cache['stamp'])
            now_stamp = datetime.datetime.now(datetime.timezone.utc)
            delta = (now_stamp - cached_stamp).total_seconds()
            flavors = cache['data']
            num_flavors = len(flavors)
            if num_flavors == 0:
                message = f"There were no model available flavors '{delta:.3f}s' ago."
            else:
                message = f"There {'was' if num_flavors == 1 else 'were'} '{num_flavors}' model flavor{'' if num_flavors == 1 else 's'} available '{delta:.3f}s' ago: {flavors}"
            return True, message, flavors
        self._logger.debug(message)

    # retrieve API key
    api_key = self.get_api_key()[2]

    # use model_flavors API
    headers = {
        'Authorization': f"Bearer {api_key}"
    }
    success, message, response = get_request(
        api_name="NimbRo-Vision-Servers API",
        api_url=f"{self._endpoint['api_url']}/model_flavors",
        headers=headers,
        timeout=(self._settings['timeout_connect'], self._settings['timeout_read']),
        logger=self._logger
    )
    # [2025-11-19T16:32:56.228505]| {
    # [2025-11-19T16:32:56.228505]|   "flavors": [
    # [2025-11-19T16:32:56.228505]|     "tiny",
    # [2025-11-19T16:32:56.228505]|     "base",
    # [2025-11-19T16:32:56.228505]|     "large",
    # [2025-11-19T16:32:56.228505]|     "large_zeroshot",
    # [2025-11-19T16:32:56.228505]|     "llmdet_tiny",
    # [2025-11-19T16:32:56.228505]|     "llmdet_base",
    # [2025-11-19T16:32:56.228505]|     "llmdet_large"
    # [2025-11-19T16:32:56.228505]|   ]
    # [2025-11-19T16:32:56.228505]| }

    if success:
        # parse response
        flavors = response.json()
        flavors = flavors['flavors']
        num_flavors = len(flavors)
        if num_flavors == 0:
            message = "There are no available model flavors."
        elif num_flavors == 1:
            message = f"There is '1' available model flavor: {flavors}"
        else:
            message = f"There are '{num_flavors}' available model flavors: {flavors}"

        # update cache
        _success, _message = nimbro_api.update_cache(
            category="nimbro_vision_servers",
            identifier=f"{self._endpoint['api_url']}/model_flavors",
            data=flavors,
            mute=True
        )
        if _success:
            self._logger.debug(_message)
        else:
            self._logger.warn(_message)
    else:
        flavors = None

    return success, message, flavors

def load(self, flavor=None, age=0):
    # parse arguments
    if flavor is None:
        flavor = self._settings['flavor']
    assert_type_value(obj=flavor, type_or_value=str, name="argument 'flavor'")

    # get status
    if self._model_family == "sam2_realtime":
        success, message, old_flavor, _ = self.get_status(age=age)
    else:
        success, message, old_flavor = self.get_status(age=age)
    if success:
        if old_flavor == flavor:
            return True, message
    else:
        return False, message
    self._logger.debug(message)

    # get flavors
    success, message, flavors = self.get_flavors(age=age)
    if success:
        if flavor not in flavors:
            message = f"Cannot load model flavor '{flavor}'. Available model flavors are: {flavors}"
            return False, message
    else:
        return False, message
    self._logger.debug(message)

    # retrieve API key
    api_key = self.get_api_key()[2]

    # use load API
    headers = {
        'Authorization': f"Bearer {api_key}"
    }
    success, message, _ = post_request(
        api_name="NimbRo-Vision-Servers API",
        api_url=f"{self._endpoint['api_url']}/load",
        headers=headers,
        data={'flavor': flavor},
        timeout=(self._settings['timeout_connect'], self._settings['timeout_read_load']),
        logger=self._logger
    )
    # [2025-11-19T17:00:40.743519]| {
    # [2025-11-19T17:00:40.743519]|   "loaded_model": "mmgroundingdino",
    # [2025-11-19T17:00:40.743519]|   "flavor": "tiny"
    # [2025-11-19T17:00:40.743519]| }

    if success:
        if old_flavor is None:
            message = f"Loaded model flavor '{flavor}'."
        else:
            message = f"Loaded model flavor '{flavor}' in favor of '{old_flavor}'."

    return success, message

def unload(self):
    # get status
    if self._model_family == "sam2_realtime":
        success, message, flavor, _ = self.get_status(age=0)
    else:
        success, message, flavor = self.get_status(age=0)
    if success:
        if flavor is None:
            return True, message
    else:
        return False, message
    self._logger.debug(message)

    # retrieve API key
    api_key = self.get_api_key()[2]

    # use unload API
    headers = {
        'Authorization': f"Bearer {api_key}"
    }
    success, message, response = post_request(
        api_name="NimbRo-Vision-Servers API",
        api_url=f"{self._endpoint['api_url']}/unload",
        headers=headers,
        data=None,
        timeout=(self._settings['timeout_connect'], self._settings['timeout_read']),
        logger=self._logger
    )
    # [2025-11-19T17:00:39.666735]| {
    # [2025-11-19T17:00:39.666735]|   "unloaded": true
    # [2025-11-19T17:00:39.666735]| }
    # [2025-11-19T16:54:58.874884]| {
    # [2025-11-19T16:54:58.874884]|   "detail": "No model loaded"
    # [2025-11-19T16:54:58.874884]| }'.

    if success:
        message = f"Unloaded model flavor '{flavor}'."
    elif response.status_code == 503 and response.json()['detail'] == "No model loaded":
        success = True
        message = "There was no model loaded."

    return success, message
