import time
import copy

from nimbro_api.client import ClientBase
from nimbro_api.utility.io import parse_image_b64
from nimbro_api.utility.api import get_api_key, validate_endpoint, post_request
from nimbro_api.utility.misc import assert_type_value, assert_log
from ..utility import get_status, get_health, get_flavors, load, unload

class Sam2RealtimeBase(ClientBase):

    def __init__(self, settings, default_settings, **kwargs):
        super().__init__(settings=settings, default_settings=default_settings, **kwargs)
        self._model_family = "sam2_realtime"
        self.get_api_key = get_api_key.__get__(self)
        self.get_status = get_status.__get__(self)
        self.get_health = get_health.__get__(self)
        self.get_flavors = get_flavors.__get__(self)
        self.load = load.__get__(self)
        self.unload = unload.__get__(self)
        self._logger.debug(f"Initialized '{type(self).__name__}' object.")
        self._initialized = True

    def set_settings(self, settings, mode="set"):
        settings = self._introduce_settings(settings=settings, mode=mode)

        # endpoints
        assert_type_value(obj=settings['endpoints'], type_or_value=dict, name="setting 'endpoints'")
        assert_log(expression=len(settings['endpoints']) > 0, message="Expected setting 'endpoints' to define at least one endpoint.")
        for endpoint in settings['endpoints']:
            assert_type_value(obj=endpoint, type_or_value=str, name="all endpoint names in setting 'endpoints'")
            assert_log(expression=len(endpoint) > 0, message="Expected all endpoint names in setting 'endpoints' to be non-empty.")
            validate_endpoint(endpoint=settings['endpoints'][endpoint], flavors=None, require_key=True, require_name=False, setting_name=f"endpoint '{endpoint}' in setting 'endpoints'")

        # endpoint
        if isinstance(settings['endpoint'], dict):
            validate_endpoint(endpoint=settings['endpoint'], flavors=None, require_key=True, require_name=True, setting_name="endpoint provided through setting 'endpoint'")
            settings['endpoints'][settings['endpoint']['name']] = settings['endpoint']
            settings['endpoint'] = settings['endpoint']['name']
            del settings['endpoints'][settings['endpoint']]['name']
        else:
            assert_type_value(obj=settings['endpoint'], type_or_value=list(settings['endpoints'].keys()), name="setting 'endpoint'")

        # validate_health
        assert_type_value(obj=settings['validate_health'], type_or_value=[float, int, bool], name="setting 'validate_health'")
        if isinstance(settings['validate_health'], (float, int)):
            assert_log(
                expression=settings['validate_health'] >= 0,
                message=f"Expected setting 'validate_health' provided as '{type(settings['validate_health']).__name__}' to be non-negative but got '{settings['validate_health']}'."
            )

        # validate_status
        assert_type_value(obj=settings['validate_status'], type_or_value=[float, int, bool], name="setting 'validate_status'")
        if isinstance(settings['validate_status'], (float, int)):
            assert_log(
                expression=settings['validate_status'] >= 0,
                message=f"Expected setting 'validate_status' provided as '{type(settings['validate_status']).__name__}' to be non-negative but got '{settings['validate_status']}'."
            )

        # flavor
        assert_type_value(obj=settings['flavor'], type_or_value=str, name="setting 'flavor'")

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

        # timeout_read_load
        assert_type_value(obj=settings['timeout_read_load'], type_or_value=[float, int, None], name="setting 'timeout_read_load'")
        if settings['timeout_read_load'] is not None:
            assert_log(
                expression=settings['timeout_read_load'] > 0.0,
                message=f"Expected setting 'timeout_read_load' to be None or greater zero but got '{settings['timeout_read_load']}'."
            )

        # timeout_read_infer
        assert_type_value(obj=settings['timeout_read_infer'], type_or_value=[float, int, None], name="setting 'timeout_read_infer'")
        if settings['timeout_read_infer'] is not None:
            assert_log(
                expression=settings['timeout_read_infer'] > 0.0,
                message=f"Expected setting 'timeout_read_infer' to be None or greater zero but got '{settings['timeout_read_infer']}'."
            )

        # apply settings
        self._endpoint = settings['endpoints'][settings['endpoint']]
        return self._apply_settings(settings, mode)

    def get_response(self, image, prompts):
        stamp = time.perf_counter()

        # parse image
        success, message, image_file, image_path = parse_image_b64(image=image, logger=self._logger)
        if not success:
            return False, message, None

        # parse prompts
        assert_type_value(obj=prompts, type_or_value=[list, None], name="argument 'prompts'")
        if isinstance(prompts, list):
            assert_log(expression=len(prompts) > 0, message="Expected argument 'prompts' provided as list to be non-empty.")

            for i, item in enumerate(prompts):
                assert_type_value(obj=item, type_or_value=dict, name=f"element '{i}' in argument 'prompts'")
                keys_item = set(item.keys())
                keys_box = {'object_id', 'bbox'}
                keys_point = {'object_id', 'points', 'labels'}
                assert_log(
                    expression=keys_item == keys_box or keys_item == keys_point,
                    message=f"Expected keys of element '{i}' in argument 'prompts' to be either {keys_box} or {keys_point} but got {keys_item}."
                )
                assert_type_value(obj=item['object_id'], type_or_value=int, name=f"key 'object_id' of element '{i}' in argument 'prompts'")
                if keys_item == keys_box:
                    assert_type_value(obj=item['bbox'], type_or_value=list, name=f"key 'bbox' of element '{i}' in argument 'prompts'")
                    assert_log(expression=len(item['bbox']) == 4, message=f"Expected value of key 'bbox' of element '{i}' in argument 'prompts' to be a list of length '4' but got '{len(item['bbox'])}'.")
                    for j, value in enumerate(item['bbox']):
                        assert_type_value(obj=value, type_or_value=[float, int], name=f"element '{j}' of value of key 'bbox' of element '{i}' in argument 'prompts'")
                else:
                    assert_type_value(obj=item['points'], type_or_value=list, name=f"key 'points' of element '{i}' in argument 'prompts'")
                    assert_type_value(obj=item['labels'], type_or_value=list, name=f"key 'labels' of element '{i}' in argument 'prompts'")
                    assert_log(
                        expression=len(item['points']) == len(item['labels']),
                        message=f"Expected values of keys 'points' and 'labels' of element '{i}' in argument 'prompts' to be a lists with matching size but got '{len(item['points'])}' and '{len(item['labels'])}'."
                    )
                    for j, point in enumerate(item['points']):
                        assert_type_value(obj=point, type_or_value=list, name=f"element '{j}' in value of key 'points' of element '{i}' in argument 'prompts'")
                        assert_log(
                            expression=len(point) == 2,
                            message=f"Expected element '{j}' of value of key 'points' of element '{i}' in argument 'prompts' to be a list of length '2' but got '{len(point)}'."
                        )
                        assert_type_value(obj=point[0], type_or_value=[float, int], name=f"element '0' of element '{j}' in value of key 'points' of element '{i}' in argument 'prompts'")
                        assert_type_value(obj=point[1], type_or_value=[float, int], name=f"element '1' of element '{j}' in value of key 'points' of element '{i}' in argument 'prompts'")
                    for label in item['labels']:
                        assert_type_value(obj=label, type_or_value=[0, 1], name=f"element '{j}' in value of key 'labels' of element '{i}' in argument 'prompts'")
            prompts = copy.deepcopy(prompts)

        # validate health
        if self._settings['validate_health'] is not False:
            success, message, is_healthy = self.get_health(
                age=0 if self._settings['validate_health'] is True else self._settings['validate_health']
            )
            if success:
                self._logger.debug(message)
            else:
                return False, message, None

        # validate status
        if self._settings['validate_status'] is not False:
            success, message, flavor, init = self.get_status(
                age=0 if self._settings['validate_status'] is True else self._settings['validate_status']
            )
            if success:
                if flavor != self._settings['flavor']:
                    if prompts is None:
                        if flavor is None:
                            message = "Model is not loaded and loading it would leave it uninitialized."
                        else:
                            message = f"Model flavor '{flavor}' is loaded but loading model flavor '{self._settings['flavor']}' would leave it uninitialized."
                        return False, message, None
                    else:
                        success, message = self.load(
                            flavor=self._settings['flavor'],
                            age=0
                        )
                        if success:
                            self._logger.debug(message)
                        else:
                            return False, message, None

                    self._logger.debug(message)
                elif prompts is None and not init:
                    message = "Tracker is not initialized."
                    return False, message, None
            else:
                return False, message, None
            self._logger.debug(message)

        # retrieve API key
        api_key = self.get_api_key()[2]

        # construct payload
        headers = {
            'Authorization': f"Bearer {api_key}"
        }
        if prompts is None:
            data = {
                'images': [image_file],
            }
        else:
            data = {
                'image': image_file,
                'prompts': prompts
            }

        # use API
        success, message, response = post_request(
            api_name="NimbRo-Vision-Servers API",
            api_url=f"{self._endpoint['api_url']}/{'infer' if prompts is None else 'update'}",
            headers=headers,
            data=data,
            timeout=(self._settings['timeout_connect'], self._settings['timeout_read_infer']),
            logger=self._logger
        )

        if success:
            # parse response
            response = response.json()
            response = response['artifact']
            if response['model'] != self._model_family:
                success = False
                message = f"Received response for unexpected model family '{response['model']}' instead of '{self._model_family}'."
                response = None
            elif prompts is None and len(response['tracks']) == 0:
                success = False
                message = "Tracker was not initialized."
                response = None
            else:
                response = response['tracks'][0]
                suffix = "" if image_path is None else f" from image '{image_path}'"
                if prompts is None:
                    message = f"Obtained '{len(response)}' track{'' if len(response) == 1 else 's'}{suffix} in '{time.perf_counter() - stamp:.3f}s'."
                else:
                    message = f"Initialized '{len(response)}' track{'' if len(response) == 1 else 's'}{suffix} in '{time.perf_counter() - stamp:.3f}s'."
        else:
            response = None

        return success, message, response
