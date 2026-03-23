import copy
import time

from nimbro_api.client import ClientBase
from nimbro_api.utility.io import parse_image_b64
from nimbro_api.utility.api import get_api_key, validate_endpoint, post_request
from nimbro_api.utility.misc import assert_type_value, assert_keys, assert_log, format_obj
from nimbro_api.utility.string import is_base64
from ..utility import get_status, get_health, get_flavors, load, unload

class DamBase(ClientBase):

    def __init__(self, settings, default_settings, **kwargs):
        super().__init__(settings=settings, default_settings=default_settings, **kwargs)
        self._model_family = "dam"
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

        # message_results
        assert_type_value(obj=settings['message_results'], type_or_value=bool, name="setting 'message_results'")

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

        # validate_flavor
        assert_type_value(obj=settings['validate_flavor'], type_or_value=[float, int, bool], name="setting 'validate_flavor'")
        if isinstance(settings['validate_flavor'], (float, int)):
            assert_log(
                expression=settings['validate_flavor'] >= 0,
                message=f"Expected setting 'validate_flavor' provided as '{type(settings['validate_flavor']).__name__}' to be non-negative but got '{settings['validate_flavor']}'."
            )

        # flavor
        assert_type_value(obj=settings['flavor'], type_or_value=str, name="setting 'flavor'")

        # query
        assert_type_value(obj=settings['query'], type_or_value=str, name="setting 'query'")

        # temperature
        assert_type_value(obj=settings['temperature'], type_or_value=[float, int], name="setting 'temperature'")
        settings['temperature'] = float(settings['temperature'])
        assert_log(
            expression=0.0 <= settings['temperature'] <= 1.0,
            message=f"Expected setting 'temperature' to be between zero and one (inclusive) but got '{settings['temperature']}'."
        )

        # top_p
        assert_type_value(obj=settings['top_p'], type_or_value=[float, int], name="setting 'top_p'")
        settings['top_p'] = float(settings['top_p'])
        assert_log(
            expression=0.0 <= settings['top_p'] <= 1.0,
            message=f"Expected setting 'top_p' to be between zero and one (inclusive) but got '{settings['top_p']}'."
        )

        # num_beams
        assert_type_value(obj=settings['num_beams'], type_or_value=int, name="setting 'num_beams'")
        assert_log(
            expression=settings['num_beams'] > 0,
            message=f"Expected setting 'num_beams' to be non-negative but got '{settings['num_beams']}'."
        )

        # max_new_tokens
        assert_type_value(obj=settings['max_new_tokens'], type_or_value=int, name="setting 'max_new_tokens'")
        assert_log(
            expression=settings['max_new_tokens'] > 0,
            message=f"Expected setting 'max_new_tokens' to be non-negative but got '{settings['max_new_tokens']}'."
        )

        # max_batch_size
        assert_type_value(obj=settings['max_batch_size'], type_or_value=int, name="setting 'max_batch_size'")
        assert_log(
            expression=settings['max_batch_size'] > 0,
            message=f"Expected setting 'max_batch_size' to be non-negative but got '{settings['max_batch_size']}'."
        )

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

    def get_descriptions(self, image, prompts):
        stamp = time.perf_counter()

        # parse image
        success, message, image_file, image_path = parse_image_b64(image=image, logger=self._logger)
        if not success:
            return False, message, None

        # parse prompts
        assert_type_value(obj=prompts, type_or_value=list, name="argument 'prompts'")
        assert_log(expression=len(prompts) > 0, message="Expected argument 'prompts' to be a non-empty list.")
        for i, item in enumerate(prompts):
            assert_type_value(obj=item, type_or_value=dict, name=f"element '{i}' in argument 'prompts'")
            assert_keys(obj=item, keys=['mask', 'bbox'], mode="match", name=f"element '{i}' in argument 'prompts'")
            assert_log(expression=is_base64(string=item['mask']), message=f"key 'mask' of element '{i}' in argument 'prompts'")
            assert_type_value(obj=item['bbox'], type_or_value=list, name=f"key 'bbox' of element '{i}' in argument 'prompts'")
            assert_log(expression=len(item['bbox']) == 4, message=f"Expected value of key 'bbox' of element '{i}' in argument 'prompts' to be a list of length '4' but got '{len(item['bbox'])}'.")
            for j, value in enumerate(item['bbox']):
                assert_type_value(obj=value, type_or_value=[float, int], name=f"element '{j}' of key 'bbox' of element '{i}' in argument 'prompts'")
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

        # validate model
        if self._settings['validate_flavor'] is not False:
            success, message = self.load(
                flavor=self._settings['flavor'],
                age=0 if self._settings['validate_flavor'] is True else self._settings['validate_flavor']
            )
            if success:
                self._logger.debug(message)
            else:
                return False, message, None

        # retrieve API key
        api_key = self.get_api_key()[2]

        # construct payload
        headers = {
            'Authorization': f"Bearer {api_key}"
        }
        data = {
            'images': [image_file],
            'prompts': [prompts],
            'queries': [self._settings['query']],
            'inference_parameters': {
                'temperature': self._settings['temperature'],
                'top_p': self._settings['top_p'],
                'num_beams': self._settings['num_beams'],
                'max_new_tokens': self._settings['max_new_tokens'],
                'max_batch_size': self._settings['max_batch_size']
            }
        }

        # use API
        success, message, response = post_request(
            api_name="NimbRo-Vision-Servers API",
            api_url=f"{self._endpoint['api_url']}/infer",
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
            else:
                response = response['descriptions'][0]
                suffix = "" if image_path is None else f" for image '{image_path}'"
                message = f"Generated '{len(response)}' description{'' if len(response) == 1 else 's'}{suffix} in '{time.perf_counter() - stamp:.3f}s'"
                if self._settings['message_results']:
                    if len(response) > 0:
                        message = f"{message}: {format_obj(obj=response)}."
                    else:
                        message = f"{message}."
                else:
                    message = f"{message}."
        else:
            response = None

        return success, message, response
