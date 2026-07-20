import copy
import time

from nimbro_api.client import ClientBase
from nimbro_api.utility.api import get_api_key, validate_endpoint, post_request
from nimbro_api.utility.misc import UnrecoverableError, assert_type_value, assert_log
from ..utility import validate_connection, get_models

class ClassifyBase(ClientBase):

    def __init__(self, settings, default_settings, **kwargs):
        super().__init__(settings=settings, default_settings=default_settings, **kwargs)
        self.get_api_key = get_api_key.__get__(self)
        self.validate_connection = validate_connection.__get__(self)
        self.get_models = get_models.__get__(self)
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

        # mode
        assert_type_value(obj=settings['mode'], type_or_value=["input", "messages"], name="setting 'mode'")

        # model
        assert_type_value(obj=settings['model'], type_or_value=[str, None], name="setting 'model'")

        # validate_model
        assert_type_value(obj=settings['validate_model'], type_or_value=[float, int, bool], name="setting 'validate_model'")
        if isinstance(settings['validate_model'], (int, float)):
            assert_log(
                expression=settings['validate_model'] >= 0,
                message=f"Expected setting 'validate_model' provided as '{type(settings['validate_model']).__name__}' to be non-negative but got '{settings['validate_model']}'."
            )

        # truncate_prompt_tokens
        assert_type_value(obj=settings['truncate_prompt_tokens'], type_or_value=[int, None], name="setting 'truncate_prompt_tokens'")
        if isinstance(settings['truncate_prompt_tokens'], int):
            assert_log(
                expression=settings['truncate_prompt_tokens'] >= -1,
                message=f"Expected setting 'truncate_prompt_tokens' to be -1 or greater but got '{settings['truncate_prompt_tokens']}'."
            )

        # truncation_side
        assert_type_value(obj=settings['truncation_side'], type_or_value=[str, None], name="setting 'truncation_side'")

        # request_id
        assert_type_value(obj=settings['request_id'], type_or_value=[str, None], name="setting 'request_id'")

        # priority
        assert_type_value(obj=settings['priority'], type_or_value=int, name="setting 'priority'")
        assert_log(expression=settings['priority'] >= 0, message=f"Expected setting 'priority' to be non-negative but got '{settings['priority']}'.")

        # mm_processor_kwargs
        assert_type_value(obj=settings['mm_processor_kwargs'], type_or_value=[dict, None], name="setting 'mm_processor_kwargs'")

        # cache_salt
        assert_type_value(obj=settings['cache_salt'], type_or_value=[str, None], name="setting 'cache_salt'")

        # use_activation
        assert_type_value(obj=settings['use_activation'], type_or_value=[bool, None], name="setting 'use_activation'")

        # add_special_tokens
        assert_type_value(obj=settings['add_special_tokens'], type_or_value=bool, name="setting 'add_special_tokens'")

        # add_generation_prompt
        assert_type_value(obj=settings['add_generation_prompt'], type_or_value=bool, name="setting 'add_generation_prompt'")

        # continue_final_message
        assert_type_value(obj=settings['continue_final_message'], type_or_value=bool, name="setting 'continue_final_message'")

        # chat_template
        assert_type_value(obj=settings['chat_template'], type_or_value=[str, None], name="setting 'chat_template'")

        # chat_template_kwargs
        assert_type_value(obj=settings['chat_template_kwargs'], type_or_value=[dict, None], name="setting 'chat_template_kwargs'")

        # media_io_kwargs
        assert_type_value(obj=settings['media_io_kwargs'], type_or_value=[dict, None], name="setting 'media_io_kwargs'")

        # timeout_connect
        assert_type_value(obj=settings['timeout_connect'], type_or_value=[float, int, None], name="setting 'timeout_connect'")
        if settings['timeout_connect'] is not None:
            assert_log(
                expression=settings['timeout_connect'] > 0.0,
                message=f"Expected setting 'timeout_connect' to be None or greater than zero but got '{settings['timeout_connect']}'."
            )

        # timeout_read
        assert_type_value(obj=settings['timeout_read'], type_or_value=[float, int, None], name="setting 'timeout_read'")
        if settings['timeout_read'] is not None:
            assert_log(
                expression=settings['timeout_read'] > 0.0,
                message=f"Expected setting 'timeout_read' to be None or greater than zero but got '{settings['timeout_read']}'."
            )

        # apply settings
        self._endpoint = settings['endpoints'][settings['endpoint']]
        return self._apply_settings(settings, mode)

    def classify(self, prompt):
        stamp = time.perf_counter()

        if self._settings['mode'] == "input":
            assert_type_value(obj=prompt, type_or_value=[list, str], name=f"argument 'prompt' in mode '{self._settings['mode']}'")
        elif self._settings['mode'] == "messages":
            assert_type_value(obj=prompt, type_or_value=list, name=f"argument 'prompt' in mode '{self._settings['mode']}'")
        else:
            raise NotImplementedError(f"Unknown mode '{self._settings['mode']}'.")

        # retrieve API key
        success, message, api_key = self.get_api_key()
        if not success:
            raise UnrecoverableError(message)

        # validate connection
        if self._settings['model'] is not None:
            success, message = self.validate_connection(api_key=api_key)
            if not success:
                return False, message, None

        # construct payload
        headers = {
            'Content-Type': "application/json",
            'Authorization': f"Bearer {api_key}",
            'HTTP-Referer': "https://github.com/ptzld/nimbro-api",
            'X-Title': "NimbRo API"
        }
        if api_key == "":
            del headers['Authorization']

        if self._settings['mode'] == "input":
            self._logger.debug("Using mode 'input'.")
            data = {
                'input': prompt,
                'model': self._settings['model'],
                'truncate_prompt_tokens': self._settings['truncate_prompt_tokens'],
                'truncation_side': self._settings['truncation_side'],
                'request_id': self._settings['request_id'],
                'priority': self._settings['priority'],
                'mm_processor_kwargs': self._settings['mm_processor_kwargs'],
                'cache_salt': self._settings['cache_salt'],
                'use_activation': self._settings['use_activation'],
                'add_special_tokens': self._settings['add_special_tokens'],
            }
        elif self._settings['mode'] == "messages":
            self._logger.debug("Using mode 'messages'.")
            data = {
                'messages': prompt,
                'model': self._settings['model'],
                'truncate_prompt_tokens': self._settings['truncate_prompt_tokens'],
                'truncation_side': self._settings['truncation_side'],
                'request_id': self._settings['request_id'],
                'priority': self._settings['priority'],
                'mm_processor_kwargs': self._settings['mm_processor_kwargs'],
                'cache_salt': self._settings['cache_salt'],
                'use_activation': self._settings['use_activation'],
                'add_special_tokens': self._settings['add_special_tokens'],
                'add_generation_prompt': self._settings['add_generation_prompt'],
                'continue_final_message': self._settings['continue_final_message'],
                'chat_template': self._settings['chat_template'],
                'chat_template_kwargs': self._settings['chat_template_kwargs'],
                'media_io_kwargs': self._settings['media_io_kwargs'],
            }
        else:
            raise NotImplementedError(f"Unknown mode '{self._settings['mode']}'.")

        for key in copy.deepcopy(list(data.keys())):
            if data[key] is None:
                del data[key]

        # use API
        success, message, classification = post_request(
            api_name="Classify API",
            api_url=self._endpoint['api_url'],
            headers=headers,
            data=data,
            timeout=(self._settings['timeout_connect'], self._settings['timeout_read']),
            logger=self._logger
        )
        if success:
            # parse API response
            try:
                classification = classification.json()
            except Exception:
                classification = classification.text.strip()
            if isinstance(classification, str):
                classification_str = classification
            else:
                try:
                    assert_type_value(obj=classification, type_or_value=dict, name="classification response")
                    # assert_log(expression='text' in classification, message=f"Expected classification response to contain the key 'text': {format_obj(classification)}")
                    # assert_type_value(obj=classification['text'], type_or_value=str, name="key 'text' in classification response")
                except UnrecoverableError as e:
                    success = False
                    message = str(e)
                else:
                    if self._settings['message_results'] and 'data' in classification:
                        classification_str = f": {classification}"
                    else:
                        classification_str = "."

        # finalize response
        if success:
            message = f"Classified request in '{time.perf_counter() - stamp:.3f}s'{classification_str}"
        else:
            classification = None

        return success, message, classification
