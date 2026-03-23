import os
import time

from nimbro_api.client import ClientBase
from nimbro_api.utility.io import parse_audio_bytes
from nimbro_api.utility.api import get_api_key, validate_endpoint, post_request
from nimbro_api.utility.misc import UnrecoverableError, assert_type_value, assert_log, format_obj
from ..utility import validate_connection, get_models

class TranslationsBase(ClientBase):

    def __init__(self, settings, default_settings, **kwargs):
        super().__init__(settings=settings, default_settings=default_settings, **kwargs)
        self.get_api_key = get_api_key.__get__(self)
        self.validate_connection = validate_connection.__get__(self)
        self.get_models = get_models.__get__(self)
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

        # model
        assert_type_value(obj=settings['model'], type_or_value=str, name="setting 'model'")

        # validate_model
        assert_type_value(obj=settings['validate_model'], type_or_value=[float, int, bool], name="setting 'validate_model'")
        if isinstance(settings['validate_model'], (int, float)):
            assert_log(
                expression=settings['validate_model'] >= 0,
                message=f"Expected setting 'validate_model' provided as '{type(settings['validate_model']).__name__}' to be non-negative but got '{settings['validate_model']}'."
            )

        # temperature
        assert_type_value(obj=settings['temperature'], type_or_value=[float, int], name="setting 'temperature'")
        temperature_range = [0.0, 1.0]
        assert_log(
            expression=temperature_range[0] <= settings['temperature'] <= temperature_range[1],
            message=f"Expected setting 'temperature' to be in interval [0.0, 1.0] but got '{settings['temperature']}'."
        )
        if isinstance(settings['temperature'], int):
            settings['temperature'] = float(settings['temperature'])

        # prompt
        assert_type_value(obj=settings['prompt'], type_or_value=str, name="setting 'prompt'")

        # response_format
        supported_formats = ["json", "verbose_json", "text", "srt", "vtt"]
        assert_type_value(obj=settings['response_format'], type_or_value=supported_formats, name="setting 'response_format'")

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

    def get_translation(self, audio):
        stamp = time.perf_counter()

        # parse audio
        success, message, audio_file, audio_path = parse_audio_bytes(audio=audio, logger=self._logger)
        if not success:
            return False, message, None

        # retrieve API key
        api_key = self.get_api_key()[2]

        # validate connection
        success, message = self.validate_connection(api_key=api_key)
        if not success:
            return False, message, None

        # construct payload
        headers = {
            'Authorization': f"Bearer {api_key}"
        }
        data = {
            'model': self._settings['model'],
            'temperature': self._settings['temperature'],
            'response_format': self._settings['response_format']
        }
        if self._settings['prompt'] != "":
            data['prompt'] = self._settings['prompt']

        # use API
        success, message, translation = post_request(
            api_name="Translations API",
            api_url=self._endpoint['api_url'],
            headers=headers,
            data=data,
            files={'file': ("" if audio_path is None else os.path.basename(audio_path), audio_file)},
            timeout=(self._settings['timeout_connect'], self._settings['timeout_read']),
            logger=self._logger
        )
        if success:
            # parse API response
            try:
                translation = translation.json()
            except Exception:
                translation = translation.text.strip()
            if isinstance(translation, str):
                translation_str = translation
            else:
                try:
                    assert_type_value(obj=translation, type_or_value=dict, name="translation response")
                    assert_log(expression='text' in translation, message=f"Expected translation response to contain the key 'text': {format_obj(translation)}")
                    assert_type_value(obj=translation['text'], type_or_value=str, name="key 'text' in translation response")
                except UnrecoverableError as e:
                    success = False
                    message = str(e)
                else:
                    translation_str = translation['text']
                    if self._settings['response_format'] == "text":
                        translation = translation_str

        # finalize response
        if success:
            if audio_path is None:
                message = f"Translated audio in '{time.perf_counter() - stamp:.3f}s': '{translation_str}'"
            else:
                message = f"Translated audio '{audio_path}' in '{time.perf_counter() - stamp:.3f}s': '{translation_str}'"
        else:
            translation = None

        return success, message, translation
