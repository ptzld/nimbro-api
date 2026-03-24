import os
import re
import copy
import json
import time
import uuid
import datetime
import threading
import importlib.util
import multiprocessing

import requests
try:
    import jsonschema
    JSONSCHEMA_AVAILABLE = True
except ImportError:
    JSONSCHEMA_AVAILABLE = False

from nimbro_api.client import ClientBase
from nimbro_api.utility.io import download_file, read_as_b64, encode_b64
from nimbro_api.utility.api import get_api_key, validate_endpoint
from nimbro_api.utility.misc import UnrecoverableError, assert_type_value, assert_keys, assert_log, escape, format_obj
from nimbro_api.utility.string import is_url, is_base64, extract_json
from ..utility import validate_connection, get_models

class CustomException(Exception):
    pass

class ChatCompletionsBase(ClientBase):

    def __init__(self, settings, default_settings, **kwargs):
        super().__init__(settings=settings, default_settings=default_settings, **kwargs)
        self.get_api_key = get_api_key.__get__(self)
        self.validate_connection = validate_connection.__get__(self)
        self.get_models = get_models.__get__(self)

        self.is_prompting = False
        self.messages, self.tools = [], []

        self._logger.debug(f"Initialized '{type(self).__name__}' object.")
        self._initialized = True

    def set_settings(self, settings, mode="set"):
        settings = self._introduce_settings(settings=settings, mode=mode)

        # logger_info_prompt
        assert_type_value(obj=settings['logger_info_prompt'], type_or_value=[bool, int], name="setting 'logger_info_prompt'")

        # logger_info_prompt_threshold
        assert_type_value(obj=settings['logger_info_prompt_threshold'], type_or_value=int, name="setting 'logger_info_prompt_threshold'")
        assert_log(expression=settings['logger_info_prompt_threshold'] > 0, message=f"Expected setting 'logger_info_prompt_threshold' to be greater zero but got '{settings['logger_info_prompt_threshold']}'.")

        # logger_info_prompt_cutoff
        assert_type_value(obj=settings['logger_info_prompt_cutoff'], type_or_value=int, name="setting 'logger_info_prompt_cutoff'")
        assert_log(expression=settings['logger_info_prompt_cutoff'] >= 0, message=f"Expected setting 'logger_info_prompt_cutoff' to be zero or greater but got '{settings['logger_info_prompt_cutoff']}'.")

        # logger_info_completion
        assert_type_value(obj=settings['logger_info_completion'], type_or_value=bool, name="setting 'logger_info_completion'")

        # logger_debug_chunks
        assert_type_value(obj=settings['logger_debug_chunks'], type_or_value=bool, name="setting 'logger_debug_chunks'")

        valid_flavors = ["openai", "mistral", "openrouter", "vllm"]

        # endpoints
        assert_type_value(obj=settings['endpoints'], type_or_value=dict, name="setting 'endpoints'")
        assert_log(expression=len(settings['endpoints']) > 0, message="Expected setting 'endpoints' to define at least one endpoint.")
        for endpoint in settings['endpoints']:
            assert_type_value(obj=endpoint, type_or_value=str, name="all endpoint names in setting 'endpoints'")
            assert_log(expression=len(endpoint) > 0, message="Expected all endpoint names in setting 'endpoints' to be non-empty.")
            validate_endpoint(endpoint=settings['endpoints'][endpoint], flavors=valid_flavors, require_key=True, require_name=False, setting_name=f"endpoint '{endpoint}' in setting 'endpoints'")

        # endpoint
        if isinstance(settings['endpoint'], dict):
            validate_endpoint(endpoint=settings['endpoint'], flavors=valid_flavors, require_key=True, require_name=True, setting_name="endpoint provided through setting 'endpoint'")
            settings['endpoints'][settings['endpoint']['name']] = settings['endpoint']
            settings['endpoint'] = settings['endpoint']['name']
            del settings['endpoints'][settings['endpoint']]['name']
        else:
            assert_type_value(obj=settings['endpoint'], type_or_value=list(settings['endpoints'].keys()), name="setting 'endpoint'")

        # model
        assert_type_value(obj=settings['model'], type_or_value=str, name="setting 'model'")

        # validate_model
        assert_type_value(obj=settings['validate_model'], type_or_value=[float, int, bool], name="setting 'validate_model'")
        if isinstance(settings['validate_model'], (float, int)):
            assert_log(
                expression=settings['validate_model'] >= 0,
                message=f"Expected setting 'validate_model' provided as '{type(settings['validate_model']).__name__}' to be non-negative but got '{settings['validate_model']}'."
            )

        # temperature
        assert_type_value(obj=settings['temperature'], type_or_value=[float, int], name="setting 'temperature'")
        settings['temperature'] = float(settings['temperature'])
        assert_log(
            expression=0.0 <= settings['temperature'] <= 1.5,
            message=f"Expected setting 'temperature' to be between '0.0' and '1.5' (inclusive) but got '{settings['temperature']}'."
        )

        # top_p
        assert_type_value(obj=settings['top_p'], type_or_value=[float, int], name="setting 'top_p'")
        settings['top_p'] = float(settings['top_p'])
        assert_log(
            expression=0.0 <= settings['top_p'] <= 2.0,
            message=f"Expected setting 'top_p' to be between '0.0' and '+2.0' (inclusive) but got '{settings['top_p']}'."
        )

        # max_tokens
        assert_type_value(obj=settings['max_tokens'], type_or_value=int, name="setting 'max_tokens'")
        assert_log(
            expression=settings['max_tokens'] > 0,
            message=f"Expected setting 'max_tokens' to be greater zero but got '{settings['max_tokens']}'."
        )

        # presence_penalty
        assert_type_value(obj=settings['presence_penalty'], type_or_value=[float, int], name="setting 'presence_penalty'")
        settings['presence_penalty'] = float(settings['presence_penalty'])
        assert_log(
            expression=-2.0 <= settings['presence_penalty'] <= 2.0,
            message=f"Expected setting 'presence_penalty' to be between '-2.0' and '+2.0' (inclusive) but got '{settings['presence_penalty']}'."
        )

        # frequency_penalty
        assert_type_value(obj=settings['frequency_penalty'], type_or_value=[float, int], name="setting 'frequency_penalty'")
        settings['frequency_penalty'] = float(settings['frequency_penalty'])
        assert_log(
            expression=-2.0 <= settings['frequency_penalty'] <= 2.0,
            message=f"Expected setting 'frequency_penalty' to be between '-2.0' and '+2.0' (inclusive) but got '{settings['frequency_penalty']}'."
        )

        # reasoning_effort
        assert_type_value(obj=settings['reasoning_effort'], type_or_value=str, name="setting 'reasoning_effort'")
        valid_values = ["", "none", "minimal", "low", "medium", "high"]
        assert_log(
            expression=settings['reasoning_effort'] in valid_values,
            message=f"Expected setting 'reasoning_effort' to be in {valid_values} but got '{settings['reasoning_effort']}'."
        )

        # download_image
        assert_type_value(obj=settings['download_image'], type_or_value=bool, name="setting 'download_image'")

        # download_audio
        assert_type_value(obj=settings['download_audio'], type_or_value=bool, name="setting 'download_audio'")

        # download_video
        assert_type_value(obj=settings['download_video'], type_or_value=bool, name="setting 'download_video'")

        # download_file
        assert_type_value(obj=settings['download_file'], type_or_value=bool, name="setting 'download_file'")

        # stream
        assert_type_value(obj=settings['stream'], type_or_value=bool, name="setting 'stream'")

        # max_tool_calls
        assert_type_value(obj=settings['max_tool_calls'], type_or_value=[int, None], name="setting 'max_tool_calls'")
        if isinstance(settings['max_tool_calls'], int):
            assert_log(
                expression=settings['max_tool_calls'] > 0,
                message=f"Expected setting 'max_tool_calls' to be None or greater zero but got '{settings['max_tool_calls']}'."
            )

        # correction
        assert_type_value(obj=settings['correction'], type_or_value=bool, name="setting 'correction'")

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

        # timeout_chunk_first
        assert_type_value(obj=settings['timeout_chunk_first'], type_or_value=[float, int, None], name="setting 'timeout_chunk_first'")
        if settings['timeout_chunk_first'] is not None:
            assert_log(
                expression=settings['timeout_chunk_first'] > 0.0,
                message=f"Expected setting 'timeout_chunk_first' to be None or greater zero but got '{settings['timeout_chunk_first']}'."
            )

        # timeout_chunk_next
        assert_type_value(obj=settings['timeout_chunk_next'], type_or_value=[float, int, None], name="setting 'timeout_chunk_next'")
        if settings['timeout_chunk_next'] is not None:
            assert_log(
                expression=settings['timeout_chunk_next'] > 0.0,
                message=f"Expected setting 'timeout_chunk_next' to be None or greater zero but got '{settings['timeout_chunk_next']}'."
            )

        # timeout_completion
        assert_type_value(obj=settings['timeout_completion'], type_or_value=[float, int, None], name="setting 'timeout_completion'")
        if settings['timeout_completion'] is not None:
            assert_log(
                expression=settings['timeout_completion'] > 0.0,
                message=f"Expected setting 'timeout_completion' to be None or greater zero but got '{settings['timeout_completion']}'."
            )

        # request_safeguard
        assert_type_value(obj=settings['request_safeguard'], type_or_value=bool, name="setting 'request_safeguard'")

        # parsers
        assert_type_value(obj=settings['parser'], type_or_value=[list, str], name="setting 'parser'")
        if isinstance(settings['parser'], str):
            settings['parser'] = [settings['parser']]
        for item in settings['parser']:
            assert_type_value(obj=item, type_or_value=str, name="element in setting 'parser'")

        # apply settings
        self._endpoint = settings['endpoints'][settings['endpoint']]
        return self._apply_settings(settings, mode)

    # Utilities

    def check_message_validity(self, message):
        if not isinstance(message, dict):
            raise CustomException(f"Message must be of type 'dict' but it is of type {type(message).__name__}.")
        if 'role' not in message:
            raise CustomException("Message must contain key 'role'.")
        if message['role'] not in ['system', 'user', 'assistant', 'tool']:
            raise CustomException(f"Message must contain key 'role' with value in ['system', 'user', 'assistant', 'tool'] but it is '{message['role']}'.")

        if message['role'] == 'system':
            if 'content' not in message:
                raise CustomException("System message must contain key 'content'.")
            if not isinstance(message['content'], str):
                raise CustomException(f"System message value of key 'content' must be of type 'str' but it is of type {type(message['content']).__name__}.")
            if len(message['content']) == 0:
                raise CustomException("System message value of key 'content' must not be empty.")

            if 'name' in message:
                if not isinstance(message['name'], str):
                    raise CustomException(f"System message can contain key 'name' with value that must be of type 'str' but it is of type {type(message['name']).__name__}.")
                if len(message['name']) == 0:
                    raise CustomException("System message can contain key 'name' with a value that must not be empty.")

            for key in message:
                if key not in ['role', 'content', 'name']:
                    raise CustomException(f"System message keys must be in ['role', 'content', 'name'] which '{key}' is not.")

        if message['role'] == 'user':
            if 'content' not in message:
                raise CustomException("User message must contain key 'content'.")
            if not isinstance(message['content'], str) and not isinstance(message['content'], list):
                raise CustomException(f"User message value of key 'content' must be of type 'str' or 'list' but it is of type {type(message['content']).__name__}.")

            if isinstance(message['content'], list):
                for element in message['content']:
                    if not isinstance(element, dict):
                        raise CustomException(f"User message content elements must be of type 'dict' but it is of type {type(element).__name__}.")
                    if 'type' not in element:
                        raise CustomException("User message content element must contain key 'type'.")
                    valid_types = ["text", "image_url", "input_audio", "video_url", "file"]
                    if element['type'] not in valid_types:
                        raise CustomException(f"User message content element type must be in {valid_types} but it is '{element['type']}'.")

                    if element['type'] == "text":
                        if 'text' not in element:
                            raise CustomException("User message content element of type text must contain key 'text'.")
                        if not isinstance(element['text'], str):
                            raise CustomException(f"User message content element of type text must contain key 'text' of type 'str' but it is of type '{type(element['text']).__name__}'.")
                        if len(element['text']) == 0:
                            raise CustomException("User message content element of type text must contain key 'text' that is not empty.")
                        if not len(element) == 2:
                            raise CustomException(f"User message content element of type text must contain exactly two keys 'type' and 'text' but it contains {list(element.keys())}.")

                    elif element['type'] == "image_url":
                        if 'image_url' not in element:
                            raise CustomException("User message content element of type image_url must contain key 'image_url'.")
                        if not isinstance(element['image_url'], dict):
                            raise CustomException(f"User message content element of type image_url must be of type 'dict' but it is of type {type(element['image_url']).__name__}.")
                        if len(element['image_url']) != 2:
                            raise CustomException(f"User message content element of type image_url must contain exactly two keys 'detail' and 'url' but it contains {list(element['image_url'].keys())}.")
                        if 'detail' not in element['image_url']:
                            raise CustomException("User message content element of type image_url must contain key 'detail'.")
                        if element['image_url']['detail'] not in ["low", "high", "auto"]:
                            raise CustomException(f"User message content element of type image_url must contain key 'detail' with value in ['low', 'high', 'auto'] but it is '{element['image_url']['detail']}'.")
                        if 'url' not in element['image_url']:
                            raise CustomException("User message content element of type image_url must contain key 'url'.")
                        if not isinstance(element['image_url']['url'], str):
                            raise CustomException(f"User message content element of type image_url must contain key 'url' of type 'str' but it is of type '{type(element['image_url']['url']).__name__}'.")
                        if len(element['image_url']['url']) == 0:
                            raise CustomException("User message content element of type image_url must contain key 'url' that is not empty.")

                    elif element['type'] == "input_audio":
                        if 'input_audio' not in element:
                            raise CustomException("User message content element of type input_audio must contain key 'input_audio'.")
                        if not isinstance(element['input_audio'], dict):
                            raise CustomException(f"User message content element of type input_audio must be of type 'dict' but it is of type {type(element['input_audio']).__name__}.")
                        if len(element['input_audio']) != 2:
                            raise CustomException(f"User message content element of type input_audio must contain exactly two keys 'data' and 'format' but it contains {list(element['input_audio'].keys())}.")
                        if 'data' not in element['input_audio']:
                            raise CustomException("User message content element of type input_audio must contain key 'data'.")
                        if not isinstance(element['input_audio']['data'], str):
                            raise CustomException(f"User message content element of type input_audio must contain key 'data' of type 'str' but it is of type '{type(element['input_audio']['data']).__name__}'.")
                        if len(element['input_audio']['data']) == 0:
                            raise CustomException("User message content element of type input_audio must contain key 'data' that is not empty.")
                        if 'format' not in element['input_audio']:
                            raise CustomException("User message content element of type input_audio must contain key 'format'.")
                        if element['input_audio']['format'] not in ["wav", "mp3"]:
                            raise CustomException(f"User message content element of type input_audio must contain key 'format' with value in ['wav', 'mp3'] but it is '{element['input_audio']['format']}'.")

                    elif element['type'] == "video_url":
                        if 'video_url' not in element:
                            raise CustomException("User message content element of type video_url must contain key 'video_url'.")
                        if not isinstance(element['video_url'], dict):
                            raise CustomException(f"User message content element of type video_url must be of type 'dict' but it is of type {type(element['video_url']).__name__}.")
                        if len(element['video_url']) != 1:
                            raise CustomException(f"User message content element of type video_url must contain exactly one key 'url' but it contains {list(element['video_url'].keys())}.")
                        if 'url' not in element['video_url']:
                            raise CustomException("User message content element of type video_url must contain key 'url'.")
                        if not isinstance(element['video_url']['url'], str):
                            raise CustomException(f"User message content element of type video_url must contain key 'url' of type 'str' but it is of type '{type(element['video_url']['url']).__name__}'.")
                        if len(element['video_url']['url']) == 0:
                            raise CustomException("User message content element of type video_url must contain key 'url' that is not empty.")

                    elif element['type'] == "file":
                        if 'file' not in element:
                            raise CustomException("User message content element of type file must contain key 'file'.")
                        if not isinstance(element['file'], dict):
                            raise CustomException(f"User message content element of type file must be of type 'dict' but it is of type {type(element['file']).__name__}.")
                        if len(element['file']) != 2:
                            raise CustomException(f"User message content element of type file must contain exactly two keys 'filename' and 'file_data' but it contains {list(element['file'].keys())}.")
                        if 'filename' not in element['file']:
                            raise CustomException("User message content element of type file must contain key 'filename'.")
                        if not isinstance(element['file']['filename'], str):
                            raise CustomException(f"User message content element of type file must contain key 'filename' of type 'str' but it is of type '{type(element['file']['filename']).__name__}'.")
                        if len(element['file']['filename']) == 0:
                            raise CustomException("User message content element of type file must contain key 'filename' that is not empty.")
                        if 'file_data' not in element['file']:
                            raise CustomException("User message content element of type file must contain key 'file_data'.")
                        if not isinstance(element['file']['file_data'], str):
                            raise CustomException(f"User message content element of type file must contain key 'file_data' of type 'str' but it is of type '{type(element['file']['file_data']).__name__}'.")
                        if len(element['file']['file_data']) == 0:
                            raise CustomException("User message content element of type file must contain key 'file_data' that is not empty.")

            if 'name' in message:
                if not isinstance(message['name'], str):
                    raise CustomException(f"User message can contain key 'name' with value that must be of type 'str' but it is of type {type(message['name']).__name__}.")
                if len(message['name']) == 0:
                    raise CustomException("User message can contain key 'name' with a value that must not be empty.")

            for key in message:
                if key not in ['role', 'content', 'name']:
                    raise CustomException(f"User message keys must be in ['role', 'content', 'name'] which '{key}' is not.")

        if message['role'] == 'assistant':
            if 'content' not in message:
                raise CustomException("Assistant message must contain key 'content'.")
            if message['content'] is None:
                if 'tool_calls' not in message:
                    raise CustomException("Assistant message can only contain key 'content' with value 'None' if it also contains key 'tool_calls'.")
            elif isinstance(message['content'], str):
                if len(message['content']) == 0:
                    raise CustomException("Assistant message value of key 'content' must not be an empty string.")
            else:
                raise CustomException(f"Assistant message must contain key 'content' with value of type 'None' or 'str' but it is of type '{type(message['content']).__name__}'.")

            if 'tool_calls' in message:
                if not isinstance(message['tool_calls'], list):
                    raise CustomException(f"Assistant message key 'tool_calls' must be of type 'list' but it is of type '{type(message['tool_calls']).__name__}'.")
                for element in message['tool_calls']:
                    if not isinstance(element, dict):
                        raise CustomException(f"Assistant message elements of key 'tool_calls' must be of type 'dict' but it is of type '{type(element).__name__}'.")
                    if 'id' not in element:
                        raise CustomException("Assistant message elements of key 'tool_calls' must contain key 'id'.")
                    if not isinstance(element['id'], str):
                        raise CustomException(f"Assistant message elements of key 'tool_calls' must contain key 'id' with value of type 'str' but it of type '{type(element['id']).__name__}'.")
                    if 'type' not in element:
                        raise CustomException("Assistant message elements of key 'tool_calls' must contain key 'type'.")
                    if element['type'] != 'function':
                        raise CustomException("Assistant message elements of key 'tool_calls' must contain key 'type' with value 'function' but it is '{element['type']}'.")
                    if 'function' not in element:
                        raise CustomException("Assistant message elements of key 'tool_calls' must contain key 'function'.")
                    if not isinstance(element['function'], dict):
                        raise CustomException(f"Assistant message elements of key 'tool_calls' must contain key 'function' with value of type 'dict' but it of type '{type(element['function']).__name__}'.")
                    if 'name' not in element['function']:
                        raise CustomException("Assistant message elements of key 'tool_calls' must contain dict 'function' that must contain key 'name'.")
                    if not isinstance(element['function']['name'], str):
                        raise CustomException(f"Assistant message elements of key 'tool_calls' must contain dict 'function' that must contain key 'name' with value of type 'str' but it of type '{type(element['function']['name']).__name__}'.")
                    if len(element['function']['name']) == 0:
                        raise CustomException("Assistant message elements of key 'tool_calls' must contain dict 'function' that must contain key 'name' with value that must not be empty.")
                    if 'arguments' not in element['function']:
                        raise CustomException("Assistant message elements of key 'tool_calls' must contain dict 'function' that must contain key 'arguments'.")
                    if not isinstance(element['function']['arguments'], str):
                        raise CustomException(f"Assistant message elements of key 'tool_calls' must contain dict 'function' that must contain key 'arguments' with value of type 'str' but it of type '{type(element['function']['arguments']).__name__}'.")
                    if len(element['function']['arguments']) == 0:
                        raise CustomException("Assistant message elements of key 'tool_calls' must contain dict 'function' that must contain key 'arguments' with value that must not be empty.")
                    if not len(element['function']) == 2:
                        raise CustomException(f"Assistant message elements of key 'tool_calls' must contain dict 'function' with exactly two keys 'name' and 'arguments' but it contains {list(element['function'].keys())}.")
                    if not len(element) == 3:
                        raise CustomException(f"Assistant message elements of key 'tool_calls' must contain exactly three keys 'id', 'type' and 'functions' but it contains {list(element.keys())}.")

            if 'name' in message:
                if not isinstance(message['name'], str):
                    raise CustomException(f"Assistant message can contain key 'name' with value that must be of type 'str' but it is of type {type(message['name']).__name__}.")
                if len(message['name']) == 0:
                    raise CustomException("Assistant message can contain key 'name' with a value that must not be empty.")

            for key in message:
                if key not in ['role', 'content', 'tool_calls', 'name']:
                    raise CustomException(f"Assistant message keys must be in ['role', 'content', 'tool_calls', 'name'] which '{key}' is not.")

        awaited_tools = self.get_awaited_tools()[2]

        if message['role'] == "tool":
            if len(awaited_tools) == 0:
                raise CustomException("There is no tool response awaited.")
            if 'tool_call_id' not in message:
                raise CustomException("Tool message must contain key 'tool_call_id'.")
            if not isinstance(message['tool_call_id'], str):
                raise CustomException(f"Tool message value of key 'tool_call_id' must be of type 'str' but it is of type {type(message['tool_call_id']).__name__}.")
            if len(message['tool_call_id']) == 0:
                raise CustomException("Tool message value of key 'tool_call_id' must not be empty.")
            if message['tool_call_id'] not in awaited_tools:
                raise CustomException(f"Tool message value of key 'tool_call_id' must be in list of awaited responses {awaited_tools} but it is '{message['tool_call_id']}'.")
            if message['tool_call_id'] != awaited_tools[0]:
                raise CustomException(f"Tool message value of key 'tool_call_id' must be the first in list of awaited responses {awaited_tools} but it is '{message['tool_call_id']}'.")

            if 'content' not in message:
                raise CustomException("Tool message must contain key 'content'.")
            if not isinstance(message['content'], str):
                raise CustomException(f"Tool message value of key 'content' must be of type 'str' but it is of type {type(message['content']).__name__}.")
            if len(message['content']) == 0:
                raise CustomException("Tool message value of key 'content' must not be empty.")
            if not len(message) == 3:
                raise CustomException(f"Tool message must contain exactly three keys 'role', 'content' and 'tool_call_id' but it contains {list(message.keys())}.")
        elif len(awaited_tools) == 1:
            raise CustomException(f"Expected message responding to awaited tool-call '{awaited_tools[0]}'.")
        elif len(awaited_tools) > 1:
            raise CustomException(f"Expected message responding to awaited tool-calls: {awaited_tools}")

    def encode_files(self, message):
        lut = {
            'image_url': {
                'name': "image",
                'prefix': "data:image/jpeg;base64,",
                'data': "url"
            },
            'input_audio': {
                'name': "audio",
                'prefix': "",
                'data': "data"
            },
            'video_url': {
                'name': "video",
                'prefix': "data:video/mp4;base64,",
                'data': "url"
            },
            'file': {
                'name': "file",
                'prefix': "data:application/pdf;base64,",
                'data': "file_data"
            }
        }

        if message['role'] == "user":
            if isinstance(message['content'], list):
                for i, element in enumerate(message['content']):
                    for modality in lut:
                        if element['type'] == modality:
                            if lut[modality]['prefix'] != "" and element[modality][lut[modality]['data']][:len(lut[modality]['prefix'])] == lut[modality]['prefix']:
                                self._logger.debug(f"Provided {lut[modality]['name']} is Base64-encoded.")
                            elif is_base64(element[modality][lut[modality]['data']]):
                                self._logger.debug(f"Provided {lut[modality]['name']} is Base64-encoded without prefix.")
                                message['content'][i][modality][lut[modality]['data']] = f"{lut[modality]['prefix']}{element[modality][lut[modality]['data']]}"
                            elif is_url(element[modality][lut[modality]['data']]):
                                self._logger.debug(f"Provided {lut[modality]['name']} is a valid URL.")
                                if (lut[modality]['name'] == "image" and self._settings['download_image']) or \
                                    (lut[modality]['name'] == "audio" and self._settings['download_audio']) or \
                                    (lut[modality]['name'] == "video" and self._settings['download_video']) or \
                                        (lut[modality]['name'] == "file" and self._settings['download_file']):
                                    success, _message, download_bytes = download_file(url=element[modality][lut[modality]['data']], name=f"{lut[modality]['name']}", retry=1, logger=self._logger)
                                    if success:
                                        self._logger.debug(_message)
                                        success, _message, download_b64 = encode_b64(obj=download_bytes, name=f"{lut[modality]['name']}", logger=self._logger)
                                        if success:
                                            self._logger.debug(_message)
                                            message['content'][i][modality][lut[modality]['data']] = f"{lut[modality]['prefix']}{download_b64}"
                                        else:
                                            raise CustomException(_message)
                                    else:
                                        raise CustomException(_message)
                            elif os.path.exists(element[modality][lut[modality]['data']]):
                                if os.path.isfile(element[modality][lut[modality]['data']]):
                                    success, _message, b64_encoded = read_as_b64(
                                        file_path=element[modality][lut[modality]['data']],
                                        name=f"{lut[modality]['name']}",
                                        logger=self._logger
                                    )
                                    if success:
                                        self._logger.debug(_message)
                                        message['content'][i][modality][lut[modality]['data']] = f"{lut[modality]['prefix']}{b64_encoded}"
                                    else:
                                        raise CustomException(_message)
                                else:
                                    raise CustomException(f"Provided {lut[modality]['name']} points to folder '{element[modality][lut[modality]['data']]}'.")
                            else:
                                raise CustomException(f"Provided {lut[modality]['name']} is neither Base64-encoded, a valid local path, or a web URL.")
                            break
        return message

    def validate_function_properties(self, schema, function_name, path, strict):
        if not isinstance(schema, dict):
            return False, f"Function '{function_name}' does not satisfy the required format: '{path}' must be a dict."

        if schema.get('type') != "object":
            return False, f"Function '{function_name}' does not satisfy the required format: '{path}' must be of type 'object'."

        properties = schema.get("properties")
        if not isinstance(properties, dict):
            return False, f"Function '{function_name}' does not satisfy the required format: '{path}' is missing a 'properties' dict."

        valid_types = ['boolean', 'string', 'number', 'null', 'object']

        # Validate each property
        for prop_name, prop in properties.items():
            prop_path = f"{path}::properties::{prop_name}"

            if not isinstance(prop, dict):
                return False, f"Function '{function_name}' does not satisfy the required format: The field '{prop_path}' must be of type 'dict'."

            if prop.get('type') == "object":
                # Recursively validate nested object
                ok, msg = self.validate_function_properties(schema=prop, function_name=function_name, path=prop_path, strict=strict)
                if not ok:
                    return False, msg
            else:
                keys = set(prop.keys())
                if not keys.issubset({'type', 'description', 'enum'}):
                    return False, f"Function '{function_name}' does not satisfy the required format: The field '{prop_path}' must only contain 'type', 'description', and optionally 'enum'."

                if 'type' not in prop or 'description' not in prop:
                    return False, f"Function '{function_name}' does not satisfy the required format: The field '{prop_path}' must contain both 'type' and 'description'."

                if prop['type'] not in valid_types:
                    return False, f"Function '{function_name}' does not satisfy the required format: The field '{prop_path}::type' must be in {valid_types}."

                if not isinstance(prop['description'], str):
                    return False, f"Function '{function_name}' does not satisfy the required format: The field '{prop_path}::description' must be a string."

                if 'enum' in prop:
                    if not isinstance(prop['enum'], list):
                        return False, f"Function '{function_name}' does not satisfy the required format: The field '{prop_path}::enum' must be a list."

                    expected_type = str if prop['type'] == "string" else bool if prop['type'] == "boolean" else (int, float)
                    for e in prop['enum']:
                        if not isinstance(e, expected_type):
                            return False, f"Function '{function_name}' does not satisfy the required format: The field '{prop_path}::enum' must only contain elements of type '{expected_type}' but got '{type(e).__name__}'."

        # Validate 'required'
        if "required" in schema:
            required_list = schema["required"]
            if not isinstance(required_list, list):
                return False, f"Function '{function_name}' does not satisfy the required format: '{path}::required' must be a list."

            for r in required_list:
                if not isinstance(r, str):
                    return False, f"Function '{function_name}' does not satisfy the required format: All items in '{path}::required' must be strings."
                if r not in properties:
                    return False, f"Function '{function_name}' does not satisfy the required format: Required key '{r}' in '{path}::required' is not defined in 'properties'."

        # Validate 'additionalProperties' if 'strict' is true
        if strict:
            if "additionalProperties" not in schema:
                return False, f"Function '{function_name}' does not satisfy the required format: '{path}' must include 'additionalProperties' when 'strict' is True."
            if not isinstance(schema["additionalProperties"], bool):
                return False, f"Function '{function_name}' does not satisfy the required format: '{path}::additionalProperties' must be a boolean."
            if schema["additionalProperties"] is True:
                return False, f"Function '{function_name}' does not satisfy the required format: '{path}::additionalProperties' must be False when 'strict' is True."

        return True, "Properties are valid."

    # Prompt Pipeline

    def log_prompt(self, mode, response_type):
        assert mode in ["history", "reset", "add"], mode
        if self._settings['logger_info_prompt'] is not False:
            if mode != "history":
                if mode == "reset":
                    insert_1 = "Cleared context and added request"
                else:
                    insert_1 = "Added request to context"

                if response_type == "none":
                    insert_2 = "without generating a completion"
                elif response_type == "text":
                    insert_2 = "before generating a text-completion"
                elif response_type == "json":
                    insert_2 = "before generating JSON"
                elif response_type == "auto":
                    insert_2 = "before generating a completion"
                elif response_type == "always":
                    if self._settings['max_tool_calls'] is None:
                        insert_2 = "before generating any number of tool-calls"
                    elif self._settings['max_tool_calls'] == 1:
                        insert_2 = "before generating a tool-call"
                    else:
                        insert_2 = f"before generating up to '{self._settings['max_tool_calls']}' tool-calls"
                else:
                    insert_2 = f"before generating a '{response_type}' tool-call"

                self._logger.info(f"{insert_1} {insert_2}.")

            if self._settings['logger_info_prompt'] != 0 and len(self.messages) > 0:
                messages_fmt = []
                if self._settings['logger_info_prompt'] is True:
                    message_range = range(len(self.messages))
                elif self._settings['logger_info_prompt'] > 0:
                    message_range = range(max(0, len(self.messages) - self._settings['logger_info_prompt']), len(self.messages), 1)
                else:
                    message_range = range(min(len(self.messages), -self._settings['logger_info_prompt']))
                for i in message_range:
                    if isinstance(self.messages[i]['content'], str):
                        if self.messages[i]['role'] == "tool":
                            messages_fmt.append(f"{escape['blue']}>{escape['end']} {i} {self.messages[i]['role']} '{self.messages[i]['tool_call_id']}'{escape['bold']}:{escape['end']} {self.format_message(self.messages[i]['content'])}")
                        else:
                            messages_fmt.append(f"{escape['blue']}>{escape['end']} {i} {self.messages[i]['role']}{escape['bold']}:{escape['end']} {self.format_message(self.messages[i]['content'])}")
                    elif isinstance(self.messages[i]['content'], list):
                        num_contents = len(self.messages[i]['content'])
                        for j, content in enumerate(self.messages[i]['content']):
                            if num_contents > 1:
                                suffix = f".{j}"
                            else:
                                suffix = ""
                            if self.messages[i]['role'] == "tool":
                                messages_fmt.append(f"{escape['blue']}>{escape['end']} {i}{suffix} {self.messages[i]['role']} '{self.messages[i]['tool_call_id']}'{escape['bold']}:{escape['end']} {self.format_message(content)}")
                            else:
                                messages_fmt.append(f"{escape['blue']}>{escape['end']} {i}{suffix} {self.messages[i]['role']}{escape['bold']}:{escape['end']} {self.format_message(content)}")

                    if 'tool_calls' in self.messages[i]:
                        if len(self.messages[i]['tool_calls']) == 1:
                            messages_fmt.append(f"{escape['blue']}>{escape['end']} {i} {self.messages[i]['role']}{escape['bold']}:{escape['end']} {format_obj(self.messages[i]['tool_calls'][0])} (TOOL)".replace("\n", "\n| "))
                        else:
                            for j, tool_call in enumerate(self.messages[i]['tool_calls']):
                                messages_fmt.append(f"{escape['blue']}>{escape['end']} {i}.{j} {self.messages[i]['role']}{escape['bold']}:{escape['end']} {format_obj(tool_call)} (TOOL)".replace("\n", "\n| "))

                messages_fmt = '\n'.join(messages_fmt)
                self._logger.info(f"{escape['blue']}{escape['bold']}{escape['underline']}Prompt{escape['end']}:\n{messages_fmt}")

    def format_message(self, content):
        assert_type_value(obj=content, type_or_value=[str, dict], name="argument 'content'")

        threshold = self._settings['logger_info_prompt_threshold']
        cutoff = self._settings['logger_info_prompt_cutoff']

        if isinstance(content, str):
            if len(content) > threshold:
                message = f"'{content[:threshold]}...'"
            else:
                message = f"'{content}'"
        else:
            assert_keys(obj=content, keys=['type'], mode="required", name="content")
            assert_type_value(obj=content['type'], type_or_value=['text', 'image_url', 'input_audio', 'video_url', 'file'], name="content type")
            if content['type'] == "text":
                if len(content['text']) > threshold:
                    message = f"'{content['text'][:threshold]}...'"
                else:
                    message = f"'{content['text']}'"
            elif content['type'] == "image_url":
                if len(content['image_url']['url']) > threshold:
                    if cutoff > 0:
                        message = f"'{content['image_url']['url'][:cutoff]}...'"
                    else:
                        message = "..."
                else:
                    message = f"'{content['image_url']['url']}'"
                message += f" (IMAGE, detail '{content['image_url']['detail']}')"
            elif content['type'] == "input_audio":
                if len(content['input_audio']['data']) > threshold:
                    if cutoff > 0:
                        message = f"'{content['input_audio']['data'][:cutoff]}...'"
                    else:
                        message = "..."
                else:
                    message = f"'{content['input_audio']['data']}'"
                message += f" (AUDIO, format '{content['input_audio']['format']}')"
            elif content['type'] == "video_url":
                if len(content['video_url']['url']) > threshold:
                    if cutoff > 0:
                        message = f"'{content['video_url']['url'][:cutoff]}...'"
                    else:
                        message = "..."
                else:
                    message = f"'{content['video_url']['url']}'"
                message += " (VIDEO)"
            elif content['type'] == "file":
                if len(content['file']['file_data']) > threshold:
                    if cutoff > 0:
                        message = f"'{content['file']['file_data'][:cutoff]}...'"
                    else:
                        message = "..."
                else:
                    message = f"'{content['file']['file_data']}'"
                message += f" (FILE, name '{content['file']['filename']}')"
            else:
                raise NotImplementedError(content['type'])

        message = message.replace("\n", "\n| ")

        return message

    def generate_completion(self, response_type):
        # retrieve API key
        try:
            api_key = self.get_api_key()[2]
        except UnrecoverableError as e:
            self._logger.debug("Restoring original context.")
            self.messages = self.context_dump
            raise e

        # validate connection
        success, message = self.validate_connection(api_key=api_key)
        if not success:
            self._logger.debug("Restoring original context.")
            self.messages = self.context_dump
            return False, message, None

        is_valid, is_correction, logs = False, False, []
        self.set_tool_choice(response_type)

        while True:
            reasoning, text, tool_calls, usage, is_complete, stamp_last_chunk = "", "", [], None, False, None
            datetime_start = datetime.datetime.now(datetime.timezone.utc)
            stamp_start = time.perf_counter()

            # start completion thread
            self.pipe = multiprocessing.Pipe(duplex=True)
            self.is_prompting = True
            request_thread = threading.Thread(target=self.completion_thread, kwargs={'pipe': self.pipe, 'api_key': api_key})
            request_thread.daemon = True
            request_thread.start()

            # receive response
            while True:
                now = time.perf_counter()

                if self._settings['timeout_completion'] is not None and now - stamp_start > self._settings['timeout_completion']:
                    self.pipe[0].send("INTERNAL")
                    usage = self.save_usage(None, datetime_start)
                    logs.append(f"Error while receiving completion: Timeout after '{self._settings['timeout_completion']}s' before completion was finished.")
                    break

                if self._settings['stream'] is True:
                    if stamp_last_chunk is None:
                        if self._settings['timeout_chunk_first'] is not None and now - stamp_start > self._settings['timeout_chunk_first']:
                            self.pipe[0].send("INTERNAL")
                            usage = self.save_usage(None, datetime_start)
                            logs.append(f"Error while receiving completion: Timeout after '{self._settings['timeout_chunk_first']}s' without receiving the first chunk.")
                            break
                    elif self._settings['timeout_chunk_next'] is not None and now - stamp_last_chunk > self._settings['timeout_chunk_next']:
                        self.pipe[0].send("INTERNAL")
                        usage = self.save_usage(None, datetime_start)
                        logs.append(f"Error while receiving completion: Timeout after '{self._settings['timeout_chunk_next']}s' without receiving the next chunk.")
                        break

                alive = request_thread.is_alive()

                if self.pipe[0].poll():
                    stamp_last_chunk = time.perf_counter()

                    chunk = self.pipe[0].recv()
                    assert isinstance(chunk, dict), f"Expected chunk '{chunk}' to be of type 'dict' but got '{type(chunk).__name__}'."
                    assert set(chunk.keys()) == {'code', 'content'}, f"Expected chunk '{chunk}' to have keys 'code' and 'content'."
                    assert isinstance(chunk['code'], str), f"Expected chunk code '{chunk['code']}' to be of type 'str' but got '{type(chunk['code']).__name__}'."
                    assert chunk['code'] in ['INTERRUPT', 'ERROR', 'COMPLETION', 'USAGE', 'ALL_CHUNKS_RECEIVED'], f"Expected chunk code '{chunk['code']}' to be in ['INTERRUPT', 'ERROR', 'COMPLETION', 'USAGE', 'ALL_CHUNKS_RECEIVED']."

                    if chunk['code'] == "ERROR" or chunk['code'] == 'INTERRUPT':
                        assert isinstance(chunk['content'], str), f"Expected chunk content '{chunk['content']}' to be of type 'str' but got '{type(chunk['content']).__name__}'."

                        if usage is None:
                            usage = self.save_usage(None, datetime_start)

                        logs.append(chunk['content'])
                        if len(logs[-1]) > 5000:
                            logs[-1] = logs[-1][:5000] + "..."

                        if chunk['code'] == 'INTERRUPT':
                            if self._settings['request_safeguard'] and request_thread.is_alive():
                                stamp = time.perf_counter()
                                self._logger.info("Waiting for completion-thread to terminate before acknowledging interrupt.")
                                request_thread.join()
                                self._logger.warn(f"Joined completion-thread after waiting '{time.perf_counter() - stamp:.3f}s'.")

                            response = self.post_process_completion(
                                response_type=response_type,
                                reasoning=None,
                                text=None,
                                tool_calls=[],
                                is_valid=False,
                                correction=False,
                                usage=usage,
                                logs=logs
                            )
                            raise UnrecoverableError("Interrupted completion.")
                        break

                    if chunk['code'] == "COMPLETION":
                        reasoning, text, tool_calls = self.parse_chunk(chunk['content'], reasoning, text, tool_calls)

                    elif chunk['code'] == "USAGE":
                        assert isinstance(chunk['content'], dict), f"Expected chunk content '{chunk['content']}' to be of type 'dict' but got '{type(chunk['content']).__name__}'."
                        if 'prompt_tokens' not in chunk['content']:
                            logs.append("Ignoring received usage message that misses expected key 'prompt_tokens'.")
                            self._logger.warn(logs[-1])
                        elif 'completion_tokens' not in chunk['content']:
                            logs.append("Ignoring received usage message that misses expected key 'completion_tokens'.")
                            self._logger.warn(logs[-1])
                        else:
                            usage = self.save_usage(chunk, datetime_start)

                    elif chunk['code'] == "ALL_CHUNKS_RECEIVED":
                        if reasoning == "" and text == "" and len(tool_calls) == 0:
                            logs.append("Completion finished before receiving any content.")
                            self._logger.warn(logs[-1])
                            break

                        is_complete = True

                        if usage is None:
                            usage = self.save_usage(None, datetime_start)
                            logs.append("Received completion without usage information.")
                            self._logger.warn(logs[-1])

                        # move reasoning to text when completion contains reasoning only
                        if reasoning != "" and len(tool_calls) == 0 and text == "":
                            self._logger.warn("Completion contains nothing but reasoning. Interpreting reasoning content as text-completion.")
                            text = reasoning
                            reasoning = ""

                        # extract tool-calls from text
                        if len(tool_calls) == 0 and response_type not in ["text", "auto", "json"]:
                            text, tool_calls, logs = self.extract_tool(text, tool_calls, logs)

                        # extract JSON from text
                        if response_type == "json":
                            try:
                                dict_extracted = json.loads(text)
                            except Exception:
                                dict_extracted = extract_json(text, first_over_longest=False)
                                if dict_extracted is not None:
                                    if "\n" in text:
                                        text_str = f"\n{text}"
                                    else:
                                        text_str = text
                                    logs.append(f"Extracted JSON {format_obj(dict_extracted)} from invalid text-completion: '{text_str}'.")
                                    self._logger.warn(logs[-1])
                                    text = json.dumps(dict_extracted, indent=2)
                            else:
                                text = json.dumps(dict_extracted, indent=2)

                        # ensure tool-call arguments are valid and extract them if not
                        for i, tool in enumerate(tool_calls):
                            if tool['arguments'] == "":
                                tool_calls[i]['arguments'] = r"{}"
                            else:
                                try:
                                    parameters = json.loads(tool['arguments'])
                                except Exception:
                                    parameters = extract_json(tool['arguments'], first_over_longest=False)
                                    if parameters is not None:
                                        if "\n" in tool['arguments']:
                                            text_str = f"\n{tool['arguments']}"
                                        else:
                                            text_str = tool['arguments']
                                        logs.append(f"Extracted JSON {format_obj(parameters)} from invalid tool-call arguments: '{text_str}'.")
                                        self._logger.warn(logs[-1])
                                        tool_calls[i]['arguments'] = json.dumps(parameters, indent=2)
                                else:
                                    tool_calls[i]['arguments'] = json.dumps(parameters, indent=2)

                        # ensure valid tool names
                        tool_calls, logs = self.clean_tool_calls(tool_calls, logs)

                        logs = self.add_completion_to_context(reasoning, text, tool_calls, logs)
                        break
                elif not alive:
                    usage = self.save_usage(None, datetime_start)
                    logs.append(f"Error while receiving completion: Completion thread unexpectedly died after '{now - stamp_start:.3f}s'.")
                    break
                else:
                    time.sleep(0.01)

            if is_complete:
                is_valid, correction_messages, logs = self.validate_completion(response_type, text, tool_calls, logs)
                if is_valid:
                    break
                if not is_correction and self._settings['correction']:
                    is_correction = True
                    logs[-1] = f"Completion failed after '{time.perf_counter() - stamp_start:.3f}s': {logs[-1]}"
                    logs.append("Attempting to correct invalid completion: ")
                    self._logger.warn(f"{logs[-1]}{logs[-2]}")
                    logs[-1] = f"{logs[-1]}{copy.deepcopy(self.messages[-1])}"
                    success, message = self.set_context(mode="insert", messages=correction_messages, index=0, reverse_indexing=True)
                    assert success, message
                    self._logger.debug(message)
                    self.log_prompt(mode="history", response_type=response_type)
                    if self._settings['request_safeguard'] and request_thread.is_alive():
                        stamp = time.perf_counter()
                        self._logger.warn("Waiting for old completion-thread to terminate before starting correction attempt.")
                        request_thread.join()
                        self._logger.info(f"Joined completion-thread after waiting '{time.perf_counter() - stamp:.3f}s'.")
                else:
                    logs[-1] = f"Completion failed after '{time.perf_counter() - stamp_start:.3f}s': {logs[-1]}"
                    reasoning, text, tool_calls = "", "", []
                    break
            else:
                logs[-1] = f"Completion failed after '{time.perf_counter() - stamp_start:.3f}s': {logs[-1]}"
                reasoning, text, tool_calls = "", "", []
                break

        if self._settings['request_safeguard'] and request_thread.is_alive():
            stamp = time.perf_counter()
            self._logger.info("Waiting for completion-thread to terminate before returning completion.")
            request_thread.join()
            self._logger.warn(f"Joined completion-thread after waiting '{time.perf_counter() - stamp:.3f}s'.")

        response = self.post_process_completion(
            response_type=response_type,
            reasoning=reasoning if len(reasoning) > 0 else None,
            text=text if len(text) > 0 else None,
            tool_calls=tool_calls,
            is_valid=is_valid,
            correction=is_correction,
            usage=usage,
            logs=logs
        )
        return response

    def set_tool_choice(self, response_type):
        if self._endpoint['api_flavor'] == "openai":
            if response_type == "text":
                self.response_format = {'type': "text"}
                self.tool_choice = "none"
            elif response_type == "json":
                self.response_format = {'type': "json_object"}
                self.tool_choice = "none"
            elif response_type == "always":
                self.response_format = {'type': "text"}
                self.tool_choice = "required"
            elif response_type == "auto":
                self.response_format = {'type': "text"}
                self.tool_choice = "auto"
            else:
                self.response_format = {'type': "text"}
                self.tool_choice = {'type': "function", 'function': {'name': response_type}}

        elif self._endpoint['api_flavor'] == "mistral":
            if response_type == "text":
                self.response_format = {'type': "text"}
                self.tool_choice = "none"
            elif response_type == "json":
                self.response_format = {'type': "json_object"}
                self.tool_choice = "none"
            elif response_type == "always":
                self.response_format = {'type': "text"}
                self.tool_choice = "any"
            elif response_type == "auto":
                self.response_format = {'type': "text"}
                self.tool_choice = "auto"
            else:
                self.response_format = {'type': "text"}
                self.tool_choice = {'type': "function", 'function': {'name': response_type}}

        elif self._endpoint['api_flavor'] == "openrouter":
            if response_type == "text":
                self.response_format = {'type': "text"}
                self.tool_choice = "none"
            elif response_type == "json":
                self.response_format = {'type': "json_object"}
                self.tool_choice = "none"
            elif response_type == "always":
                self.response_format = {'type': "text"}
                self.tool_choice = "required"
            elif response_type == "auto":
                self.response_format = {'type': "text"}
                self.tool_choice = "auto"
            else:
                self.response_format = {'type': "text"}
                self.tool_choice = {'type': "function", 'function': {'name': response_type}}

        elif self._endpoint['api_flavor'] == "vllm":
            if response_type == "text":
                self.response_format = {'type': "text"}
                self.tool_choice = "none"
            elif response_type == "json":
                self.response_format = {'type': "json_object"}
                # self.response_format = {'type': "text"} # set this to deactivate JSON-mode; response with invalid JSON will still trigger self-correction.
                self.tool_choice = "none"
            elif response_type == "always":
                self.response_format = {'type': "text"}
                self.tool_choice = "auto" # wait until v1 engines supports 'required'
                self._logger.warn(f"Tool-choice '{response_type}' is not available for API flavor '{self._endpoint['api_flavor']}', using '{self.tool_choice}' instead.")
            elif response_type == "auto":
                self.response_format = {'type': "text"}
                self.tool_choice = "auto"
            else:
                self.response_format = {'type': "text"}
                self.tool_choice = {'type': "function", 'function': {'name': response_type}}

        else:
            raise NotImplementedError(f"Undefined API flavor '{self._endpoint['api_flavor']}'.")

    def completion_thread(self, pipe, api_key):
        self._logger.debug("Starting completion thread.")

        messages = copy.deepcopy(self.messages)

        # condense consecutive user messages
        while True:
            is_user = 0
            for i, message in enumerate(messages):
                if message['role'] == "user":
                    is_user += 1
                if is_user > 1 and (message['role'] != "user" or i == len(messages) - 1):
                    first = i - is_user
                    last = i - 1
                    if i == len(messages) - 1:
                        first += 1
                        last += 1
                    contents = []
                    for j in range(first, last + 1, 1):
                        for k in range(len(messages[j]['content'])):
                            contents.append(messages[j]['content'][k])
                    self._logger.debug(f"Condensing '{len(contents)}' consecutive user messages ('{first}' to '{last}') into a single one.")
                    new_message = {'role': "user", 'content': contents}
                    messages = messages[: first] + [new_message] + messages[last + 1:]
                    break
                if message['role'] != "user":
                    is_user = 0
            else:
                break

        messages_print = copy.deepcopy(messages)
        for i, message in enumerate(messages_print):
            if message['role'] == "user":
                if isinstance(message['content'], list):
                    for j, element in enumerate(message['content']):
                        if element['type'] == "image_url":
                            if self._endpoint['api_flavor'] == "vllm":
                                self._logger.debug(f"Temporarily stripping image detail '{message['content'][j]['image_url']['detail']}' for using vLLM.")
                                del messages[i]['content'][j]['image_url']['detail']
                                del messages_print[i]['content'][j]['image_url']['detail']
                            messages_print[i]['content'][j]['image_url']['url'] = "<IMAGE>"
                        elif element['type'] == "input_audio":
                            # if self._endpoint['api_flavor'] == "mistral":
                            #     self._logger.debug(f"Temporarily stripping audio format '{message['content'][j]['input_audio']['format']}' for using Mistral AI.")
                            #     del messages[i]['content'][j]['input_audio']['format']
                            #     del messages_print[i]['content'][j]['input_audio']['format']
                            messages_print[i]['content'][j]['input_audio']['data'] = "<AUDIO>"
                        elif element['type'] == "video_url":
                            messages_print[i]['content'][j]['video_url']['url'] = "<VIDEO>"
                        elif element['type'] == "file":
                            messages_print[i]['content'][j]['file']['file_data'] = "<FILE>"
            messages_print[i] = f"{i}: " + str(messages_print[i]).replace("\n", "\\n")

        self._logger.debug("Context:\n" + str('\n'.join(messages_print)))

        try:
            headers = {
                'Authorization': f"Bearer {api_key}",
                "HTTP-Referer": "https://github.com/AIS-Bonn/nimbro_api",
                "X-Title": "NimbRo-API",
                'Content-Type': "application/json",
            }

            if self._endpoint['api_flavor'] == "openai":
                data = {
                    'model': self._settings['model'],
                    'messages': messages,
                    'tools': self.tools,
                    'temperature': self._settings['temperature'],
                    'top_p': self._settings['top_p'],
                    'max_completion_tokens': self._settings['max_tokens'],
                    'presence_penalty': self._settings['presence_penalty'],
                    'frequency_penalty': self._settings['frequency_penalty'],
                    'response_format': self.response_format,
                    # 'verbosity': "medium",
                    'n': 1,
                    'stream': self._settings['stream']
                }
                if self._settings['reasoning_effort'] not in ["", "none"]:
                    data['reasoning_effort'] = self._settings['reasoning_effort']
                if len(self.tools) == 0:
                    del data['tools']
                else:
                    data['tool_choice'] = self.tool_choice
                    if self._settings['model'][0] != "o":
                        data['parallel_tool_calls'] = self._settings['max_tool_calls'] is None or self._settings['max_tool_calls'] > 1
                if self._settings['stream'] is True:
                    data['stream_options'] = {'include_usage': True, 'include_obfuscation': False}

            elif self._endpoint['api_flavor'] == "mistral":
                data = {
                    'model': self._settings['model'],
                    'messages': messages,
                    'tools': None if len(self.tools) == 0 else self.tools,
                    'tool_choice': self.tool_choice,
                    'temperature': self._settings['temperature'],
                    'top_p': self._settings['top_p'],
                    'max_tokens': self._settings['max_tokens'],
                    'response_format': self.response_format,
                    'n': 1,
                    'stream': self._settings['stream']
                }

            elif self._endpoint['api_flavor'] == "openrouter":
                data = {
                    'model': self._settings['model'],
                    'messages': messages,
                    'tools': None if len(self.tools) == 0 else self.tools,
                    'temperature': self._settings['temperature'],
                    'top_p': self._settings['top_p'],
                    'max_tokens': self._settings['max_tokens'],
                    'presence_penalty': self._settings['presence_penalty'],
                    'frequency_penalty': self._settings['frequency_penalty'],
                    'response_format': self.response_format,
                    'n': 1,
                    'stream': self._settings['stream']
                }
                if self._settings['reasoning_effort'] not in ["", "none"]:
                    data['reasoning'] = {
                        'effort': self._settings['reasoning_effort'],
                        'exclude': False
                    }
                if len(self.tools) > 0:
                    data['tool_choice'] = self.tool_choice
                if self._settings['stream'] is True:
                    data['stream_options'] = {'include_usage': True}

            elif self._endpoint['api_flavor'] == "vllm":
                data = {
                    'model': self._settings['model'],
                    'messages': messages,
                    'tools': None if len(self.tools) == 0 else self.tools,
                    'temperature': self._settings['temperature'],
                    'top_p': self._settings['top_p'],
                    'max_tokens': self._settings['max_tokens'],
                    'presence_penalty': self._settings['presence_penalty'],
                    'frequency_penalty': self._settings['frequency_penalty'],
                    'response_format': self.response_format,
                    'n': 1,
                    'stream': self._settings['stream'],
                    'chat_template_kwargs': {'enable_thinking': self._settings['reasoning_effort'] not in ["", "none"]}
                }
                if len(self.tools) > 0:
                    data['tool_choice'] = self.tool_choice
                    data['parallel_tool_calls'] = self._settings['max_tool_calls'] is None or self._settings['max_tool_calls'] > 1
                if self._settings['stream'] is True:
                    data['stream_options'] = {'include_usage': True}

            else:
                message = f"Undefined API flavor '{self._endpoint['api_flavor']}'."
                pipe[1].send({'code': "ERROR", 'content': message})
                return

            self._logger.debug("Sending POST request.")
            completion = requests.post(self._endpoint['api_url'], headers=headers, json=data, stream=self._settings['stream'], timeout=(self._settings['timeout_connect'], self._settings['timeout_read']))

            if not self._settings['stream']:
                if completion.status_code != 200:
                    try:
                        response_str = completion.json()
                    except Exception:
                        response_str = completion.text.strip()
                    else:
                        while True:
                            if isinstance(response_str, dict):
                                if response_str.get('code') == completion.status_code:
                                    del response_str['code']
                                if len(response_str) == 1:
                                    response_str = response_str[list(response_str.keys())[0]]
                                else:
                                    break
                            else:
                                break

                    if len(str(response_str).strip()) == 0:
                        message = f"Received unexpected HTTP status code '{completion.status_code}'."
                    else:
                        message = f"Received unexpected HTTP status code '{completion.status_code}': {format_obj(response_str)}."

                    pipe[1].send({'code': "ERROR", 'content': message})
                else:
                    try:
                        json_data = completion.json()
                    except Exception as e:
                        message = f"Error while receiving completion: Failed to parse POST response as JSON: {repr(e)}"
                        pipe[1].send({'code': "ERROR", 'content': message})
                    else:
                        self._logger.debug(f"POST response: {format_obj(json_data)}.")
                        # usage
                        if 'usage' in json_data:
                            pipe[1].send({'code': "USAGE", 'content': json_data['usage']})
                        # choices
                        if 'choices' not in json_data:
                            message = "Error while receiving completion: Expected POST response to contain key 'choices'."
                            pipe[1].send({'code': "ERROR", 'content': message})
                        elif not isinstance(json_data['choices'], list):
                            message = f"Error while receiving completion: Expected value of key 'choices' to be of type 'list' but got '{type(json_data['choices']).__name__}'."
                            pipe[1].send({'code': "ERROR", 'content': message})
                        elif len(json_data['choices']) == 0:
                            message = "Error while receiving completion: Expected list 'choices' to contain at least one element."
                            pipe[1].send({'code': "ERROR", 'content': message})
                        # finish_reason
                        elif 'finish_reason' not in json_data['choices'][0]:
                            message = "Error while receiving completion: Expected choice to contain key 'finish_reason'."
                            pipe[1].send({'code': "ERROR", 'content': message})
                        elif json_data['choices'][0]['finish_reason'] not in [None, "stop", "tool_calls", "STOP", "end_turn"]:
                            message = f"Error while receiving completion: Expected value of key 'finish_reason' to be in '{[None, 'stop', 'tool_calls', 'STOP', 'end_turn']}' but got '{json_data['choices'][0]['finish_reason']}'."
                            pipe[1].send({'code': "ERROR", 'content': message})
                        # message
                        elif 'message' not in json_data['choices'][0]:
                            message = "Error while receiving completion: Expected choice to contain key 'message'."
                            pipe[1].send({'code': "ERROR", 'content': message})
                        elif not isinstance(json_data['choices'][0]['message'], dict):
                            message = f"Error while receiving completion: Expected value of key 'message' to be of type 'dict' but got '{type(json_data['choices'][0]['message']).__name__}'."
                            pipe[1].send({'code': "ERROR", 'content': message})
                        else:
                            completion = json_data['choices'][0]['message']
                            if 'tool_calls' in completion and completion['tool_calls'] is None:
                                del completion['tool_calls']
                            pipe[1].send({'code': "COMPLETION", 'content': completion})
                            pipe[1].send({'code': "ALL_CHUNKS_RECEIVED", 'content': ''})

                self._logger.debug("Finished completion thread.")
                return

        except Exception as e:
            message = f"Failed to POST request: {repr(e)}"
            pipe[1].send({'code': "ERROR", 'content': message})

        else:
            self._logger.debug("Sent POST request.")

            decoded_buffer = ""
            undecoded_buffer = b""
            error = ""
            early_stop = False
            usage = None

            try:
                for chunk in completion.iter_content(chunk_size=1):
                    # self._logger.debug(f"Chunk: {format_obj(chunk)}.")
                    if early_stop is True:
                        break
                    decoded = False

                    # check if response was canceled from external source
                    if pipe[1].poll():
                        code = pipe[1].recv()
                        if code == "EXTERNAL":
                            message = "Interrupted completion."
                            self._logger.debug(message)
                            pipe[1].send({'code': "INTERRUPT", 'content': message})
                        else:
                            self._logger.debug("Completion was interrupted due to request from internal source.")
                        break

                    # attempt to decode chunk
                    if chunk:
                        if len(undecoded_buffer) > 0:
                            try:
                                decoded_chunk = (undecoded_buffer + chunk).decode('utf-8')
                            except UnicodeDecodeError:
                                try:
                                    decoded_chunk = chunk.decode('utf-8')
                                except UnicodeDecodeError:
                                    undecoded_buffer += chunk
                                else:
                                    decoded = True
                                    self._logger.warn(f"Ignoring byte sequence '{undecoded_buffer}' after failure to decode it.")
                                    undecoded_buffer = b""
                            else:
                                decoded = True
                                undecoded_buffer = b""
                        else:
                            try:
                                decoded_chunk = chunk.decode('utf-8')
                            except UnicodeDecodeError:
                                undecoded_buffer += chunk
                            else:
                                decoded = True

                    # process all decoded lines
                    if decoded:
                        decoded_buffer += decoded_chunk
                        # self._logger.debug(f"{"\n" in decoded_buffer}: decoded_buffer: {decoded_buffer.replace("\n", "\\n")}")
                        while '\n' in decoded_buffer:
                            line, decoded_buffer = decoded_buffer.split('\n', 1)
                            # self._logger.debug(f"line: {line}")
                            if line != "":
                                if line.find('data:') == 0:

                                    # end of response
                                    if line == 'data: [DONE]':
                                        # forward usage before end of process
                                        if self._endpoint['api_flavor'] in ["vllm", "openrouter"]:
                                            if usage is None:
                                                self._logger.warn("Expected to receive usage before [DONE] message.")
                                            else:
                                                pipe[1].send({'code': "USAGE", 'content': usage})
                                        self._logger.debug("Received [DONE] message.")
                                        pipe[1].send({'code': "ALL_CHUNKS_RECEIVED", 'content': ''})
                                    else:
                                        try:
                                            json_data = json.loads(line[6:])
                                        except Exception as e:
                                            self._logger.warn(f"Ignoring line '{line}' after failure to parse it as JSON: {repr(e)}")
                                        else:
                                            # unexpected finish reason
                                            if json_data.get('finish_reason') not in [None, "stop", "tool_calls", "STOP", "end_turn"]:
                                                # forward usage before end of process
                                                if self._endpoint['api_flavor'] in ["vllm", "openrouter"]:
                                                    if usage is None:
                                                        self._logger.warn("Expected to receive usage before [ERROR] message.")
                                                    else:
                                                        pipe[1].send({'code': "USAGE", 'content': usage})
                                                message = f"Error while receiving completion: Unexpected finish reason '{json_data.get('finish_reason')}'."
                                                pipe[1].send({'code': "ERROR", 'content': message})
                                                early_stop = True
                                                break

                                            # extract usage
                                            if json_data.get('usage') is not None:
                                                if self._endpoint['api_flavor'] in ["vllm", "openrouter"]:
                                                    usage = json_data['usage']
                                                else:
                                                    pipe[1].send({'code': "USAGE", 'content': json_data['usage']})

                                            # extract choices
                                            if len(json_data.get('choices', [])) > 0:
                                                try:
                                                    json_choice = json_data['choices'][0]
                                                except Exception as e:
                                                    self._logger.warn(f"Ignoring data '{json_data}' after failure to parse choice as JSON: {repr(e)}")
                                                else:
                                                    # unexpected finish reason
                                                    if json_choice.get('finish_reason') not in [None, "stop", "tool_calls", "STOP", "end_turn"]:
                                                        # forward usage before end of process
                                                        if self._endpoint['api_flavor'] in ["vllm", "openrouter"]:
                                                            if usage is None:
                                                                self._logger.warn("Expected to receive usage before [ERROR] message.")
                                                            else:
                                                                pipe[1].send({'code': "USAGE", 'content': usage})
                                                        message = f"Error while receiving completion: Unexpected finish reason '{json_choice.get('finish_reason')}'."
                                                        pipe[1].send({'code': "ERROR", 'content': message})
                                                        early_stop = True
                                                        break

                                                    # forward delta
                                                    pipe[1].send({'code': "COMPLETION", 'content': json_choice['delta']})
                                else:
                                    error += line
                else:
                    self._logger.debug("Received full POST response.")

                    if len(undecoded_buffer) > 0:
                        self._logger.warn(f"Ignoring byte sequence '{undecoded_buffer}' after failure to decode it.")

                    if len(decoded_buffer) > 0:
                        error += decoded_buffer

                    # forward remaining usage before end of process
                    if usage is not None and self._endpoint['api_flavor'] in ["vllm", "openrouter"]:
                        pipe[1].send({'code': "USAGE", 'content': usage})

                    # forward collected error
                    if error != "":
                        message = f"Error while receiving completion: {format_obj(error)}."
                        pipe[1].send({'code': "ERROR", 'content': message})
            except Exception as e:
                message = f"Error while receiving completion: {repr(e)}."
                pipe[1].send({'code': "ERROR", 'content': message})
            else:
                completion.close()
                self._logger.debug("Connection closed.")

        self._logger.debug("Completion thread finished.")

    def save_usage(self, chunk, stamp_start):
        stamp_stop = datetime.datetime.now(datetime.timezone.utc)

        usage = {}
        # usage['api_type'] = "completions"
        # usage['api_endpoint'] = self._settings['endpoint']
        # usage['model_name'] = self._settings['model']
        usage['stamp_start'] = stamp_start.isoformat()
        usage['stamp_stop'] = stamp_stop.isoformat()
        usage['duration'] = (stamp_stop - stamp_start).total_seconds()

        if chunk is not None:
            # Ignoring everything other than 'prompt_tokens', 'cached_tokens', and 'completion_tokens' until model providers agree on a standard to deal with reasoning/audio/image tokens.
            if chunk['content']['prompt_tokens'] > 0:
                usage['tokens_input_uncached'] = chunk['content']['prompt_tokens']
            if 'prompt_tokens_details' in chunk['content'] and chunk['content']['prompt_tokens_details'] is None:
                del chunk['content']['prompt_tokens_details']
            cashed = chunk['content'].get('prompt_tokens_details', {}).get('cached_tokens', 0)
            if cashed > 0:
                usage['tokens_input_cached'] = cashed
            if chunk['content']['completion_tokens'] > 0:
                usage['tokens_output'] = chunk['content']['completion_tokens']

            # def clean_dict(d):
            #     cleaned = {}
            #     for key, value in d.items():
            #         if isinstance(value, dict):
            #             sub = clean_dict(value)
            #             if sub:
            #                 cleaned[key] = sub
            #         elif value is not None and value != 0:
            #             cleaned[key] = value
            #     return cleaned
            # tokens = clean_dict(chunk['content'])
            # for key in tokens:
            #     assert key not in usage, f"{key}"
            #     usage[key] = tokens[key]

        usage_str = json.dumps(usage, indent=2)
        self._logger.debug(f"Usage: {format_obj(usage_str)}.")

        return usage

    def parse_chunk(self, chunk, reasoning, text, tool_calls):
        assert isinstance(chunk, dict), f"Expected chunk content '{chunk}' to be of type 'dict' but got '{type(chunk).__name__}'."

        if self._settings['logger_debug_chunks']:
            self._logger.debug(f"Received chunk: {format_obj(chunk)}.")

        # chunk contains reasoning
        for key in ['reasoning', 'reasoning_content']:
            if chunk.get(key) not in ["", None]:
                if self._settings['logger_debug_chunks']:
                    self._logger.debug(f"Chunk contains '{key}'.")
                if not isinstance(chunk[key], str):
                    raise AssertionError(f"Expected value of key '{key}' to be of type 'str' but got '{type(chunk['key']).__name__}': {chunk}")
                reasoning += chunk[key]

        # chunk contains text
        if chunk.get('content') not in ["", None]:
            if self._settings['logger_debug_chunks']:
                self._logger.debug("Chunk contains 'content'.")

            if isinstance(chunk['content'], list):
                # this entire block is required solely for Mistral
                for item in chunk['content']:
                    if not isinstance(item, dict):
                        raise AssertionError(f"Expected type of item in value of key 'content' to be of type 'dict' but got '{type(item).__name__}': {chunk}")
                    if 'type' not in item:
                        raise AssertionError(f"Expected item in value of key 'content' to contain the key 'type': {chunk}")
                    if item['type'] not in ["thinking", "text"]:
                        raise AssertionError(f"Expected value of key 'type' in item of value of key 'content' to be 'thinking' or 'text': {chunk}")
                    if item['type'] == "text":
                        if 'text' not in item:
                            raise AssertionError(f"Expected item in value of key 'content' with type 'text' to contain the key 'text': {chunk}")
                        if not isinstance(item['text'], str):
                            raise AssertionError(f"Expected value of key 'text' in item of value of key 'content' to be of type 'str' but got '{type(item['text']).__name__}': {chunk}")
                        text += item['text']
                    else:
                        if 'thinking' not in item:
                            raise AssertionError(f"Expected item in value of key 'content' with type 'thinking' to contain the key 'thinking': {chunk}")
                        if not isinstance(item['thinking'], list):
                            raise AssertionError(f"Expected value of key 'thinking' in item of value of key 'content' to be of type 'list' but got '{type(item['thinking']).__name__}': {chunk}")
                        for sub_item in item['thinking']:
                            if not isinstance(sub_item, dict):
                                raise AssertionError(f"Expected type of item in value of key 'thinking' in item of value of key 'content' to be of type 'dict' but got '{type(sub_item).__name__}': {chunk}")
                            if 'type' not in sub_item:
                                raise AssertionError(f"Expected item in value of key 'thinking' in value of key 'content' to contain the key 'type': {chunk}")
                            if sub_item['type'] != "text":
                                raise AssertionError(f"Expected value of key 'type' in value of key 'thinking' in item of value of key 'content' to be 'thinking' or 'text': {chunk}")
                            if 'text' not in sub_item:
                                raise AssertionError(f"Expected item in value of key 'thinking' in value of key 'content' with type 'text' to contain the key 'text': {chunk}")
                            if not isinstance(sub_item['text'], str):
                                raise AssertionError(f"Expected value of key 'text' in value of key 'thinking' in item of value of key 'content' to be of type 'str' but got '{type(sub_item['text']).__name__}': {chunk}")
                            reasoning += sub_item['text']
            elif not isinstance(chunk['content'], str):
                raise AssertionError(f"Expected value of key 'content' to be of type 'str' but got '{type(chunk['content']).__name__}': {chunk}")
            else:
                text += chunk['content']

        # chunk contains tool-call
        if chunk.get('tool_calls') not in [[], "", None]:
            if self._settings['logger_debug_chunks']:
                self._logger.debug("Chunk contains 'tool_calls'.")
            if not isinstance(chunk['tool_calls'], list):
                raise AssertionError(f"Expected value of key 'tool_calls' to be of type 'list' but got '{type(chunk['tool_calls']).__name__}': {chunk}")
            if self._settings['logger_debug_chunks']:
                self._logger.debug(f"Chunk contains '{len(chunk['tool_calls'])}' toll call{'' if len(chunk['tool_calls']) == 1 else 's'}.")
            for i in range(len(chunk['tool_calls'])):
                if self._settings['logger_debug_chunks']:
                    self._logger.debug(f"Handling tool-call '{i}'")
                if isinstance(chunk['tool_calls'][i].get('index'), int) and isinstance(chunk['tool_calls'][i].get('function'), dict):
                    if isinstance(chunk['tool_calls'][i].get('id'), str) and len(chunk['tool_calls'][i]['id']) > 0 and isinstance(chunk['tool_calls'][i]['function'].get('name'), str) and len(chunk['tool_calls'][i]['function']['name']) > 0:
                        if len(tool_calls) == chunk['tool_calls'][i]['index']:
                            if self._settings['logger_debug_chunks']:
                                self._logger.debug(f"Appending new tool-call with index '{chunk['tool_calls'][i]['index']}'.")
                            tool_calls.append({'id': chunk['tool_calls'][i]['id'], 'name': chunk['tool_calls'][i]['function']['name'], 'arguments': ""})
                        else:
                            raise AssertionError(f"Expected value of key 'index' in value of key 'tool_calls' to be '{len(tool_calls)}' but got '{chunk['tool_calls'][i]['index']}'.")
                    if chunk['tool_calls'][i]['function'].get('arguments') not in ["", None]:
                        if chunk['tool_calls'][i]['index'] < len(tool_calls):
                            if self._settings['logger_debug_chunks']:
                                self._logger.debug(f"Appending arguments to tool-call with index '{chunk['tool_calls'][i]['index']}'.")
                            tool_calls[chunk['tool_calls'][i]['index']]['arguments'] += chunk['tool_calls'][i]['function']['arguments']
                        else:
                            raise AssertionError(f"Expected value of key 'index' in value of key 'tool_calls' to be smaller '{len(tool_calls)}' but got '{chunk['tool_calls'][i]['index']}'.")
                elif set(chunk['tool_calls'][i].keys()) == {'id', 'type', 'function'}:
                    assert isinstance(chunk['tool_calls'][i]['id'], str) and len(chunk['tool_calls'][i]['id']) > 0, f"Expected value of key 'id' in value of key 'tool_calls' to be a non-empty string but got '{chunk['tool_calls'][i]['id']}'."
                    assert chunk['tool_calls'][i]['type'] == "function", f"Expected value of key 'id' in value of key 'tool_calls' to be 'function' but got '{chunk['tool_calls'][i]['type']}'."
                    assert isinstance(chunk['tool_calls'][i]['function'], dict), f"Expected value of key 'id' in value of key 'tool_calls' to be dictionary but got '{chunk['tool_calls'][i]['function']}'."
                    assert isinstance(chunk['tool_calls'][i]['function']['name'], str) and len(chunk['tool_calls'][i]['function']['name']) > 0, f"Expected value of key 'name' in value of key 'function' to be a non-empty string but got '{chunk['tool_calls'][i]['function']['name']}'."
                    assert isinstance(chunk['tool_calls'][i]['function']['arguments'], str), f"Expected value of key 'arguments' in value of key 'function' to be a non-empty string but got '{chunk['tool_calls'][i]['function']['arguments']}'."
                    tool_calls.append({'id': chunk['tool_calls'][i]['id'], 'name': chunk['tool_calls'][i]['function']['name'], 'arguments': chunk['tool_calls'][i]['function']['arguments']})
                else:
                    raise AssertionError(f"Expected value of key 'tool_calls' to either contain the fields 'index' and 'function', or 'id', 'type' and 'function', but got {list(chunk['tool_calls'][i].keys())}.")

        return reasoning, text, tool_calls

    def extract_tool(self, text, tool_calls, logs):
        first_text_call = extract_json(text, first_over_longest=True)
        if first_text_call is not None:
            if 'name' in first_text_call and 'arguments' in first_text_call:
                if "\n" in text:
                    text_str = f"\n{text}"
                else:
                    text_str = text
                logs.append(f"Extracted tool-call {format_obj(first_text_call)} from invalid text-completion: '{text_str}'.")
                self._logger.warn(logs[-1])
                text = text.replace(first_text_call, "")
                if 'id' in first_text_call:
                    first_text_call['id'] = first_text_call['id']
                else:
                    logs.append("Generating missing ID for extracted tool-call.")
                    self._logger.warn(logs[-1])
                    made_up_id = self.get_clock().now().seconds_nanoseconds()
                    made_up_id = f"{made_up_id[0]}_{made_up_id[1]}"
                    first_text_call['id'] = made_up_id
                first_text_call['arguments'] = json.dumps(first_text_call['arguments'], indent=2)
                tool_calls.append(first_text_call)

        return text, tool_calls, logs

    def clean_tool_calls(self, tool_calls, logs):
        # I experienced OpenAI referring to undefined function names such that they contain special characters (e.g. 'assistant.tell_joke' but got 'tell_joke').
        # Meanwhile, responding to such a function would cause the completion to trigger an 'invalid function name' error due to the illegal use of special characters.
        # Therefore, we remove special characters here, establish a legal function name before letting the self correction routines check validity w.r.t. the defined JSON Schemas.
        for i, call in enumerate(tool_calls):
            if not re.match('^[a-zA-Z0-9_-]{1,64}$', call['name']):
                tool_calls[i]['name'] = re.sub(r"[^a-zA-Z0-9_-]", "", call['name'])
                tool_calls[i]['name'] = tool_calls[i]['name'][:64]
                logs.append(f"Renaming tool-call with invalid name '{call['name']}' to '{tool_calls[i]['name']}'.")
                self._logger.warn(logs[-1])
        return tool_calls, logs

    def add_completion_to_context(self, reasoning, text, tool_calls, logs):
        def print_unvalidated_tool(dictionary):
            dictionary = copy.deepcopy(dictionary)
            try:
                dictionary['arguments'] = json.loads(dictionary['arguments'])
            except Exception:
                pass
            return json.dumps(dictionary, indent=2)

        # log
        if self._settings['logger_info_completion']:
            if sum([reasoning != "", text != "", len(tool_calls) > 0]) > 1:
                response_msg = f"{escape['green']}{escape['bold']}{escape['underline']}Mixed-completion:{escape['end']}\n"
                if reasoning != "":
                    response_msg += f"\nReasoning:\n'\n{reasoning}\n'\n"
                if text != "":
                    response_msg += f"\nText:\n'\n{text}\n'\n"
                if len(tool_calls) > 0:
                    tool_msg = ',\n'.join([f"{(str(i) + ': ') if len(tool_calls) > 1 else ''}{print_unvalidated_tool(tool)}" for i, tool in enumerate(tool_calls)])
                    if len(tool_calls) > 1:
                        tool_msg = f"[{tool_msg}]"
                    response_msg += f"\nTool-call{'' if len(tool_calls) == 1 else 's'}:\n{tool_msg}\n"
            elif reasoning != "":
                response_msg = f"{escape['green']}{escape['bold']}{escape['underline']}Reasoning-completion{escape['end']}:\n'\n{reasoning}\n'"
            elif text != "":
                response_msg = f"{escape['green']}{escape['bold']}{escape['underline']}Text-completion{escape['end']}:\n{escape['green']}'{escape['end']}\n{text}\n{escape['green']}'{escape['end']}"
            elif len(tool_calls) > 0:
                tool_msg = '\n'.join([f"{(str(i) + ': ') if len(tool_calls) > 1 else ''}{print_unvalidated_tool(tool)}" for i, tool in enumerate(tool_calls)])
                response_msg = f"{escape['green']}{escape['bold']}{escape['underline']}Tool-completion{escape['end']}:\n{tool_msg}"
            else:
                response_msg = f"{escape['green']}{escape['bold']}{escape['underline']}Malformed-completion{escape['end']}:\nReasoning: {reasoning}\nText: {text}\nTool-calls: {tool_calls}\n"
            self._logger.info(response_msg)

        # construct message
        message = {}
        message['role'] = "assistant"
        if text == "":
            message['content'] = None
        else:
            message['content'] = text
        if len(tool_calls) > 0:
            message['tool_calls'] = [{} for _ in range(len(tool_calls))]
        for i, call in enumerate(tool_calls):
            message['tool_calls'][i]['type'] = "function"
            message['tool_calls'][i]['id'] = call['id']
            message['tool_calls'][i]['function'] = {}
            message['tool_calls'][i]['function']['name'] = call['name']
            message['tool_calls'][i]['function']['arguments'] = call['arguments']

        # validate message
        try:
            self.check_message_validity(message)
        except CustomException as e:
            logs.append(f"Unexpected error in validity check of completion '{message}': {e}")
            self._logger.warn(logs[-1])

        # add message to context
        self.messages.append(message)
        self._logger.debug(f"Completion added to context: {format_obj(message)}.")

        return logs

    def validate_completion(self, response_type, text, tool_calls, logs):
        is_valid = True

        # create generic correction response # TODO have a single correction response rather than number-of-tools-calls + 1 single messages

        correction_messages = []
        tool_call_is_valid_default_correction = "This tool-call is valid and does not require any correction."
        for i, call in enumerate(tool_calls):
            correction_messages.append({})
            correction_messages[-1]['role'] = "tool"
            correction_messages[-1]['tool_call_id'] = call['id']
            correction_messages[-1]['content'] = tool_call_is_valid_default_correction
        correction_messages.append({})
        correction_messages[-1]['role'] = "user"
        correction_messages[-1]['content'] = "Your response is invalid. Please correct it based on the provided error messages and try again!"

        # test error cases

        # error case: tool use when there should not be any tool use
        if (len(self.tools) == 0 or response_type == "text") and len(tool_calls) > 0:
            is_valid = False
            logs.append(f"Completion contains a tool-call despite {'no tools being defined' if len(self.tools) == 0 else 'only text was requested'}.")
            for i, message in enumerate(correction_messages):
                if 'tool_call_id' in message:
                    correction_messages[i]['content'] = "Your response must not contain any tool-call, but only text content."

        # error case: tool-choice "use specific function" was violated
        if len(self.tools) > 0 and response_type != "text" and response_type != "auto" and response_type != "always" and response_type != "json":
            if text != "":
                is_valid = False
                logs.append(f"Completion contains text content despite tool-choice being set to '{response_type}'.")
                correction_messages[-1]['content'] = f"Your response must only contain a tool-call of '{response_type}' without additional text."
            else:
                valid_ids = []
                invalid_ids_names = {}
                for c in tool_calls:
                    if c['name'] == response_type:
                        valid_ids.append(c['id'])
                    else:
                        invalid_ids_names[c['id']] = c['name']
                for i, message in enumerate(correction_messages):
                    if message['role'] == "tool":
                        if message['tool_call_id'] not in valid_ids:
                            is_valid = False
                            logs.append(f"Completion contains tool-call '{invalid_ids_names[message['tool_call_id']]}' despite tool-choice being set to '{response_type}'.")
                            correction_messages[i]['content'] = f"Your response must only contain the tool-call '{response_type}'."

        # error case: exceeding maximum number of tool-calls per response
        if self._settings['max_tool_calls'] is not None and len(tool_calls) > self._settings['max_tool_calls']:
            is_valid = False
            logs.append(f"Completion contains '{len(tool_calls)}' tool-calls, but the maximum number of tool-calls allowed per completion is '{self._settings['max_tool_calls']}'.")
            for i, message in enumerate(correction_messages):
                if 'tool_call_id' in message:
                    correction_messages[i]['content'] = f"Your response must contain at most {self._settings['max_tool_calls']} tool-call{'' if self._settings['max_tool_calls'] == 1 else 's'}, but yours contains {len(tool_calls)} tool-calls. Please filter accordingly and try again!"

        # error case: custom tool-choice "always" was violated
        if response_type == "always" and len(tool_calls) == 0:
            is_valid = False
            logs.append("Completion does not contain a tool-call despite tool-choice being set to value 'always'.")
            for i, message in enumerate(correction_messages):
                if 'tool_call_id' not in message:
                    correction_messages[i]['content'] = "Please express your last message in a tool-call instead of a text response!"

        # error case: function call violates JSON Schema
        for call in tool_calls:
            for i, message in enumerate(correction_messages):
                if 'tool_call_id' in message:
                    if call['id'] == message['tool_call_id']:
                        if message['content'] == tool_call_is_valid_default_correction:
                            valid, reason, logs = self.validate_tool_call(call, logs)
                            if not valid:
                                is_valid = False
                                correction_messages[i]['content'] = reason
                        else:
                            self._logger.debug(f"Skipping JSON Schema based validity check of tool-call '{call['name']}' as it is already considered invalid by some previous filter.")

        # error case: text response cannot be parsed as JSON despite JSON-mode being activated
        if response_type == "json":
            try:
                json.loads(text)
            except Exception as e:
                is_valid = False
                logs.append(f"Expected text-completion to be JSON-compliant: {repr(e)}")
                correction_messages[-1]['content'] = "Your response cannot be parsed as JSON. Please try again and respond only with valid JSON and no additional text."
            else:
                self._logger.debug("Text-completion parses as JSON.")

        return is_valid, correction_messages, logs

    def validate_tool_call(self, tool_call, logs):
        success = True
        reason = None

        call_name = tool_call.get('name')
        call_args = tool_call.get('arguments')

        if call_name is None or call_args is None:
            success = False
            logs.append("Completion contains a tool-call that that misses the keys 'name' and/or 'arguments'.")
            reason = "Your response contains an invalid tool-call. A tool-call must contain the keys 'name' and 'arguments'."
        else:
            matched = next(
                (tool for tool in self.tools if tool.get('type') == "function" and tool.get("function", {}).get('name') == call_name),
                None
            )

            if matched is None:
                success = False
                logs.append("Completion contains a tool-call that cannot be associated with any defined tool.")
                reason = "Your response contains a tool-call that cannot be associated with any of the defined tools."
            else:
                schema = matched["function"].get("parameters", {})
                try:
                    arguments = json.loads(call_args)
                except json.JSONDecodeError as e:
                    success = False
                    logs.append(f"Completion contains a tool-call with key 'arguments' that cannot be parsed as JSON: {e.msg}")
                    reason = f"Your response contains a tool-call of which the arguments cannot be parsed as JSON: {e.msg}"
                else:
                    if JSONSCHEMA_AVAILABLE:
                        validator = jsonschema.Draft7Validator(schema)
                        errors = sorted(validator.iter_errors(arguments), key=lambda e: e.path)
                        if errors:
                            success = False
                            logs.append(f"Completion contains a tool-call that violates the JSON Schema: {errors[0].message}")
                            reason = f"Your response contains a tool-call that violates the JSON Schema: {errors[0].message}"
                    else:
                        logs.append("Tool-call cannot be validated against tool definitions because the 'jsonschema' module is not available.")
                        self._logger.warn(logs[-1], once=True)

        if success:
            self._logger.debug("Tool-call is valid.")

        return success, reason, logs

    def post_process_completion(self, response_type, reasoning, text, tool_calls, is_valid, correction, usage, logs):
        assert self.is_prompting
        self.is_prompting = False

        if not is_valid:
            error_log_i = len(logs) - 1
            self._logger.debug("Restoring original context.")
            self.messages = self.context_dump
        elif correction:
            self._logger.debug("Completion is valid after automatic correction.")
            # current context = original context + (new message) + (invalid completion + correction) * n + valid completion
            num_messages_after_correction = len(self.messages)
            num_removed = len(self.messages) - len(self.new_messages) - 1 if self.reset_context else len(self.messages) - len(self.context_dump) - len(self.new_messages) - 1
            self._logger.info(f"Removing '{num_removed}' correction-related message{'' if num_removed == 1 else 's'} from context.")
            self.messages = self.new_messages + [self.messages[-1]] if self.reset_context else self.context_dump + self.new_messages + [self.messages[-1]]
            assert len(self.messages) == num_messages_after_correction - num_removed, \
                f"Expected context to contain '{num_messages_after_correction - num_removed}' messages after removing corrections but it contains '{len(self.messages)}'."
        else:
            num_expected = len(self.new_messages) + 1 if self.reset_context else len(self.context_dump) + len(self.new_messages) + 1
            assert len(self.messages) == num_expected, \
                f"Expected context to contain '{num_expected}' messages after completion without corrections but it contains '{len(self.messages)}'."

        success = is_valid

        completion = {}

        if usage is not None:
            assert 'duration' in usage, str(usage)
            completion = {'usage': usage}
        if reasoning is not None:
            completion['reasoning'] = reasoning
        if len(tool_calls) > 0:
            completion['tools'] = []
        for call in tool_calls:
            if call['arguments'] == "": # fix empty arguments (e.g. Claude does that)
                logs.append(f"Fixing empty arguments of tool-call '{call['name']}' to empty dictionary.")
                self._logger.debug(logs[-1])
                call['arguments'] = r"{}"
            call['arguments'] = json.loads(call['arguments'])
            completion['tools'].append(call)
        if text is not None:
            if response_type == "json":
                text = json.loads(text)
            completion['text'] = text

        if is_valid:
            if sum([completion.get('reasoning', "") != "", completion.get('text', "") != "", len(completion.get('tools', [])) > 0]) > 1:
                type_str = "mixed"
            elif 'reasoning' in completion:
                type_str = "reasoning"
            elif 'text' in completion:
                type_str = "text"
            elif 'tools' in completion:
                type_str = "tool"
            else:
                raise RuntimeError

            if correction:
                correction_str = " after automatic correction"
            else:
                correction_str = ""

            if 'tokens_output' in usage:
                logs.append(f"Generated {type_str}-completion with '{usage['tokens_output']}' token{'' if usage['tokens_output'] == 1 else 's'} in '{usage['duration']:.3f}s'{correction_str}.")
            else:
                logs.append(f"Generated {type_str}-completion in '{usage['duration']:.3f}s'{correction_str}.")

            completion['logs'] = logs
            message = logs[-1]

            for parser in self._settings['parser']:
                success, message, completion, allow_retry = self.execute_parser(
                    path_or_name=parser,
                    success=success,
                    message=message,
                    completion=completion
                )
                if not success and not allow_retry:
                    self._logger.debug("Restoring original context.")
                    self.messages = self.context_dump
                    raise UnrecoverableError(message)

            if not success:
                self._logger.debug("Restoring original context.")
                self.messages = self.context_dump
        else:
            completion['logs'] = logs
            message = logs[error_log_i]

        return success, message, completion

    def execute_parser(self, path_or_name, success, message, completion):
        # resolve path
        custom_path = os.path.abspath(os.path.expanduser(path_or_name))
        if os.path.isfile(custom_path):
            self._logger.debug(f"Executing custom completion parser '{custom_path}'.")
            file_path = custom_path
        elif os.path.isfile(f"{custom_path}.py"):
            custom_path = f"{custom_path}.py"
            self._logger.debug(f"Executing custom completion parser '{custom_path}'.")
        else:
            default_path = os.path.join(os.path.dirname(os.path.dirname(os.path.realpath(__file__))), "parser", path_or_name)
            if os.path.isfile(default_path):
                self._logger.debug(f"Executing default completion parser '{default_path}'.")
                file_path = default_path
            elif os.path.isfile(f"{default_path}.py"):
                default_path = f"{default_path}.py"
                self._logger.debug(f"Executing default completion parser '{default_path}'.")
                file_path = default_path
            else:
                raise UnrecoverableError(f"Parser '{custom_path}' does not exist.")

        parser_executed = False
        allow_retry = True

        # import
        try:
            spec = importlib.util.spec_from_file_location(f"_dynamic_{uuid.uuid4().hex}", file_path)
            if spec is None or spec.loader is None:
                raise CustomException("Failed to load module.")
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            if not hasattr(module, "parse"):
                raise CustomException("Function 'parse' is not defined.")
        except CustomException as e:
            import_error = str(e)
            self._logger.debug(f"Failed to import completion parser '{file_path}': {import_error}")
        except Exception as e:
            import_error = repr(e)
            self._logger.debug(f"Failed to import completion parser '{file_path}': {import_error}")
        else:
            # execute
            try:
                response = module.parse(self, success, message, completion)
            except UnrecoverableError as e:
                message = f"Failed to execute completion parser '{file_path}': {e}"
                allow_retry = False
            except Exception as e:
                message = f"Failed to execute completion parser '{file_path}': {repr(e)}"
            else:
                parser_executed = True

        # validate response
        if parser_executed:
            self._logger.debug(f"Executed completion parser '{file_path}'.")
            try:
                assert_type_value(obj=response, type_or_value=tuple, name="completion parser response")
                assert_log(expression=len(response) == 3, message=f"Expected completion parser response to be tuple of length '3' but got tuple of length '{len(response)}'.")
                success, message, completion = response
                assert_type_value(obj=success, type_or_value=bool, name="first element in completion parser response")
                assert_type_value(obj=message, type_or_value=str, name="second element in completion parser response")
                assert_type_value(obj=completion, type_or_value=dict, name="third element in completion parser response")
            except UnrecoverableError as e:
                success = False
                message = f"Failed to execute completion parser '{file_path}': {e}"
                allow_retry = False
        else:
            success = False

        return success, message, completion, allow_retry

    # Callbacks

    def prompt(self, text, reset_context, response_type):
        # parse arguments
        assert_type_value(obj=text, type_or_value=[str, dict, list, None], name="argument 'text'")
        assert_type_value(obj=reset_context, type_or_value=bool, name="argument 'reset_context'")
        response_choices = ["none", "text", "json", "auto"] + (["always"] if len(self.tools) > 0 else []) + [f['function']['name'] for f in self.tools]
        assert_type_value(obj=response_type, type_or_value=response_choices, name="argument 'response_type'")
        assert_log(expression=response_type != "none" or reset_context or text is not None, message="A prompt must either alter the context or trigger a chat completion.")

        # save context for resetting
        if response_type != "none":
            self.context_dump = copy.deepcopy(self.messages)
            self.reset_context = reset_context

        # set context
        if reset_context or text is not None:
            if text is None:
                self.new_messages = []
            elif isinstance(text, dict):
                self.new_messages = [text]
            elif isinstance(text, list):
                self.new_messages = []
                for item in text:
                    if isinstance(item, dict):
                        self.new_messages.append(item)
                    else:
                        awaited_tools = self.get_awaited_tools()[2]
                        if len(awaited_tools) > 0:
                            self.new_messages.append({'role': "tool", 'tool_call_id': awaited_tools[0], 'content': text})
                        else:
                            self.new_messages.append({'role': "user", 'content': [{'type': "text", 'text': text}]})
            elif reset_context:
                self.new_messages = [{'role': "user", 'content': [{'type': "text", 'text': text}]}]
            else:
                awaited_tools = self.get_awaited_tools()[2]
                if len(awaited_tools) > 0:
                    self.new_messages = [{'role': "tool", 'tool_call_id': awaited_tools[0], 'content': text}]
                else:
                    self.new_messages = [{'role': "user", 'content': [{'type': "text", 'text': text}]}]
            success, message = self.set_context(mode="reset" if reset_context else "insert", messages=self.new_messages, index=0, reverse_indexing=True)
            if success:
                if response_type == "none":
                    return True, message, None
            else:
                raise UnrecoverableError(message)
        else:
            self.new_messages = []

        # prevent completion while a tool response is waited (OpenAI allows this actually)
        awaited_tools = self.get_awaited_tools()[2]
        num_awaited = len(awaited_tools)
        if num_awaited > 0:
            if reset_context or text is not None:
                self._logger.debug("Restoring original context.")
                self.messages = self.context_dump
            message = f"Cannot trigger completion while '{num_awaited}' tool response{' is' if num_awaited == 1 else 's are'} awaited: {awaited_tools}"
            raise UnrecoverableError(message)

        # prevent completion when context is empty
        if len(self.messages) == 0:
            if reset_context or text is not None:
                self._logger.debug("Restoring original context.")
                self.messages = self.context_dump
            message = "Cannot trigger completion when context is empty."
            raise UnrecoverableError(message)

        success, message = self.set_context(mode="reset", messages=self.messages, index=0, reverse_indexing=True)
        if not success:
            if reset_context or text is not None:
                self._logger.debug("Restoring original context.")
                self.messages = self.context_dump
            # raise UnrecoverableError(message) # failure cause can be syntax error, which is not resolvable via retry
            return False, message, None # failure cause can be failed download, which is resolvable via retry

        self.log_prompt(mode="reset" if reset_context else "add", response_type=response_type)

        response = self.generate_completion(response_type)

        return response

    def interrupt(self):
        if not self.is_prompting:
            message = "There is no completion in progress that could be interrupted."
            self._logger.info(message)
            return True, message

        def send_external():
            # keep trying until is_prompting is cleared
            while self.is_prompting:
                from multiprocessing.connection import wait
                ready = wait([self.pipe[0]], timeout=0.01)
                if ready:
                    try:
                        self.pipe[0].send("EXTERNAL")
                    except (BrokenPipeError, OSError):
                        break  # receiver gone
                self._logger.debug("Waiting until completion is interrupted.", throttle=1.0, skip_first=True)
                time.sleep(0.005)

        self._logger.info("Interrupting completion.")

        tic = time.perf_counter()
        thread = threading.Thread(target=send_external, daemon=True)
        thread.start()
        thread.join()  # wait until is_prompting becomes False
        toc = time.perf_counter()

        message = f"Interrupted completion after '{toc - tic:.3f}s'."
        self._logger.info(message)

        return True, message

    def get_context(self):
        return True, "Retrieved context.", copy.deepcopy(self.messages)

    def set_context(self, mode, messages, index, reverse_indexing):
        # parse arguments
        assert_type_value(obj=mode, type_or_value=["reset", "insert", "replace", "remove"], name="argument 'mode'")
        assert_type_value(obj=messages, type_or_value=[list, dict, None], name="argument 'messages'")
        if messages is None:
            messages = []
        elif isinstance(messages, dict):
            messages = [messages]
        else:
            for i, item in enumerate(messages):
                assert_type_value(obj=item, type_or_value=dict, name=f"element '{i}' in argument 'messages'")
        assert_type_value(obj=index, type_or_value=int, name="argument 'index'")
        assert_type_value(obj=reverse_indexing, type_or_value=bool, name="argument 'reverse_indexing'")

        if mode == "reset":
            if len(messages) > 0:
                messages_before = copy.deepcopy(self.messages)
            if len(self.messages) == 0:
                message = "Kept empty context."
            else:
                self.messages = []
                message = "Cleared context."
            if len(messages) == 0:
                success = True
            else:
                for i, msg in enumerate(messages):
                    try:
                        self.check_message_validity(msg)
                        msg = self.encode_files(msg)
                    except CustomException as e:
                        success = False
                        message = f"Failed to construct context at message {format_obj(msg)} (index '{i}'): {e}"
                        self.messages = messages_before
                        return success, message
                    self.messages.append(msg)
                success = True
                message = f"Set new context with '{len(self.messages)}' message{'' if len(self.messages) == 1 else 's'}."

        elif mode == "insert":
            if len(messages) == 0:
                success = False
                message = "Cannot insert messages into context without providing one."
                return success, message
            if reverse_indexing:
                i = len(self.messages) - index
            else:
                i = index

            messages_before = copy.deepcopy(self.messages)

            self.messages = []
            for j, msg in enumerate(messages_before[:i] + messages + messages_before[i:]):
                try:
                    self.check_message_validity(msg)
                    msg = self.encode_files(msg)
                except CustomException as e:
                    success = False
                    message = f"Failed to construct context at message {format_obj(msg)} (index '{j}'): {e}"
                    self.messages = messages_before
                    return success, message
                self.messages.append(msg)
            success = True
            message = f"Inserted '{len(messages)}' message{'' if len(messages) == 1 else 's'} into context."

        elif mode == "replace":
            if len(messages) == 0:
                success = False
                message = "Cannot replace message in context without providing one."
                return success, message
            if reverse_indexing:
                i = len(self.messages) - index - 1
            else:
                i = index
            if i < 0 or i > len(self.messages) - 1:
                success = False
                message = f"Cannot replace message at index '{i}' in context containing '{len(self.messages)}' message{'' if len(self.messages) == 1 else 's'}."
                return success, message

            messages_before = copy.deepcopy(self.messages)

            self.messages = []
            for i, msg in enumerate(messages_before[:i] + messages + messages_before[i + len(messages):]):
                try:
                    self.check_message_validity(msg)
                    msg = self.encode_files(msg)
                except CustomException as e:
                    success = False
                    message = f"Failed to construct context at message {format_obj(msg)} (index '{i}'): {e}"
                    self.messages = messages_before
                    return success, message
                self.messages.append(msg)
            success = True
            added = len(self.messages) - len(messages_before)
            if added == 0:
                message = f"Replaced '{len(messages)}' message{'' if len(messages) == 1 else 's'} in context."
            else:
                message = f"Replaced '{len(messages) - added}' and added '{added}' message{'' if added == 1 else 's'} to context."

        elif mode == "remove":
            if reverse_indexing:
                i = len(self.messages) - index - 1
            else:
                i = index
            if i < 0 or i > len(self.messages) - 1:
                success = False
                message = f"Cannot remove message '{i}' from context containing '{len(self.messages)}' message{'' if len(self.messages) == 1 else 's'}."
                return success, message

            messages_before = copy.deepcopy(self.messages)

            self.messages = []
            for j, msg in enumerate(copy.deepcopy(messages_before)):
                if i != j:
                    try:
                        self.check_message_validity(msg)
                        msg = self.encode_files(msg)
                    except CustomException as e:
                        success = False
                        message = f"Failed to construct context at message {format_obj(msg)} (index '{self.messages[j]}'): {e}"
                        self.messages = messages_before
                        return success, message
                    self.messages.append(msg)
            success = True
            message = f"Removed message '{i}' from context."

        return success, message

    def get_tools(self):
        if len(self.tools) == 0:
            return True, "There are no tools defined.", []

        tools = copy.deepcopy(self.tools)
        return True, f"Retrieved '{len(tools)}' tool{'' if len(tools) == 1 else 's'}.", tools

    def set_tools(self, tools):
        # parse arguments
        if tools is None:
            tools = []
        else:
            assert_type_value(obj=tools, type_or_value=[dict, list], name="argument 'tools'")
            if isinstance(tools, dict):
                tools = [tools]
            elif len(tools) > 0:
                for i, item in enumerate(tools):
                    assert_type_value(obj=item, type_or_value=dict, name=f"element '{i}' in argument 'tools'")

        success = True

        if len(tools) == 0:
            if len(self.tools) == 0:
                message = "Kept zero tool definitions."
            else:
                self.tools = []
                message = "Undefined all tools."
        elif tools == self.tools:
            message = f"Kept already existing tool definition{'' if len(tools) == 1 else 's'}."
        else:
            used_names = []

            for i, tool in enumerate(tools):
                if set(tool.keys()) != {'type', 'function'}:
                    success = False
                    message = f"Tool '{i}' does not satisfy the required format: The top-level keys must be ['type', 'function'] but got {list(tool.keys())}."
                    break

                if tool['type'] != "function":
                    success = False
                    message = f"Tool '{i}' does not satisfy the required format: Expected the value of key 'type' to be 'function' but got '{tool['type']}'."
                    break

                if not isinstance(tool['function'], dict):
                    success = False
                    message = f"Tool '{i}' does not satisfy the required format: Expected the value of key 'function' to be of type 'dict' but got '{type(tool['function']).__name__}'."
                    break

                keys_required = {'name', 'description', 'parameters'} # OpenAI allows omitting 'parameters', Mistral does not, and OpenRouter does with some models.
                keys_optional = {'strict'}

                if not (set(tool['function'].keys()).issubset(keys_required | keys_optional) and keys_required.issubset(tool['function'])):
                    success = False
                    message = f"Tool '{i}' does not satisfy the required format: The top-level keys must be {list(keys_required)} and optionally {list(keys_optional)} but got {list(tool['function'].keys())}."
                    break

                if not isinstance(tool['function']['name'], str):
                    success = False
                    message = f"Tool '{tool['function']['name']}' does not satisfy the required format: The value of key 'name' must be of type 'str' but got '{type(tool['function']['name']).__name__}'."
                    break

                if tool['function']['name'] in used_names:
                    success = False
                    message = f"All tools must feature a unique name - The name '{tool['function']['name']}' is featured more than once."
                    break

                used_names.append(tool['function']['name'])

                if not isinstance(tool['function']['description'], str):
                    success = False
                    message = f"Tool '{tool['function']['name']}' does not satisfy the required format: The value of key 'description' must be of type 'str' but got '{type(tool['function']['description']).__name__}'."
                    break

                if 'strict' in tool['function']:
                    if not isinstance(tool['function']['strict'], bool):
                        success = False
                        message = f"Tool '{tool['function']['name']}' does not satisfy the required format: The value of key 'strict' must be of type 'bool' but got '{type(tool['function']['strict']).__name__}'."
                        break

                if 'parameters' in tool['function']:
                    if not isinstance(tool['function']['parameters'], dict):
                        success = False
                        message = f"Tool '{tool['function']['name']}' does not satisfy the required format: The value of key 'parameters' must be of type 'dict' but got '{type(tool['function']['parameters']).__name__}'."
                        break

                    keys_required = {'type', 'properties'}
                    keys_optional = {'required', 'additionalProperties'}
                    if not set(tool['function']['parameters'].keys()).issubset(keys_required | keys_optional) and keys_required.issubset(tool['function']['parameters']):
                        success = False
                        message = f"Tool '{tool['function']['name']}' does not satisfy the required format: The value of key 'parameters' must contain the keys {list(keys_required)} and optionally {list(keys_optional)}."
                        break

                    if tool['function']['parameters']['type'] != "object":
                        success = False
                        message = f"Tool '{tool['function']['name']}' does not satisfy the required format: The value of key 'type' in 'parameters' must be 'object' but got '{tool['function']['parameters']['type']}'."
                        break

                    success, message = self.validate_function_properties(schema=tool['function']['parameters'], function_name=tool['function']['name'], path="parameters", strict=tool['function'].get('strict', False))
                    if not success:
                        break
            else:
                if len(self.tools) == 0:
                    message = f"Set tool definition{'' if len(tools) == 1 else 's'}."
                    tool_msg = message.rstrip(".") + ":\n" + '\n'.join([f"{i}: {json.dumps(tool, indent=2)}" for i, tool in enumerate(tools)])
                else:
                    message = "Updated tool definitions."
                    updates = 0
                    lines = []
                    for i, tool in enumerate(tools):
                        exists = tool in self.tools
                        if exists:
                            updates += 1
                        lines.append(f"{i}{'*' if exists else ''}: {json.dumps(tool, indent=2)}")
                    tool_msg = message.rstrip(".") + ":\n" + '\n'.join(lines)
                    if updates > 0:
                        tool_msg += f"\n*tool existed before ({updates} of {len(tools)})"

                self._logger.info(tool_msg)
                self.tools = copy.deepcopy(tools)

        return success, message

    def get_awaited_tools(self):
        all_ids = []
        awaited_tool_responses = []
        for i, message in enumerate(self.messages):
            if message['role'] == 'tool':
                assert 'tool_call_id' in message, f"{message}"
                if message['tool_call_id'] in awaited_tool_responses:
                    awaited_tool_responses.remove(message['tool_call_id'])
                else:
                    self._logger.warn(f"Message '{i}' contains a tool response without a corresponding tool-call: '{message}'")
            elif message['role'] == 'assistant':
                if 'tool_calls' in message:
                    for call in message['tool_calls']:
                        assert 'id' in call, f"{call}"
                        all_ids.append(call['id'])
                        awaited_tool_responses.append(call['id'])

        if len(awaited_tool_responses) == 0:
            message = "Awaiting '0' tool responses."
        else:
            message = f"Awaiting '{len(awaited_tool_responses)}' tool response{'' if len(awaited_tool_responses) == 1 else 's'}: {awaited_tool_responses}"

        return True, message, copy.deepcopy(awaited_tool_responses)
