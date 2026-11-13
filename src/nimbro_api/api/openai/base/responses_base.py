import os
import copy
import json
import math
import time
import base64
import codecs
import mimetypes
import threading
from collections import OrderedDict
from urllib.parse import urlsplit, urlunsplit

import httpx

from nimbro_api.client import ClientBase
from nimbro_api.utility.io import parse_image_b64, read_as_b64, encode_b64, download_file
from nimbro_api.utility.api import get_api_key, validate_endpoint, http_request, HttpRequestCancelled, _HttpRequestCancelEvent
from nimbro_api.utility.misc import UnrecoverableError, assert_type_value, assert_keys, assert_log, escape, format_obj
from nimbro_api.utility.string import is_url
from ..utility import validate_connection, get_models

class ResponsesBase(ClientBase):

    def __init__(self, settings, default_settings, **kwargs):
        super().__init__(settings=settings, default_settings=default_settings, **kwargs)
        self.get_api_key = get_api_key.__get__(self)
        self.validate_connection = validate_connection.__get__(self)
        self.get_models = get_models.__get__(self)

        self._messages = []
        self._tools = []
        self._cancel = _HttpRequestCancelEvent()
        self._active = threading.Event()
        self._pending_context = None
        self._pending_tools = None
        self._openai_reasoning = set()
        self._pending_openai_reasoning = set()

        self._logger.debug(f"Initialized '{type(self).__name__}' object.")
        self._initialized = True

    def set_settings(self, settings, mode="set"):
        settings = self._introduce_settings(settings=settings, mode=mode)

        # logger_info_prompt
        assert_type_value(obj=settings['logger_info_prompt'], type_or_value=[bool, int], name="setting 'logger_info_prompt'")

        # logger_info_prompt_threshold
        assert_type_value(obj=settings['logger_info_prompt_threshold'], type_or_value=int, name="setting 'logger_info_prompt_threshold'")
        assert_log(expression=settings['logger_info_prompt_threshold'] > 0, message=f"Expected setting 'logger_info_prompt_threshold' to be greater than zero but got '{settings['logger_info_prompt_threshold']}'.")

        # logger_info_prompt_cutoff
        assert_type_value(obj=settings['logger_info_prompt_cutoff'], type_or_value=int, name="setting 'logger_info_prompt_cutoff'")
        assert_log(expression=settings['logger_info_prompt_cutoff'] >= 0, message=f"Expected setting 'logger_info_prompt_cutoff' to be non-negative but got '{settings['logger_info_prompt_cutoff']}'.")

        # logger_info_response
        assert_type_value(obj=settings['logger_info_response'], type_or_value=bool, name="setting 'logger_info_response'")

        # logger_debug_chunks
        assert_type_value(obj=settings['logger_debug_chunks'], type_or_value=bool, name="setting 'logger_debug_chunks'")

        valid_flavors = ["openai", "openrouter", "vllm", "openresponses"]

        # endpoints
        assert_type_value(obj=settings['endpoints'], type_or_value=dict, name="setting 'endpoints'")
        assert_log(expression=len(settings['endpoints']) > 0, message=f"Expected at least one endpoint in setting 'endpoints' but got '{settings['endpoints']}'.")
        for endpoint in settings['endpoints']:
            assert_type_value(obj=endpoint, type_or_value=str, name="endpoint name in setting 'endpoints'")
            assert_log(expression=len(endpoint.strip()) > 0, message=f"Expected a nonempty, non-whitespace endpoint name in setting 'endpoints' but got '{endpoint}'.")
            validate_endpoint(endpoint=settings['endpoints'][endpoint], flavors=valid_flavors, require_key=True, require_name=False, allow_models=True, setting_name=f"endpoint '{endpoint}' in setting 'endpoints'")

        # endpoint
        if isinstance(settings['endpoint'], dict):
            validate_endpoint(endpoint=settings['endpoint'], flavors=valid_flavors, require_key=True, require_name=True, allow_models=True, setting_name="endpoint provided through setting 'endpoint'")
            settings['endpoints'][settings['endpoint']['name']] = settings['endpoint']
            settings['endpoint'] = settings['endpoint']['name']
            del settings['endpoints'][settings['endpoint']]['name']
        else:
            assert_type_value(obj=settings['endpoint'], type_or_value=list(settings['endpoints'].keys()), name="setting 'endpoint'")
        flavor = settings['endpoints'][settings['endpoint']]['api_flavor']

        # model
        assert_type_value(obj=settings['model'], type_or_value=str, name="setting 'model'")
        assert_log(expression=len(settings['model'].strip()) > 0, message=f"Expected a non-whitespace character in setting 'model' but got '{settings['model']}'.")

        # validate_model
        assert_type_value(obj=settings['validate_model'], type_or_value=[float, int, bool], name="setting 'validate_model'")
        if not isinstance(settings['validate_model'], bool):
            assert_log(expression=math.isfinite(settings['validate_model']) and settings['validate_model'] >= 0, message=f"Expected a non-negative, finite setting 'validate_model' but got '{settings['validate_model']}'.")

        # service_tier
        assert_type_value(obj=settings['service_tier'], type_or_value=[None, "price", "latency", "throughput"], name="setting 'service_tier'")

        # instructions
        assert_type_value(obj=settings['instructions'], type_or_value=[str, None], name="setting 'instructions'")

        # parallel_tool_calls
        assert_type_value(obj=settings['parallel_tool_calls'], type_or_value=bool, name="setting 'parallel_tool_calls'")

        # reasoning_effort
        assert_type_value(obj=settings['reasoning_effort'], type_or_value=["", "none", "minimal", "low", "medium", "high", "xhigh", "max"], name="setting 'reasoning_effort'")

        # reasoning_summary
        assert_type_value(obj=settings['reasoning_summary'], type_or_value=[None, "auto", "concise", "detailed"], name="setting 'reasoning_summary'")

        # send_reasoning
        assert_type_value(obj=settings['send_reasoning'], type_or_value=bool, name="setting 'send_reasoning'")

        # verbosity
        assert_type_value(obj=settings['verbosity'], type_or_value=[None, "low", "medium", "high"], name="setting 'verbosity'")

        # max_tokens
        assert_type_value(obj=settings['max_tokens'], type_or_value=int, name="setting 'max_tokens'")
        assert_log(expression=settings['max_tokens'] > 0, message=f"Expected a positive setting 'max_tokens' but got '{settings['max_tokens']}'.")

        # temperature
        assert_type_value(obj=settings['temperature'], type_or_value=[float, int, None], name="setting 'temperature'")
        if settings['temperature'] is not None:
            assert_log(
                expression=0.0 <= settings['temperature'] <= 2.0,
                message=f"Expected setting 'temperature' to be between '0.0' and '2.0' (inclusive) but got '{settings['temperature']}'."
            )
            settings['temperature'] = float(settings['temperature'])

        # top_p
        assert_type_value(obj=settings['top_p'], type_or_value=[float, int, None], name="setting 'top_p'")
        if settings['top_p'] is not None:
            assert_log(
                expression=0.0 <= settings['top_p'] <= 1.0,
                message=f"Expected setting 'top_p' to be between '0.0' and '1.0' (inclusive) but got '{settings['top_p']}'."
            )
            settings['top_p'] = float(settings['top_p'])

        # presence_penalty
        assert_type_value(obj=settings['presence_penalty'], type_or_value=[float, int, None], name="setting 'presence_penalty'")
        if settings['presence_penalty'] is not None:
            assert_log(
                expression=-2.0 <= settings['presence_penalty'] <= 2.0,
                message=f"Expected setting 'presence_penalty' to be between '-2.0' and '+2.0' (inclusive) but got '{settings['presence_penalty']}'."
            )
            settings['presence_penalty'] = float(settings['presence_penalty'])

        # frequency_penalty
        assert_type_value(obj=settings['frequency_penalty'], type_or_value=[float, int, None], name="setting 'frequency_penalty'")
        if settings['frequency_penalty'] is not None:
            assert_log(
                expression=-2.0 <= settings['frequency_penalty'] <= 2.0,
                message=f"Expected setting 'frequency_penalty' to be between '-2.0' and '+2.0' (inclusive) but got '{settings['frequency_penalty']}'."
            )
            settings['frequency_penalty'] = float(settings['frequency_penalty'])

        # top_k
        assert_type_value(obj=settings['top_k'], type_or_value=[int, None], name="setting 'top_k'")
        if settings['top_k'] is not None:
            assert_log(expression=settings['top_k'] >= 0, message=f"Expected non-negative setting 'top_k' but got '{settings['top_k']}'.")

        # stop
        assert_type_value(obj=settings['stop'], type_or_value=[str, list, None], name="setting 'stop'")
        if settings['stop'] is not None:
            for stop in [settings['stop']] if isinstance(settings['stop'], str) else settings['stop']:
                assert_type_value(obj=stop, type_or_value=str, name="sequence in setting 'stop'")
                assert_log(expression=bool(stop), message=f"Expected a nonempty sequence in setting 'stop' but got '{stop}'.")
            assert_log(expression=flavor in ["openrouter", "vllm"], message=f"Expected the OpenRouter or vLLM flavor for setting 'stop' but got '{flavor}'.")
            if isinstance(settings['stop'], list):
                assert_log(expression=len(settings['stop']) <= 4, message=f"Expected at most four sequences in setting 'stop' but got '{len(settings['stop'])}'.")

        # truncation
        assert_type_value(obj=settings['truncation'], type_or_value=["auto", "disabled"], name="setting 'truncation'")

        # include
        assert_type_value(obj=settings['include'], type_or_value=[list, None], name="setting 'include'")
        if settings['include'] is not None:
            valid_values = [
                "reasoning.encrypted_content",
                "file_search_call.results",
                "code_interpreter_call.outputs",
                "web_search_call.results",
                "web_search_call.action.sources",
                "output.encrypted_content"
            ]
            for index, field in enumerate(settings['include']):
                assert_type_value(obj=field, type_or_value=valid_values, name=f"setting 'include[{index}]'")

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

        # timeout_connect
        assert_type_value(obj=settings['timeout_connect'], type_or_value=[float, int, None], name="setting 'timeout_connect'")
        if settings['timeout_connect'] is not None:
            assert_log(expression=math.isfinite(settings['timeout_connect']) and settings['timeout_connect'] > 0, message=f"Expected positive, finite setting 'timeout_connect' but got '{settings['timeout_connect']}'.")

        # timeout_read
        assert_type_value(obj=settings['timeout_read'], type_or_value=[float, int, None], name="setting 'timeout_read'")
        if settings['timeout_read'] is not None:
            assert_log(expression=math.isfinite(settings['timeout_read']) and settings['timeout_read'] > 0, message=f"Expected positive, finite setting 'timeout_read' but got '{settings['timeout_read']}'.")

        # timeout_chunk_first
        assert_type_value(obj=settings['timeout_chunk_first'], type_or_value=[float, int, None], name="setting 'timeout_chunk_first'")
        if settings['timeout_chunk_first'] is not None:
            assert_log(
                expression=math.isfinite(settings['timeout_chunk_first']) and settings['timeout_chunk_first'] > 0.0,
                message=f"Expected setting 'timeout_chunk_first' to be None or positive and finite but got '{settings['timeout_chunk_first']}'."
            )

        # timeout_chunk_next
        assert_type_value(obj=settings['timeout_chunk_next'], type_or_value=[float, int, None], name="setting 'timeout_chunk_next'")
        if settings['timeout_chunk_next'] is not None:
            assert_log(
                expression=math.isfinite(settings['timeout_chunk_next']) and settings['timeout_chunk_next'] > 0.0,
                message=f"Expected setting 'timeout_chunk_next' to be None or positive and finite but got '{settings['timeout_chunk_next']}'."
            )

        # timeout_response
        assert_type_value(obj=settings['timeout_response'], type_or_value=[float, int, None], name="setting 'timeout_response'")
        if settings['timeout_response'] is not None:
            assert_log(expression=math.isfinite(settings['timeout_response']) and settings['timeout_response'] > 0, message=f"Expected positive, finite setting 'timeout_response' but got '{settings['timeout_response']}'.")

        # apply settings
        self._endpoint = settings['endpoints'][settings['endpoint']]
        return self._apply_settings(settings, mode)

    def _generate(self, input_items, request_options):
        stamp = time.perf_counter()

        # retrieve API key
        success, message, api_key = self.get_api_key()
        if not success:
            raise UnrecoverableError(message)

        # validate connection
        success, message = self.validate_connection(api_key=api_key)
        if not success:
            return False, message, None
        if self._cancel.is_set():
            return False, "Responses generation interrupted before sending the request.", None

        # construct payload
        data = {
            'model': self._settings['model'],
            'instructions': self._settings['instructions'],
            'tools': self._tools,
            'parallel_tool_calls': self._settings['parallel_tool_calls'],
            'max_output_tokens': self._settings['max_tokens'],
            'temperature': self._settings['temperature'],
            'top_p': self._settings['top_p'],
            'presence_penalty': self._settings['presence_penalty'] if self._endpoint['api_flavor'] != "openai" else None,
            'frequency_penalty': self._settings['frequency_penalty'] if self._endpoint['api_flavor'] != "openai" else None,
            'top_k': self._settings['top_k'] if self._endpoint['api_flavor'] in ["openrouter", "vllm"] else None,
            'stop': self._settings['stop'],
            'truncation': self._settings['truncation'],
            'include': self._get_include(),
            'stream': self._settings['stream'],
            'store': False
        }
        data = copy.deepcopy({key: value for key, value in data.items() if value is not None})
        data['input'] = self._prepare_reasoning(input_items)
        flavor = self._endpoint['api_flavor']
        if self._settings['stream'] and flavor in ["openai", "openresponses"]:
            data['stream_options'] = {'include_obfuscation': False}
        reasoning = {}
        if self._settings['reasoning_effort'] != "":
            reasoning['effort'] = self._settings['reasoning_effort']
        if self._settings['reasoning_summary'] is not None and flavor != "vllm":
            reasoning['summary'] = self._settings['reasoning_summary']
        if reasoning:
            data['reasoning'] = reasoning
        tier = self._settings['service_tier']
        if flavor == "openrouter":
            if tier is not None:
                data['provider'] = {'sort': tier}
                if tier != "price":
                    data['service_tier'] = "ultrafast"
        elif flavor != "vllm" and tier is not None:
            data['service_tier'] = "flex" if tier == "price" else "priority"
        data.update(request_options)

        # use API
        success, message, response = self._request("POST", self._endpoint['api_url'], data, api_key, self._settings['stream'])
        if not success:
            return False, message, None

        # parse response
        try:
            assert_type_value(obj=response, type_or_value=dict, name="Responses API response")
            if response.get('error') is not None:
                return False, f"Responses API returned an error: {format_obj(response['error'])}", None
            self._validate_response(response)
            if response['status'] not in ["completed", "incomplete"]:
                return False, f"Responses API returned status '{response['status']}' for response '{response['id']}'.", None
        except (ValueError, UnrecoverableError) as e:
            return False, f"Failed to parse Responses API response: {e}", None

        # finalize response
        message = f"Created response '{response['id']}' with status '{response['status']}' in '{time.perf_counter() - stamp:.3f}s'."
        return True, message, response

    def _validate_response(self, response):
        assert_type_value(obj=response, type_or_value=dict, name="Responses API response")
        assert_keys(obj=response, keys=['id', 'object', 'status', 'output'], mode="required", name="Responses API response")
        assert_type_value(obj=response['id'], type_or_value=str, name="key 'id' in Responses API response")
        assert_log(expression=bool(response['id'].strip()), message=f"Expected a non-whitespace character in key 'id' of the Responses API response but got '{response['id']}'.")
        assert_type_value(obj=response['object'], type_or_value="response", name="key 'object' in Responses API response")
        assert_type_value(obj=response['status'], type_or_value=["completed", "incomplete", "failed", "cancelled", "queued", "in_progress"], name="key 'status' in Responses API response")
        self._validate_items(response['output'], "Responses API output")
        for item in response['output']:
            assert_keys(obj=item, keys=['type'], mode="required", name="Responses API output item")

    def _validate_items(self, items, name):
        assert_type_value(obj=items, type_or_value=list, name=name)
        call_types = ["function_call", "custom_tool_call", "computer_call", "shell_call", "apply_patch_call"]
        output_types = [f"{kind}_output" for kind in call_types]
        for index, item in enumerate(items):
            label = f"item '{index}' in {name}"
            assert_type_value(obj=item, type_or_value=dict, name=label)
            if 'type' in item:
                assert_type_value(obj=item['type'], type_or_value=str, name=f"key 'type' in {label}")
            if item.get('type') in call_types or item.get('type') in output_types:
                if item.get('type') == "function_call_output" and item.get('call_id') is None:
                    assert_keys(obj=item, keys=['name', 'caller'], mode="required", name=label)
                    assert_type_value(obj=item['name'], type_or_value=str, name=f"key 'name' in {label}")
                    assert_type_value(obj=item['caller'], type_or_value=dict, name=f"key 'caller' in {label}")
                    assert_log(expression=bool(item['name'].strip()), message=f"Expected a nonempty tool name in {label}.")
                else:
                    assert_keys(obj=item, keys=['call_id'], mode="required", name=label)
                    assert_type_value(obj=item['call_id'], type_or_value=str, name=f"key 'call_id' in {label}")
                    assert_log(expression=bool(item['call_id'].strip()), message=f"Expected a non-whitespace character in key 'call_id' of {label} but got '{item['call_id']}'.")
                if item.get('caller') is not None:
                    assert_type_value(obj=item['caller'], type_or_value=dict, name=f"key 'caller' in {label}")
                    assert_keys(obj=item['caller'], keys=['type'], mode="required", name=f"key 'caller' in {label}")
                    assert_type_value(obj=item['caller']['type'], type_or_value=["direct", "program"], name=f"caller type in {label}")
                    if item['caller']['type'] == "program":
                        assert_keys(obj=item['caller'], keys=['caller_id'], mode="required", name=f"key 'caller' in {label}")
                        assert_type_value(obj=item['caller']['caller_id'], type_or_value=str, name=f"caller ID in {label}")
                        assert_log(expression=bool(item['caller']['caller_id'].strip()), message=f"Expected a nonempty caller ID in {label}.")
                if item.get('namespace') is not None:
                    assert_type_value(obj=item['namespace'], type_or_value=str, name=f"key 'namespace' in {label}")
                if item.get('async') is not None:
                    assert_type_value(obj=item['async'], type_or_value=bool, name=f"key 'async' in {label}")
            if item.get('type') == "function_call":
                assert_keys(obj=item, keys=['name', 'arguments'], mode="required", name=label)
                for key in ['name', 'arguments']:
                    assert_type_value(obj=item[key], type_or_value=str, name=f"key '{key}' in {label}")
                assert_log(expression=bool(item['name'].strip()), message=f"Expected a nonempty function name in {label}.")
            if item.get('type') == "custom_tool_call":
                assert_keys(obj=item, keys=['name', 'input'], mode="required", name=label)
                for key in ['name', 'input']:
                    assert_type_value(obj=item[key], type_or_value=str, name=f"key '{key}' in {label}")
                assert_log(expression=bool(item['name'].strip()), message=f"Expected a nonempty custom tool name in {label}.")
            if item.get('type') in ["computer_call", "shell_call", "apply_patch_call"]:
                key = "operation" if item['type'] == "apply_patch_call" else "action"
                if item['type'] == "computer_call" and 'actions' in item:
                    key = "actions"
                assert_keys(obj=item, keys=[key], mode="required", name=label)
                assert_type_value(obj=item[key], type_or_value=list if key == "actions" else dict, name=f"key '{key}' in {label}")
                if key == "actions":
                    for action in item[key]:
                        assert_type_value(obj=action, type_or_value=dict, name=f"action in {label}")
            if item.get('type') in output_types and item['type'] != "apply_patch_call_output":
                assert_keys(obj=item, keys=['output'], mode="required", name=label)
            if item.get('type') in ["function_call_output", "custom_tool_call_output"]:
                assert_type_value(obj=item['output'], type_or_value=[str, list], name=f"key 'output' in {label}")
            if item.get('type') == "item_reference" or item.get('type') is None and 'id' in item and 'role' not in item and 'content' not in item:
                assert_keys(obj=item, keys=['id'], mode="required", name=label)
                assert_type_value(obj=item['id'], type_or_value=str, name=f"key 'id' in {label}")
                assert_log(expression=bool(item['id'].strip()), message=f"Expected a nonempty item reference ID in {label}.")
            elif item.get('type') in [None, "message"]:
                assert_keys(obj=item, keys=['role', 'content'], mode="required", name=label)
                assert_type_value(obj=item['role'], type_or_value=["user", "assistant", "system", "developer"], name=f"key 'role' in {label}")
                assert_type_value(obj=item['content'], type_or_value=[str, list], name=f"key 'content' in {label}")
                if isinstance(item['content'], list):
                    for part in item['content']:
                        assert_type_value(obj=part, type_or_value=dict, name=f"content part in {label}")
                        assert_keys(obj=part, keys=['type'], mode="required", name=f"content part in {label}")
                        assert_type_value(obj=part['type'], type_or_value=str, name=f"key 'type' of content part in {label}")
                        if part.get('type') in ["input_text", "output_text"]:
                            assert_keys(obj=part, keys=['text'], mode="required", name=f"content part in {label}")
                            assert_type_value(obj=part['text'], type_or_value=str, name=f"content text in {label}")
                        elif part['type'] == "refusal":
                            assert_keys(obj=part, keys=['refusal'], mode="required", name=f"content part in {label}")
                            assert_type_value(obj=part['refusal'], type_or_value=str, name=f"key 'refusal' of content part in {label}")

    def _wrap_start(self, function, args):
        if not self._initialized:
            return False
        self._cancel.clear()
        self._cancel.deadline = None
        self._pending_context = None
        self._pending_tools = None
        self._pending_openai_reasoning = set()
        if function.__name__ in ["prompt", "compact"]:
            self._active.set()
        return True

    def _retry_allowed(self, function, response):
        return not self._cancel.is_set()

    def _wrap_finish(self, function, response, operation_response, state):
        if state:
            if response[0] and self._pending_context is not None:
                self._messages = self._pending_context
                self._openai_reasoning = (self._openai_reasoning | self._pending_openai_reasoning) & {
                    (item.get('id'), item.get('encrypted_content')) for item in self._messages if item.get('type') == "reasoning"
                    and isinstance(item.get('id'), str) and isinstance(item.get('encrypted_content'), str)
                }
            if response[0] and self._pending_tools is not None:
                self._tools = self._pending_tools
            self._pending_context = None
            self._pending_tools = None
            self._pending_openai_reasoning = set()
            self._cancel.deadline = None
            self._active.clear()
        return response

    def interrupt(self):
        if self._active.is_set():
            self._cancel.set()
            message = "Requested interruption of the active response operation."
        else:
            message = "There is no response operation in progress that could be interrupted."
        self._logger.info(message)
        return True, message

    def _request(self, method, api_url, data, api_key, stream=False):
        # construct payload
        headers = {
            'Content-Type': "application/json",
            'HTTP-Referer': "https://github.com/ptzld/nimbro-api",
            'X-Title': "NimbRo API"
        }
        if api_key:
            headers['Authorization'] = f"Bearer {api_key}"
        deadline = self._cancel.deadline
        timeout = self._settings['timeout_response']
        if timeout is not None:
            limit = time.perf_counter() + timeout
            self._cancel.deadline = limit if deadline is None else min(deadline, limit)
        response_deadline = self._cancel.deadline
        if stream and self._settings['timeout_chunk_first'] is not None:
            limit = time.perf_counter() + self._settings['timeout_chunk_first']
            self._cancel.deadline = limit if response_deadline is None else min(response_deadline, limit)
        try:
            # use API
            response = http_request(
                method=method,
                api_url=api_url,
                headers=headers,
                json=data,
                stream=stream,
                timeout=(self._settings['timeout_connect'], self._settings['timeout_read']),
                cancel_event=self._cancel,
                logger=self._logger
            )
            # parse response
            if stream:
                with response:
                    if not response.is_success:
                        error = b"".join(response.iter_content(chunk_size=1024)).decode("utf-8", errors="replace")
                        return False, f"Responses API returned HTTP '{response.status_code}': {error}", None
                    result = self._read_stream(response, response_deadline)
            else:
                if not response.is_success:
                    return False, f"Responses API returned HTTP '{response.status_code}': {response.text}", None
                result = response.json()
                if self._settings['logger_debug_chunks']:
                    self._logger.debug(f"Received response: {format_obj(result)}.")
            if self._cancel.is_set() or self._cancel.deadline is not None and time.perf_counter() >= self._cancel.deadline:
                raise HttpRequestCancelled("Interrupted Responses request.")
            return True, "Received Responses API response.", result
        except HttpRequestCancelled:
            if self._cancel.is_set():
                return False, "Responses API request interrupted.", None
            return False, "Responses API response/chunk timeout expired.", None
        except (httpx.HTTPError, ValueError, OSError) as e:
            return False, f"Failed Responses API request: {e}", None
        finally:
            self._cancel.deadline = deadline

    def _read_stream(self, response, response_deadline):
        decoder = codecs.getincrementaldecoder("utf-8-sig")()
        pending, data, output = [], [], {}
        skip_lf = False
        for chunk in response.iter_chunks():
            if self._cancel.is_set():
                raise HttpRequestCancelled("Interrupted Responses stream.")
            decoded = decoder.decode(chunk)
            if skip_lf and decoded:
                if decoded.startswith("\n"):
                    decoded = decoded[1:]
                skip_lf = False
            if not decoded:
                continue
            skip_lf = decoded.endswith("\r")
            lines = decoded.replace("\r\n", "\n").replace("\r", "\n").split("\n")
            for index, line in enumerate(lines):
                if index == len(lines) - 1:
                    if line:
                        pending.append(line)
                    continue
                if pending:
                    pending.append(line)
                    line = "".join(pending)
                    pending = []
                if self._cancel.is_set():
                    raise HttpRequestCancelled("Interrupted Responses stream.")
                if line.startswith(":"):
                    continue
                if line.startswith("data:"):
                    data.append(line[5:].lstrip(" "))
                elif not line and data:
                    raw = "\n".join(data)
                    data = []
                    if raw != "[DONE]":
                        event = json.loads(raw)
                        if not isinstance(event, dict):
                            raise ValueError("Expected a JSON object in a Responses stream event.")
                        now = time.perf_counter()
                        if self._cancel.deadline is not None and now >= self._cancel.deadline:
                            raise HttpRequestCancelled("Responses stream exceeded its response/chunk deadline.")
                        timeout = self._settings['timeout_chunk_next']
                        limit = None if timeout is None else now + timeout
                        self._cancel.deadline = response_deadline if limit is None else limit if response_deadline is None else min(response_deadline, limit)
                        if self._settings['logger_debug_chunks']:
                            self._logger.debug(format_obj(event))
                        kind = event.get('type')
                        if kind == "response.output_item.done":
                            index = event.get('output_index')
                            if type(index) is not int or index < 0 or not isinstance(event.get('item'), dict):
                                raise ValueError("Expected an output index and item in a completed output event.")
                            output[index] = event['item']
                        if kind == "error":
                            raise ValueError(f"Responses stream error: {format_obj(event)}")
                        if kind in ["response.completed", "response.incomplete", "response.failed", "response.done"]:
                            terminal = event.get('response')
                            if not isinstance(terminal, dict):
                                raise ValueError("Expected a complete response in the terminal stream event.")
                            if 'output' not in terminal and kind == "response.done":
                                terminal['output'] = [output[index] for index in sorted(output)]
                            return terminal
        raise ValueError("Responses stream ended without a terminal response event.")

    def _prepare_reasoning(self, input_items):
        if isinstance(input_items, str):
            return input_items
        items = []
        adapted = 0
        flavor = self._endpoint['api_flavor']
        for item in input_items:
            if item.get('type') != "reasoning":
                items.append(item)
                continue
            if not self._settings['send_reasoning']:
                adapted += 1
                continue
            if flavor == "openai" and 'format' in item:
                summary = copy.deepcopy(item.get('summary', []))
                summary.extend({'type': "summary_text", 'text': part['text']} for part in item.get('content', []) if part.get('type') == "reasoning_text")
                if summary:
                    items.append({'type': "reasoning", 'summary': summary})
                adapted += 1
            elif (flavor == "openrouter" and 'format' not in item and isinstance(item.get('id'), str)
                  and isinstance(item.get('encrypted_content'), str) and (item['id'], item['encrypted_content']) in self._openai_reasoning):
                items.append(dict(item, format="openai-responses-v1"))
                adapted += 1
            else:
                items.append(item)
        if adapted:
            self._logger.debug(f"Adapted '{adapted}' reasoning item{'' if adapted == 1 else 's'} for the outgoing request. Native history is preserved.")
        return items

    def _prepare_items(self, input_items, name="argument 'input_items'"):
        assert_type_value(obj=input_items, type_or_value=[str, list], name=name)
        if isinstance(input_items, str):
            return input_items
        self._validate_items(input_items, name)
        items = copy.deepcopy(input_items)
        for item_index, item in enumerate(items):
            if item.get('type') in ["function_call_output", "custom_tool_call_output", "computer_call_output"]:
                output = item.get('output')
                parts = output if isinstance(output, list) else [output] if isinstance(output, dict) else None
                if parts is not None:
                    screenshot = item['type'] == "computer_call_output" and isinstance(output, dict) and output.get('type') == "computer_screenshot"
                    if screenshot:
                        parts[0]['type'] = "input_image"
                    message = {
                        'type': "message",
                        'role': "user",
                        'content': parts
                    }
                    converted = self._prepare_items([message], name)[0]['content']
                    if screenshot:
                        converted[0]['type'] = "computer_screenshot"
                    item['output'] = converted if isinstance(output, list) else converted[0]
                continue
            if item.get('type') not in [None, "message"]:
                continue
            if 'role' in item:
                item.setdefault('type', "message")
            content = item.get('content', [])
            if not isinstance(content, list):
                continue
            for part_index, part in enumerate(content):
                label = f"content part '{part_index}' of item '{item_index}' in {name}"
                if part.get('type') == "input_image" and part.get('image_url') is not None:
                    image = part['image_url']
                    if isinstance(image, str) and (image.startswith("data:") or (is_url(image) and not self._settings['download_image'])):
                        continue
                    success, message, encoded, _ = parse_image_b64(image=image, logger=self._logger)
                    if not success:
                        raise ValueError(message)
                    raw = base64.b64decode(encoded[:32])
                    if raw.startswith(b"\x89PNG\r\n\x1a\n"):
                        mime = "image/png"
                    elif raw.startswith(b"\xff\xd8\xff"):
                        mime = "image/jpeg"
                    elif raw.startswith((b"GIF87a", b"GIF89a")):
                        mime = "image/gif"
                    elif raw.startswith(b"RIFF") and raw[8:12] == b"WEBP":
                        mime = "image/webp"
                    else:
                        mime = None
                    assert_log(expression=mime is not None, message=f"Expected PNG, JPEG, GIF, or WebP data in key 'image_url' of {label} but got header '{raw.hex()}'.")
                    part['image_url'] = f"data:{mime};base64,{encoded}"
                elif part.get('type') in ["input_audio", "input_video"]:
                    audio = part['type'] == "input_audio"
                    target = part.get('input_audio') if audio else part
                    target_label = f"key 'input_audio' of {label}" if audio else label
                    assert_type_value(obj=target, type_or_value=dict, name=target_label)
                    key = "data" if audio else "video_url"
                    assert_keys(obj=target, keys=[key], mode="required", name=target_label)
                    if audio:
                        assert_keys(obj=target, keys=['format'], mode="required", name=target_label)
                        assert_type_value(obj=target['format'], type_or_value=str, name=f"key 'format' of {target_label}")
                    value = target[key]
                    assert_type_value(obj=value, type_or_value=[str, bytes], name=f"key '{key}' of {target_label}")
                    download = self._settings['download_audio' if audio else 'download_video']
                    if isinstance(value, str) and is_url(value) and download:
                        success, message, value = download_file(url=value, logger=self._logger)
                        if not success:
                            raise ValueError(message)
                    if isinstance(value, bytes):
                        success, message, encoded = encode_b64(obj=value, logger=self._logger)
                    elif os.path.isfile(value):
                        success, message, encoded = read_as_b64(file_path=value, logger=self._logger)
                    else:
                        continue
                    if not success:
                        raise ValueError(message)
                    if audio:
                        target[key] = encoded
                    else:
                        header = base64.b64decode(encoded[:256])
                        assert_log(expression=header[4:8] == b"ftyp" or header.startswith(b"\x1a\x45\xdf\xa3") and b"webm" in header, message=f"Expected MP4 or WebM data in key 'video_url' of {label} but got header '{header[:24].hex()}'.")
                        mime = "video/mp4" if header[4:8] == b"ftyp" else "video/webm"
                        target[key] = f"data:{mime};base64,{encoded}"
                elif part.get('type') == "input_file":
                    if part.get('file_url') is not None and self._settings['download_file']:
                        url = part['file_url']
                        success, message, value = download_file(url=url, logger=self._logger)
                        if not success:
                            raise ValueError(message)
                        part.pop('file_url')
                        part.setdefault('filename', url.split("?")[0].rsplit("/", 1)[-1] or "file")
                    else:
                        value = part.get('file_data')
                    if isinstance(value, bytes):
                        success, message, encoded = encode_b64(obj=value, logger=self._logger)
                    elif isinstance(value, str) and os.path.isfile(value):
                        part.setdefault('filename', os.path.basename(value))
                        success, message, encoded = read_as_b64(file_path=value, logger=self._logger)
                    else:
                        continue
                    if not success:
                        raise ValueError(message)
                    assert_keys(obj=part, keys=['filename'], mode="required", name=label)
                    assert_type_value(obj=part['filename'], type_or_value=str, name=f"key 'filename' of {label}")
                    mime = mimetypes.guess_type(part['filename'])[0] or "application/octet-stream"
                    part['file_data'] = f"data:{mime};base64,{encoded}"
        return items

    def get_context(self):
        return True, "Retrieved context.", copy.deepcopy(self._messages)

    def set_context(self, mode, messages, index, reverse_indexing):
        # parse arguments
        assert_type_value(obj=mode, type_or_value=["reset", "insert", "replace", "remove"], name="argument 'mode'")
        assert_type_value(obj=messages, type_or_value=[list, dict, None], name="argument 'messages'")
        assert_type_value(obj=index, type_or_value=int, name="argument 'index'")
        assert_log(expression=index >= 0, message=f"Expected argument 'index' to be non-negative but got '{index}'.")
        assert_type_value(obj=reverse_indexing, type_or_value=bool, name="argument 'reverse_indexing'")
        try:
            items = self._prepare_items([messages] if isinstance(messages, dict) else messages or [], name="argument 'messages'")
        except ValueError as e:
            return False, f"Failed to prepare context items: {e}"
        if mode == "reset":
            context = items
        else:
            context = copy.deepcopy(self._messages)
            position = len(context) - index - (mode != "insert") if reverse_indexing else index
            assert_log(expression=0 <= position <= len(context) if mode == "insert" else 0 <= position < len(context), message=f"Expected argument 'index' to select a valid context position for mode '{mode}' but got '{index}'.")
            if mode == "insert":
                assert_log(expression=bool(items), message=f"Expected at least one item in argument 'messages' for mode 'insert' but got '{len(items)}'.")
                context[position:position] = items
            elif mode == "replace":
                assert_log(expression=bool(items), message=f"Expected at least one item in argument 'messages' for mode 'replace' but got '{len(items)}'.")
                context[position:position + len(items)] = items
            else:
                assert_log(expression=not items, message=f"Expected no items in argument 'messages' for mode 'remove' but got '{len(items)}'.")
                del context[position]
        self._pending_context = context
        return True, f"Updated context to '{len(context)}' item{'' if len(context) == 1 else 's'}."

    def get_tools(self):
        return True, "Retrieved tools.", copy.deepcopy(self._tools)

    def set_tools(self, tools):
        # parse arguments
        assert_type_value(obj=tools, type_or_value=[list, dict, None], name="argument 'tools'")
        if tools is None:
            tools = []
        elif isinstance(tools, dict):
            tools = [tools]
        tools = copy.deepcopy(tools)
        used_names = set()
        for index, tool in enumerate(tools):
            assert_type_value(obj=tool, type_or_value=dict, name=f"argument 'tools[{index}]'")
            assert_keys(obj=tool, keys=['type'], mode="required", name=f"argument 'tools[{index}]'")
            assert_type_value(obj=tool['type'], type_or_value=str, name=f"argument 'tools[{index}].type'")
            if tool['type'] in ["function", "custom"]:
                assert_keys(obj=tool, keys=['name'], mode="required", name=f"argument 'tools[{index}]'")
                assert_type_value(obj=tool['name'], type_or_value=str, name=f"argument 'tools[{index}].name'")
                assert_log(expression=bool(tool['name'].strip()), message=f"Expected a non-whitespace character in argument 'tools[{index}].name' but got '{tool['name']}'.")
                assert_log(expression=tool['name'] not in used_names, message=f"Expected a unique name in argument 'tools[{index}].name' but got duplicate '{tool['name']}'.")
                used_names.add(tool['name'])
                if 'description' in tool:
                    assert_type_value(obj=tool['description'], type_or_value=[str, None], name=f"argument 'tools[{index}].description'")
                if tool['type'] == "function":
                    if 'parameters' in tool:
                        assert_type_value(obj=tool['parameters'], type_or_value=[dict, None], name=f"argument 'tools[{index}].parameters'")
                    if 'strict' in tool:
                        assert_type_value(obj=tool['strict'], type_or_value=[bool, None], name=f"argument 'tools[{index}].strict'")
        self._pending_tools = tools
        return True, f"Set '{len(tools)}' tool{'' if len(tools) == 1 else 's'}."

    def _awaited_tools(self, context, calls=None):
        if calls is None:
            calls = OrderedDict()
        call_types = ["function_call", "custom_tool_call", "computer_call", "shell_call", "apply_patch_call"]
        output_types = [f"{kind}_output" for kind in call_types]
        for item in context:
            if item.get('type') in call_types:
                if item.get('status') in [None, "completed"]:
                    calls[item['call_id']] = item
                else:
                    calls.pop(item['call_id'], None)
            elif item.get('type') in output_types:
                call_id = item.get('call_id')
                if call_id is None and item['type'] == "function_call_output":
                    matches = [key for key, call in calls.items() if call['type'] == "function_call" and call['name'] == item['name'] and call.get('namespace') == item.get('namespace') and (call.get('caller') or {'type': "direct"}) == item.get('caller')]
                    if len(matches) == 1:
                        call_id = matches[0]
                call = calls.get(call_id)
                if call is not None and f"{call['type']}_output" == item['type']:
                    calls.pop(call_id)
        return calls

    def _log_prompt(self, context, response_type):
        setting = self._settings['logger_info_prompt']
        if setting is False:
            return
        self._logger.info(f"Requesting response type '{response_type}' with '{len(context)}' context item{'' if len(context) == 1 else 's'}.")
        if setting == 0 or not context:
            return
        if setting is True:
            indices = range(len(context))
        elif setting > 0:
            indices = range(max(0, len(context) - setting), len(context))
        else:
            indices = range(min(len(context), -setting))

        threshold = self._settings['logger_info_prompt_threshold']
        cutoff = self._settings['logger_info_prompt_cutoff']

        def shorten(value, key=None):
            if isinstance(value, str) and len(value) > threshold:
                length = cutoff if key in ['image_url', 'file_url', 'file_data', 'video_url', 'data', 'encrypted_content'] else threshold
                return value[:length] + "..."
            if isinstance(value, dict):
                return {key: shorten(item, key) for key, item in value.items()}
            if isinstance(value, list):
                return [shorten(item) for item in value]
            return value

        lines = [f"{escape['blue']}>{escape['end']} {index}: {format_obj(shorten(context[index]))}" for index in indices]
        self._logger.info(f"{escape['blue']}{escape['bold']}{escape['underline']}Prompt{escape['end']}:\n" + "\n".join(lines))

    def _parse_usage(self, data):
        usage = {
            'tokens_input': None,
            'tokens_input_cached': None,
            'tokens_input_uncached': None,
            'tokens_output': None,
            'tokens_output_reasoning': None,
            'tokens_total': None
        }
        if data is None:
            self._logger.warn("Received a response without token usage.")
            return usage
        if not isinstance(data, dict):
            self._logger.warn(f"Expected a dictionary for response usage but got '{type(data).__name__}'.")
            return usage
        paths = {
            'tokens_input': ['input_tokens'],
            'tokens_input_cached': ['input_tokens_details', 'cached_tokens'],
            'tokens_output': ['output_tokens'],
            'tokens_output_reasoning': ['output_tokens_details', 'reasoning_tokens'],
            'tokens_total': ['total_tokens']
        }
        for key, path in paths.items():
            value = data
            for part in path:
                value = value.get(part) if isinstance(value, dict) else None
            if type(value) is int and value >= 0:
                usage[key] = value
            else:
                self._logger.warn(f"Expected a non-negative integer for key '{'.'.join(path)}' in response usage but got {format_obj(value)}.")
        if usage['tokens_input'] is not None:
            if usage['tokens_input_cached'] is not None and usage['tokens_input_cached'] > usage['tokens_input']:
                self._logger.warn(f"Expected cached input tokens to be at most '{usage['tokens_input']}' but got '{usage['tokens_input_cached']}'.")
                usage['tokens_input_cached'] = None
            usage['tokens_input_uncached'] = usage['tokens_input'] - (usage['tokens_input_cached'] or 0)
        if usage['tokens_output'] is not None and usage['tokens_output_reasoning'] is not None and usage['tokens_output_reasoning'] > usage['tokens_output']:
            self._logger.warn(f"Expected reasoning output tokens to be at most '{usage['tokens_output']}' but got '{usage['tokens_output_reasoning']}'.")
            usage['tokens_output_reasoning'] = None
        return usage

    def get_awaited_tools(self):
        return True, "Retrieved awaited tool IDs.", list(self._awaited_tools(self._messages))

    def _resolve_tool_choice(self, identifier):
        assert_type_value(obj=identifier, type_or_value=str, name="tool identifier in argument 'response_type'")
        assert_log(expression=identifier not in ["auto", "none", "text", "json", "always"], message=f"Expected a non-reserved tool identifier but got '{identifier}'.")
        choices = []
        for tool in self._tools:
            kind = tool['type']
            if kind in ["function", "custom"]:
                if identifier == tool['name']:
                    choices.append({'type': kind, 'name': tool['name']})
            elif kind == "mcp":
                prefix = f"mcp/{tool.get('server_label', '')}"
                if tool.get('server_label') and (identifier == prefix or identifier.startswith(prefix + "/")):
                    choice = {'type': "mcp", 'server_label': tool['server_label']}
                    if identifier != prefix:
                        name = identifier[len(prefix) + 1:]
                        assert_log(expression=bool(name) and "/" not in name, message=f"Expected one MCP tool name after the server label but got '{identifier}'.")
                        choice['name'] = name
                    choices.append(choice)
            elif identifier == kind:
                choices.append({'type': kind})
        assert_log(expression=len(choices) == 1, message=f"Expected tool identifier '{identifier}' to select exactly one configured tool but got '{len(choices)}' matches.")
        return choices[0]

    def prompt(self, text, reset_context, response_type):
        stamp = time.perf_counter()

        # parse arguments
        assert_type_value(obj=text, type_or_value=[str, dict, list, None], name="argument 'text'")
        assert_type_value(obj=reset_context, type_or_value=bool, name="argument 'reset_context'")
        assert_type_value(obj=response_type, type_or_value=[str, dict], name="argument 'response_type'")
        tool_choice = "auto"
        text_format = {'type': "text"}
        parse_json = response_type == "json"
        if isinstance(response_type, dict):
            assert_log(expression=('tools' in response_type) != ('schema' in response_type), message="Expected exactly one of 'tools' or 'schema' in argument 'response_type'.")
            if 'tools' in response_type:
                assert_keys(obj=response_type, keys=['tools', 'mode'], mode="whitelist", name="argument 'response_type'")
                assert_type_value(obj=response_type['tools'], type_or_value=list, name="argument 'response_type.tools'")
                assert_log(expression=bool(response_type['tools']), message="Expected a nonempty list in argument 'response_type.tools'.")
                mode = response_type.get('mode', "auto")
                assert_type_value(obj=mode, type_or_value=["auto", "required"], name="argument 'response_type.mode'")
                tool_choice = {'type': "allowed_tools", 'mode': mode, 'tools': [self._resolve_tool_choice(identifier) for identifier in response_type['tools']]}
            else:
                assert_keys(obj=response_type, keys=['schema', 'name', 'strict', 'description'], mode="whitelist", name="argument 'response_type'")
                assert_type_value(obj=response_type['schema'], type_or_value=dict, name="argument 'response_type.schema'")
                name = response_type.get('name', "response")
                assert_type_value(obj=name, type_or_value=str, name="argument 'response_type.name'")
                assert_log(expression=bool(name.strip()), message="Expected a nonempty schema name in argument 'response_type.name'.")
                strict = response_type.get('strict', True)
                assert_type_value(obj=strict, type_or_value=bool, name="argument 'response_type.strict'")
                text_format = {
                    'type': "json_schema",
                    'name': name,
                    'schema': copy.deepcopy(response_type['schema']),
                    'strict': strict
                }
                if 'description' in response_type:
                    assert_type_value(obj=response_type['description'], type_or_value=str, name="argument 'response_type.description'")
                    text_format['description'] = response_type['description']
                tool_choice = "none"
                parse_json = True
        elif response_type in ["text", "json"]:
            tool_choice = "none"
            if parse_json:
                text_format = {'type': "json_object"}
        elif response_type == "always":
            tool_choice = "required"
        elif response_type not in ["auto", "none"]:
            tool_choice = self._resolve_tool_choice(response_type)
        assert_log(expression=response_type != "none" or reset_context or text is not None, message="Expected argument 'text' or 'reset_context' to alter context when 'response_type' is 'none' but got no context change.")
        context = [] if reset_context else list(self._messages)
        awaited = self._awaited_tools(context)
        for item in text if isinstance(text, list) else ([] if text is None else [text]):
            assert_type_value(obj=item, type_or_value=[str, dict], name="item in argument 'text'")
            if isinstance(item, str):
                if awaited:
                    call = next(iter(awaited.values()))
                    kind = f"{call['type']}_output"
                    assert_log(expression=call['type'] in ["function_call", "custom_tool_call"], message=f"Expected a native '{kind}' item for this tool result.")
                    item = {
                        'type': kind,
                        'call_id': call['call_id'],
                        'output': item
                    }
                else:
                    item = {
                        'type': "message",
                        'role': "user",
                        'content': item
                    }
            self._validate_items([item], "argument 'text'")
            context.append(item)
            self._awaited_tools([item], awaited)
        blocking = [call for call in awaited.values() if not call.get('async')]
        assert_log(expression=response_type == "none" or not blocking, message=f"Expected outputs for synchronous tools before requesting a response but got '{len(blocking)}' unanswered calls.")
        assert_log(expression=response_type != "always" or bool(self._tools), message=f"Expected at least one configured tool for argument 'response_type' to be 'always' but got '{len(self._tools)}'.")

        # parse media
        try:
            context = self._prepare_items(context, name="prompt context")
        except ValueError as e:
            return False, f"Failed to prepare prompt items: {e}", None
        if response_type == "none":
            self._pending_context = context
            return True, "Updated context without generating a response.", None
        self._log_prompt(context, response_type)

        # construct payload
        request_options = {
            'tool_choice': tool_choice,
            'text': {'format': text_format}
        }
        if self._settings['verbosity'] is not None and self._endpoint['api_flavor'] != "vllm":
            request_options['text']['verbosity'] = self._settings['verbosity']

        # use API
        success, message, response = self._generate(context, request_options)
        if not success:
            return False, message, None

        # parse response
        output = response['output']
        parts = [part for item in output if item.get('type') == "message" and isinstance(item.get('content'), list) for part in item['content']]
        refusals = [part['refusal'] for part in parts if part.get('type') == "refusal"]
        if refusals:
            reason = "\n".join(refusals).strip() or "No refusal reason provided."
            return False, f"Model refused the request: {reason}", None
        completion = "".join(part['text'] for part in parts if part.get('type') == "output_text")
        result = {
            'text': completion,
            'refusal': "",
            'reasoning': copy.deepcopy([item for item in output if item.get('type') == "reasoning"]),
            'tools': copy.deepcopy(list(self._awaited_tools(output).values())),
            'usage': self._parse_usage(response.get('usage')),
            'response': response
        }
        if parse_json:
            try:
                result['text'] = json.loads(completion)
            except ValueError:
                result['text'] = None
                if response['status'] == "completed" and not result['tools']:
                    return False, "Failed to parse structured response text as JSON.", None

        # finalize response
        if self._settings['logger_info_response']:
            reasoning = "\n".join(part['text'] for item in result['reasoning'] if isinstance(item.get('summary'), list) for part in item['summary'] if isinstance(part, dict) and isinstance(part.get('text'), str))
            tool_messages = []
            for index, tool in enumerate(result['tools']):
                tool = copy.deepcopy(tool)
                if tool['type'] == "function_call":
                    try:
                        tool['arguments'] = json.loads(tool['arguments'])
                    except ValueError:
                        pass
                prefix = f"{index}: " if len(result['tools']) > 1 else ""
                tool_messages.append(prefix + json.dumps(tool, indent=2))
            if sum([reasoning != "", completion != "", bool(tool_messages)]) > 1:
                response_msg = f"{escape['green']}{escape['bold']}{escape['underline']}Mixed-response{escape['end']}:\n"
                if reasoning != "":
                    response_msg += f"\n{escape['yellow']}{escape['bold']}>{escape['end']} Reasoning\n{escape['yellow']}{escape['bold']}'{escape['end']}\n{reasoning}\n{escape['yellow']}{escape['bold']}'{escape['end']}\n"
                if completion != "":
                    response_msg += f"\n{escape['cyan']}{escape['bold']}>{escape['end']} Text\n{escape['cyan']}{escape['bold']}'{escape['end']}\n{completion}\n{escape['cyan']}{escape['bold']}'{escape['end']}\n"
                if tool_messages:
                    tool_msg = ',\n'.join(tool_messages)
                    if len(tool_messages) > 1:
                        tool_msg = f"[{tool_msg}]"
                    response_msg += f"\n{escape['white']}{escape['bold']}>{escape['end']} Tool-call{'' if len(tool_messages) == 1 else 's'}\n{tool_msg}\n"
            elif reasoning != "":
                response_msg = f"{escape['green']}{escape['bold']}{escape['underline']}Reasoning-response{escape['end']}:\n{escape['yellow']}{escape['bold']}'{escape['end']}\n{reasoning}\n{escape['yellow']}{escape['bold']}'{escape['end']}"
            elif completion != "":
                response_msg = f"{escape['green']}{escape['bold']}{escape['underline']}Text-response{escape['end']}:\n{escape['cyan']}{escape['bold']}'{escape['end']}\n{completion}\n{escape['cyan']}{escape['bold']}'{escape['end']}"
            elif tool_messages:
                tool_msg = '\n'.join(tool_messages)
                response_msg = f"{escape['green']}{escape['bold']}{escape['underline']}Tool-response{escape['end']}:\n{tool_msg}"
            else:
                response_msg = f"{escape['green']}{escape['bold']}{escape['underline']}Response{escape['end']}:\nReceived '{len(output)}' output item{'' if len(output) == 1 else 's'}."
            self._logger.info(response_msg)
        self._pending_context = context + copy.deepcopy(output)
        if self._endpoint['api_flavor'] == "openai":
            self._pending_openai_reasoning = {
                (item['id'], item['encrypted_content']) for item in output if item.get('type') == "reasoning"
                and isinstance(item.get('id'), str) and isinstance(item.get('encrypted_content'), str)
            }
        message = f"Generated response '{response['id']}' with status '{response['status']}' in '{time.perf_counter() - stamp:.3f}s'."
        return True, message, result

    def _get_include(self):
        fields = self._settings['include']
        if fields is None:
            return None
        flavor = self._endpoint['api_flavor']
        supported = ["reasoning.encrypted_content"]
        if flavor in ["openai", "openrouter", "vllm"]:
            supported += [
                "file_search_call.results",
                "code_interpreter_call.outputs"
            ]
        if flavor == "openai":
            supported += ["web_search_call.results", "web_search_call.action.sources"]
        elif flavor == "openrouter":
            supported.append("output.encrypted_content")
        return [field for field in fields if field in supported]

    def compact(self, input_items):
        stamp = time.perf_counter()

        # parse arguments
        assert_log(expression=self._endpoint['api_flavor'] in ["openai", "openresponses"], message=f"Expected the OpenAI or Open Responses flavor for compaction but got '{self._endpoint['api_flavor']}'.")
        try:
            input_items = self._prepare_items(input_items)
        except ValueError as e:
            return False, f"Failed to prepare compaction input: {e}", None

        # retrieve API key
        success, message, api_key = self.get_api_key()
        if not success:
            raise UnrecoverableError(message)

        # construct payload
        data = {
            'model': self._settings['model'],
            'input': self._prepare_reasoning(input_items)
        }
        if self._endpoint['api_flavor'] == "openai" and self._settings['service_tier'] is not None:
            data['service_tier'] = "flex" if self._settings['service_tier'] == "price" else "priority"
        if self._settings['instructions'] is not None:
            data['instructions'] = self._settings['instructions']

        url = urlsplit(self._endpoint['api_url'])
        api_url = urlunsplit(url._replace(path=url.path.rstrip("/") + "/compact"))

        # use API
        success, message, response = self._request("POST", api_url, data, api_key)

        # parse response
        if success:
            try:
                assert_type_value(obj=response, type_or_value=dict, name="compaction response")
                assert_keys(obj=response, keys=['id', 'object', 'output'], mode="required", name="compaction response")
                assert_type_value(obj=response['id'], type_or_value=str, name="key 'id' in compaction response")
                assert_log(expression=bool(response['id'].strip()), message="Expected a nonempty compaction response ID.")
                assert_type_value(obj=response['object'], type_or_value="response.compaction", name="key 'object' in compaction response")
                self._validate_items(response['output'], "compaction output")
            except UnrecoverableError as e:
                return False, f"Failed to parse compaction response: {e}", None

        # finalize response
        if success:
            message = f"Compacted context in '{time.perf_counter() - stamp:.3f}s'."
        else:
            response = None
        return success, message, response
