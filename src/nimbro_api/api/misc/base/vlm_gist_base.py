import re
import copy
import json
import time
import datetime
import concurrent.futures

from nimbro_api.api.openai import ChatCompletions
from nimbro_api.api.nimbro_vision_servers import MmGroundingDino, Sam2Realtime

from nimbro_api.client import ClientBase
from nimbro_api.utility.io import parse_image_b64
from nimbro_api.utility.misc import UnrecoverableError, assert_type_value, assert_log, assert_keys, count_duplicates

class VlmGistBase(ClientBase):

    def __init__(self, settings, default_settings, **kwargs):
        super().__init__(settings=settings, default_settings=default_settings, **kwargs)
        self._logger.debug(f"Initialized '{type(self).__name__}' object.")
        self._initialized = True

    def set_settings(self, settings, mode="set"):
        settings = self._introduce_settings(settings=settings, mode=mode)

        # message_results
        assert_type_value(obj=settings['message_results'], type_or_value=bool, name="setting 'message_results'")

        # include_image
        assert_type_value(obj=settings['include_image'], type_or_value=bool, name="setting 'include_image'")

        # scene_description
        assert_type_value(obj=settings['scene_description'], type_or_value=dict, name="setting 'scene_description'")
        assert_keys(obj=settings['scene_description'], keys=['skip', 'chat_completions', 'system_prompt_role', 'system_prompt', 'image_prompt_role', 'image_prompt_detail', 'description_prompt_role', 'description_prompt'], mode="match", name="setting 'scene_description'")

        # scene_description.skip
        assert_type_value(obj=settings['scene_description']['skip'], type_or_value=bool, name="setting 'scene_description.skip'")

        # scene_description.chat_completions
        client = ChatCompletions()
        success, message = client.set_settings(settings=copy.deepcopy(settings['scene_description']['chat_completions']), mute=True)
        assert_log(expression=success, message=message.replace("Unrecoverable error in 'set_settings()': ", ""))
        settings['scene_description']['chat_completions'] = client.get_settings()

        # scene_description.system_prompt_role
        assert_type_value(obj=settings['scene_description']['system_prompt_role'], type_or_value=["system", "user"], name="setting 'scene_description.system_prompt_role'")

        # scene_description.system_prompt
        assert_type_value(obj=settings['scene_description']['system_prompt'], type_or_value=str, name="setting 'scene_description.system_prompt'")

        # scene_description.image_prompt_role
        assert_type_value(obj=settings['scene_description']['image_prompt_role'], type_or_value=["system", "user"], name="setting 'scene_description.image_prompt_role'")

        # scene_description.image_prompt_detail
        assert_type_value(obj=settings['scene_description']['image_prompt_detail'], type_or_value=["high", "low", "auto"], name="setting 'scene_description.image_prompt_detail'")

        # scene_description.description_prompt_role
        assert_type_value(obj=settings['scene_description']['description_prompt_role'], type_or_value=["system", "user"], name="setting 'scene_description.description_prompt_role'")

        # scene_description.description_prompt
        assert_type_value(obj=settings['scene_description']['description_prompt'], type_or_value=str, name="setting 'scene_description.description_prompt'")

        # structured_description
        assert_type_value(obj=settings['structured_description'], type_or_value=dict, name="setting 'structured_description'")
        assert_keys(obj=settings['structured_description'], keys=['skip', 'chat_completions', 'use_scene_description', 'system_prompt_role', 'system_prompt', 'image_prompt_role', 'image_prompt_detail', 'description_prompt_role', 'description_prompt', 'response_type', 'keys_required', 'keys_required_types', 'keys_optional', 'keys_optional_types'], mode="match", name="setting 'structured_description'")

        # structured_description.skip
        assert_type_value(obj=settings['structured_description']['skip'], type_or_value=bool, name="setting 'structured_description.skip'")

        # structured_description.chat_completions
        client = ChatCompletions()
        success, message = client.set_settings(settings=settings['structured_description']['chat_completions'], mute=True)
        assert_log(expression=success, message=message.replace("Unrecoverable error in 'set_settings()': ", ""))
        settings['structured_description']['chat_completions'] = client.get_settings()

        # structured_description.use_scene_description
        assert_type_value(obj=settings['structured_description']['use_scene_description'], type_or_value=bool, name="setting 'structured_description.use_scene_description'")

        # structured_description.system_prompt_role
        assert_type_value(obj=settings['structured_description']['system_prompt_role'], type_or_value=["system", "user"], name="setting 'structured_description.system_prompt_role'")

        # structured_description.system_prompt
        assert_type_value(obj=settings['structured_description']['system_prompt'], type_or_value=str, name="setting 'structured_description.system_prompt'")

        # structured_description.image_prompt_role
        assert_type_value(obj=settings['structured_description']['image_prompt_role'], type_or_value=["system", "user"], name="setting 'structured_description.image_prompt_role'")

        # structured_description.image_prompt_detail
        assert_type_value(obj=settings['structured_description']['image_prompt_detail'], type_or_value=["high", "low", "auto"], name="setting 'structured_description.image_prompt_detail'")

        # structured_description.description_prompt_role
        assert_type_value(obj=settings['structured_description']['description_prompt_role'], type_or_value=["system", "user"], name="setting 'structured_description.description_prompt_role'")

        # structured_description.description_prompt
        assert_type_value(obj=settings['structured_description']['description_prompt'], type_or_value=str, name="setting 'structured_description.description_prompt'")

        # structured_description.response_type
        assert_type_value(obj=settings['structured_description']['response_type'], type_or_value=["json", "text"], name="setting 'structured_description.response_type'")

        # structured_description.keys_required
        assert_type_value(obj=settings['structured_description']['keys_required'], type_or_value=list, name="setting 'structured_description.keys_required'")
        assert_log(expression=len(settings['structured_description']['keys_required']) > 0, message="Expected setting 'structured_description.keys_required' to be a non-empty list.")
        for item in settings['structured_description']['keys_required']:
            assert_type_value(obj=item, type_or_value=str, name="all elements of setting 'structured_description.keys_required'")
            assert_log(expression=len(item), message="Expected all elements of setting 'structured_description.keys_required' to be non-empty strings.")

        # structured_description.keys_required_types
        assert_type_value(obj=settings['structured_description']['keys_required_types'], type_or_value=list, name="setting 'structured_description.keys_required_types'")
        assert_log(expression=len(settings['structured_description']['keys_required_types']) == len(settings['structured_description']['keys_required']), message=f"Expected setting 'structured_description.keys_required_types' to be a list of length '{len(settings['structured_description']['keys_required'])}' but got '{len(settings['structured_description']['keys_required_types'])}'.")
        for item in settings['structured_description']['keys_required_types']:
            assert_type_value(obj=item, type_or_value=["str", "bool", "int", "float", "unit", "list", "bbox[int]", "bbox[float]"], name="all elements of setting 'structured_description.keys_required_types'")

        # structured_description.keys_optional
        assert_type_value(obj=settings['structured_description']['keys_optional'], type_or_value=list, name="setting 'structured_description.keys_optional'")
        for item in settings['structured_description']['keys_optional']:
            assert_type_value(obj=item, type_or_value=str, name="all elements of setting 'structured_description.keys_optional'")
            assert_log(expression=len(item), message="Expected all elements of setting 'structured_description.keys_optional' to be non-empty strings.")

        # structured_description.keys_optional_types
        assert_type_value(obj=settings['structured_description']['keys_optional_types'], type_or_value=list, name="setting 'structured_description.keys_optional_types'")
        assert_log(expression=len(settings['structured_description']['keys_optional_types']) == len(settings['structured_description']['keys_optional']), message=f"Expected setting 'structured_description.keys_optional_types' to be a list of length '{len(settings['structured_description']['keys_optional'])}' but got '{len(settings['structured_description']['keys_optional_types'])}'.")
        for item in settings['structured_description']['keys_optional_types']:
            assert_type_value(obj=item, type_or_value=["str", "bool", "int", "float", "unit", "list", "bbox[int]", "bbox[float]"], name="all elements of setting 'structured_description.keys_optional_types'")

        # detection
        assert_type_value(obj=settings['detection'], type_or_value=dict, name="setting 'detection'")
        assert_keys(obj=settings['detection'], keys=['skip', 'extract_from_description', 'mmgroundingdino', 'prompt_key'], mode="match", name="setting 'detection'")

        # detection.skip
        assert_type_value(obj=settings['detection']['skip'], type_or_value=bool, name="setting 'detection.skip'")

        # detection.extract_from_description
        assert_type_value(obj=settings['detection']['extract_from_description'], type_or_value=bool, name="setting 'detection.extract_from_description'")
        if settings['detection']['extract_from_description']:
            num_required_bbox_keys = sum([t in settings['structured_description']['keys_required_types'] for t in ["bbox[int]", "bbox[float]"]])
            num_optional_bbox_keys = sum([t in settings['structured_description']['keys_optional_types'] for t in ["bbox[int]", "bbox[float]"]])
            assert_log(expression=num_required_bbox_keys == 1, message=f"Expected setting 'structured_description.keys_required_types' to contain exactly one bbox type when 'detection.extract_from_description' is 'True' but got '{num_required_bbox_keys}': {settings['structured_description']['keys_required_types']}")
            assert_log(expression=num_optional_bbox_keys == 0, message=f"Expected setting 'structured_description.keys_optional_types' to contain exactly one bbox type when 'detection.extract_from_description' is 'True' but got '{num_optional_bbox_keys}': {settings['structured_description']['keys_optional_types']}")

        # detection.mmgroundingdino
        client = MmGroundingDino()
        success, message = client.set_settings(settings=settings['detection']['mmgroundingdino'], mute=True)
        assert_log(expression=success, message=message.replace("Unrecoverable error in 'set_settings()': ", ""))
        settings['detection']['mmgroundingdino'] = client.get_settings()

        # detection.prompt_key
        assert_type_value(obj=settings['detection']['prompt_key'], type_or_value=settings['structured_description']['keys_required'], name="setting 'detection.prompt_key'")

        # segmentation
        assert_type_value(obj=settings['segmentation'], type_or_value=dict, name="setting 'segmentation'")
        assert_keys(obj=settings['segmentation'], keys=['skip', 'track', 'sam2_realtime'], mode="match", name="setting 'segmentation'")

        # segmentation.skip
        assert_type_value(obj=settings['segmentation']['skip'], type_or_value=bool, name="setting 'segmentation.skip'")

        # segmentation.track
        if settings['segmentation']['skip']:
            assert_type_value(obj=settings['segmentation']['track'], type_or_value=False, name="setting 'segmentation.track'")
        else:
            assert_type_value(obj=settings['segmentation']['track'], type_or_value=bool, name="setting 'segmentation.track'")

        # segmentation.sam2_realtime
        client = Sam2Realtime()
        success, message = client.set_settings(settings=settings['segmentation']['sam2_realtime'], mute=True)
        assert_log(expression=success, message=message.replace("Unrecoverable error in 'set_settings()': ", ""))
        settings['segmentation']['sam2_realtime'] = client.get_settings()

        # batch_size
        assert_type_value(obj=settings['batch_size'], type_or_value=int, name="setting 'batch_size'")
        assert_log(expression=settings['batch_size'] >= 0, message=f"Expected setting 'batch_size' to be non-negative but got '{settings['batch_size']}'.")

        # batch_style
        assert_type_value(obj=settings['batch_style'], type_or_value=["threading", "multiprocessing"], name="setting 'batch_style'")

        # batch_logger_severity
        assert_type_value(obj=settings['batch_logger_severity'], type_or_value=[10, 20, 30, 40, 50, None], name="setting 'batch_logger_severity'")

        # do not skip all steps
        assert_log(expression=not (settings['scene_description']['skip'] and settings['structured_description']['skip'] and settings['detection']['skip'] and settings['segmentation']['skip']), message="Expected at least one setting 'skip' to be 'False'.")

        # apply settings
        return self._apply_settings(settings, mode)

    def run(self, image, scene_description, structured_description, detection, is_worker=False):
        stamp_global = time.perf_counter()
        data = {'run': {'stamp': datetime.datetime.now().isoformat(), 'type': "normal"}}

        if is_worker:
            settings = self._settings
            data['run']['type'] = "worker"
        else:
            image, data, settings = self.parse_arguments(image=image, scene_description=scene_description, structured_description=structured_description, detection=detection, data=data)
            if isinstance(image, list):
                return self.batch_orchestrator(image=image, data=data, settings=settings, stamp_global=stamp_global)
            else:
                data['run']['settings'] = {name: settings[name] for name in ['logger_severity', 'logger_name', 'message_results', 'include_image', 'retry']}

        success, message, data = self.read_image(image=image, data=data, stamp_global=stamp_global)
        if not success:
            return False, message, data

        success, message, data = self.generate_scene_description(data=data, settings=settings, is_worker=is_worker, stamp_global=stamp_global)
        if not success:
            return False, message, data

        success, message, data = self.generate_structured_description(data=data, settings=settings, is_worker=is_worker, stamp_global=stamp_global)
        if not success:
            return False, message, data

        success, message, data = self.generate_detection(data=data, settings=settings, is_worker=is_worker, stamp_global=stamp_global)
        if not success:
            return False, message, data

        success, message, data = self.generate_segmentation(data=data, settings=settings, is_worker=is_worker, stamp_global=stamp_global)
        if not success:
            return False, message, data

        return self.finalize_result(data=data, settings=settings, stamp_global=stamp_global)

    def parse_arguments(self, image, scene_description, structured_description, detection, data):
        if isinstance(image, list):
            if len(image) == 1:
                image = image[0]
            else:
                assert_log(expression=len(image) != 0, message="Expected argument 'image' provided as 'list' to contain at least one element.")

        assert_log(expression=scene_description is None or self._settings['scene_description']['skip'], message="Expected setting 'scene_description.skip' to be 'True' when passing argument 'scene_description'.")
        assert_log(expression=structured_description is None or self._settings['structured_description']['skip'], message="Expected setting 'structured_description.skip' to be 'True' when passing argument 'structured_description'.")
        assert_log(expression=detection is None or self._settings['detection']['skip'], message="Expected setting 'detection.skip' to be 'True' when passing argument 'detection'.")
        if detection is not None:
            assert_log(expression=structured_description is not None, message="Expected argument 'structured_description' to be provided when providing argument 'detection'.")
        if not self._settings['structured_description']['skip'] and self._settings['structured_description']['use_scene_description']:
            assert_log(expression=not self._settings['scene_description']['skip'] or scene_description is not None, message="Expected to generate 'scene_description' or receive argument 'scene_description' when setting 'structured_description.use_scene_description' is 'True'.")
        if not self._settings['detection']['skip']:
            assert_log(expression=not self._settings['structured_description']['skip'] or structured_description is not None, message="Expected to generate structured description or receive argument 'structured_description' when generating detection.")
        if not self._settings['segmentation']['skip']:
            assert_log(expression=not self._settings['detection']['skip'] or detection is not None, message="Expected to generate detection or receive argument 'detection' when generating segmentation.")

        validate_settings = False
        settings = copy.deepcopy(self._settings)
        for arg, name in zip([scene_description, structured_description, detection], ["scene_description", "structured_description", "detection"]):
            if arg is not None:
                data['run'][name] = f"Provided from argument as type '{type(arg).__name__}'."
                if isinstance(arg, dict) and 'settings' in arg:
                    settings[name] = arg['settings']
                    validate_settings = True

        if validate_settings:
            from ..client.vlm_gist import VlmGist
            client = VlmGist(logger_severity=50)
            success, message = client.set_settings(settings=settings)
            assert_log(expression=success, message=message)

        for arg, name in zip([scene_description, structured_description, detection], ["scene_description", "structured_description", "detection"]):
            if arg is None:
                continue
            elif isinstance(arg, dict):
                assert_keys(obj=arg, keys=['data'], mode="required", name=f"argument '{name}' provided as 'dict'")
                names = ['stamp', 'settings', 'success', 'logs', 'duration']
                types = [str, dict, bool, list, float]
                for _name, _type in zip(names, types):
                    if _name in arg:
                        assert_type_value(obj=arg[_name], type_or_value=_type, name=f"key '{_name}' in argument '{name}' provided as 'dict'")
                if 'stamp' in arg:
                    try:
                        datetime.datetime.fromisoformat(arg['stamp'])
                    except Exception as e:
                        assert_log(expression=False, message=f"Expected value of key 'stamp' in argument '{name}' provided as 'dict' to be ISO 8601: {repr(e)}")
                data[name] = copy.deepcopy(arg)
                if 'logs' not in data[name]:
                    data[name]['logs'] = []
                else:
                    for i, item in enumerate(data[name]['logs']):
                        assert_type_value(obj=item, type_or_value=str, name=f"log '{i}' in '{name}'")
                if 'duration' in data[name]:
                    assert_log(expression=data[name]['duration'] > 0, message=f"Expected value of key 'duration' in argument '{name}' provided as 'dict' to be greater zero but got '{data[name]['duration']}s'.")
            else:
                data[name] = {'success': None, 'logs': ["Provided from argument without metadata."], 'data': arg}

            dummy_data = {
                name: {
                    'success': True,
                    'logs': [],
                    'raw': copy.deepcopy(data[name]['data']),
                }
            }

            if name == "scene_description":
                data[name]['success'], message, dummy_data = self.parse_scene_description(data=dummy_data, stamp_local=None)
                if data[name]['success']:
                    if isinstance(arg, dict):
                        if dummy_data[name]['data'] != data[name]['data']:
                            data[name]['data'] = dummy_data[name]['data']
                            data[name]['logs'].append(f"The data in argument '{name}' provided as 'dict' was altered while parsing it and may no longer correspond to the remaining metadata.")
                            self._logger.warn(data[name]['logs'][-1])
                    else:
                        dummy_data[name]['logs'] = data[name]['logs'] + dummy_data[name]['logs']
                        data[name] = dummy_data[name]
                elif self._settings['structured_description']['skip'] or not settings['structured_description']['use_scene_description']:
                    pass
                else:
                    raise UnrecoverableError(message)

            elif name == "structured_description":
                success, message, dummy_data = self.parse_structured_description(settings=settings, data=dummy_data, stamp_local=None)
                if success:
                    if not self._settings['detection']['skip'] and settings['detection']['extract_from_description'] and (not isinstance(arg, dict) or 'settings' not in arg):
                        raise UnrecoverableError("Cannot extract detection from structured description without settings being provided.")
                    if isinstance(arg, dict):
                        if dummy_data[name]['data'] != data[name]['data']:
                            data[name]['data'] = dummy_data[name]['data']
                            data[name]['logs'].append(f"The data in argument '{name}' provided as 'dict' was altered while parsing it and may no longer correspond to the remaining metadata.")
                            self._logger.warn(data[name]['logs'][-1])
                    else:
                        dummy_data[name]['logs'] = data[name]['logs'] + dummy_data[name]['logs']
                        data[name] = dummy_data[name]
                else:
                    raise UnrecoverableError(message)

            elif name == "detection":
                prompts = [item[settings['detection']['prompt_key']] for item in data['structured_description']['data']]
                dummy_data['structured_description'] = {'data': data['structured_description']['data']}
                success, message, dummy_data = self.parse_detection(data=dummy_data, prompts=prompts, stamp_local=None)
                if success:
                    if isinstance(arg, dict):
                        if dummy_data[name]['data'] != data[name]['data']:
                            data[name]['data'] = dummy_data[name]['data']
                            data[name]['logs'].append(f"The data in argument '{name}' provided as 'dict' was altered while parsing it and may no longer correspond to the remaining metadata.")
                            self._logger.warn(data[name]['logs'][-1])
                    else:
                        dummy_data[name]['logs'] = data[name]['logs'] + dummy_data[name]['logs']
                        data[name] = dummy_data[name]
                else:
                    raise UnrecoverableError(message)

        return image, data, settings

    def batch_orchestrator(self, image, data, settings, stamp_global):
        # log
        message = (
            f"Processing '{len(image)}' images using "
            f"'{settings['batch_size']}' "
            f"{('thread' if settings['batch_style'] == 'threading' else 'process')}"
            f"{'' if settings['batch_size'] == 1 else ('s' if settings['batch_style'] == 'threading' else 'es')}."
        )
        self._logger.info(message)

        # full settings here instead of individual results
        data['run']['type'] = "batch"
        data['run']['settings'] = copy.deepcopy(settings)

        # forward parsed arguments
        scene_description = data.get('scene_description')
        structured_description = data.get('structured_description')
        detection = data.get('detection')

        # configure workers
        max_workers = settings['batch_size'] if settings['batch_size'] > 0 else len(image)
        if settings['batch_style'] == "multiprocessing":
            executor_class = concurrent.futures.ProcessPoolExecutor
        else:
            executor_class = concurrent.futures.ThreadPoolExecutor

        # execute parallel batch
        with executor_class(max_workers=max_workers) as executor:
            futures = []
            for i, img in enumerate(image):
                worker_settings = copy.deepcopy(settings)
                worker_settings['logger_severity'] = worker_settings['batch_logger_severity']
                worker_settings['logger_name'] = f"{settings['logger_name']}_{i}"
                del worker_settings['batch_size']
                del worker_settings['batch_style']
                del worker_settings['batch_logger_severity']
                futures.append(executor.submit(VlmGistBase.batch_worker, img, scene_description, structured_description, detection, worker_settings))

            # evaluate in original submission order
            results = [future.result() for future in futures]

        # extract batch results
        successes = [res[0] for res in results]
        failures = len(image) - sum(successes)
        data['run']['success'] = failures == 0
        duration = time.perf_counter() - stamp_global
        data['run']['message'] = (
            f"Processed '{len(image)}' images using "
            f"'{settings['batch_size']}' "
            f"{('thread' if settings['batch_style'] == 'threading' else 'process')}"
            f"{'' if settings['batch_size'] == 1 else ('s' if settings['batch_style'] == 'threading' else 'es')} "
            f"with '{failures}' failure{'' if failures == 1 else 's'} "
            f"in '{duration:.3f}s'."
        )
        data['run']['duration'] = duration
        data['batch'] = [res[2] for res in results]
        return data['run']['success'], data['run']['message'], data

    @staticmethod
    def batch_worker(image, scene_description, structured_description, detection, settings):
        from ..client.vlm_gist import VlmGist
        client = VlmGist(settings=settings)
        return client.run(image=image, scene_description=scene_description, structured_description=structured_description, detection=detection, is_worker=True)

    def read_image(self, image, data, stamp_global):
        if isinstance(image, dict):
            assert_keys(obj=image, keys=['stamp', 'success', 'logs', 'path', 'duration'], mode="blacklist", name="image provided as 'dict'")
            assert_keys(obj=image, keys=['data'], mode="required", name="image provided as 'dict'")
            metadata = copy.deepcopy(image)
            image = metadata.pop('data')
        else:
            metadata = {}

        stamp_local = time.perf_counter()
        data['image'] = {'stamp': datetime.datetime.now().isoformat()}
        data['image']['success'], message, image_data, image_path = parse_image_b64(image=image, logger=self._logger)
        data['image']['logs'] = [message]
        for key in copy.deepcopy(metadata):
            data['image'][key] = metadata.pop(key)
            data['image']['logs'].append(f"Included metadata key '{key}'.")
        if data['image']['success']:
            data['image']['data'] = image_data
            data['image']['path'] = image_path
            data['image']['duration'] = time.perf_counter() - stamp_local
            return True, message, data
        else:
            return self.consolidate_error(key='image', message=None, data=data, stamp_local=stamp_local, stamp_global=stamp_global)

    def generate_scene_description(self, data, settings, is_worker, stamp_global):
        if self._settings['scene_description']['skip']:
            return True, "Skipping scene description.", data
        else:
            stamp_local = time.perf_counter()
            data['scene_description'] = {'stamp': datetime.datetime.now().isoformat()}
            if not is_worker:
                data['scene_description']['settings'] = copy.deepcopy(settings['scene_description'])
            chat = ChatCompletions(settings=settings['scene_description']['chat_completions'])
            messages = [
                {'role': settings['scene_description']['system_prompt_role'], 'content': settings['scene_description']['system_prompt']},
                {'role': settings['scene_description']['image_prompt_role'], 'content': [{'type': "image_url", 'image_url': {'url': data['image']['data'], 'detail': settings['scene_description']['image_prompt_detail']}}]},
                {'role': settings['scene_description']['description_prompt_role'], 'content': [{'type': "text", 'text': settings['scene_description']['description_prompt']}]},
            ]
            data['scene_description']['success'], message, completion = chat.prompt(text=messages, response_type="text")
            data['scene_description']['logs'] = [message]
            if isinstance(completion, dict):
                data['scene_description']['completion'] = completion
            if data['scene_description']['success']:
                required_keys = {'text', 'usage', 'logs'}
                reserved_keys = ['stamp', 'settings', 'data', 'duration']
                success, message, data = self.parse_completion(completion=completion, required_keys=required_keys, reserved_keys=reserved_keys, data=data, data_key='scene_description', name="scene description")
                if not success:
                    return self.consolidate_error(key='scene_description', message=message, data=data, stamp_local=stamp_local, stamp_global=stamp_global)
                success, message, data = self.parse_scene_description(data=data, stamp_local=stamp_local)
                if success:
                    return True, message, data
                else:
                    return self.consolidate_error(key='scene_description', message=message, data=data, stamp_local=stamp_local, stamp_global=stamp_global)
            else:
                return self.consolidate_error(key='scene_description', message=None, data=data, stamp_local=stamp_local, stamp_global=stamp_global)

    def parse_completion(self, completion, required_keys, reserved_keys, data, data_key, name):
        if not isinstance(completion, dict):
            return False, f"Expected {name} completion to be of type 'dict' but got '{type(completion).__name__}'.", data

        completion_keys = set(completion.keys())
        if not required_keys.issubset(completion_keys):
            return False, f"Expected {name} completion to contain the keys {required_keys} but got {completion_keys}.", data

        for key in reserved_keys:
            if key in completion:
                return False, f"Expected {name} completion to not contain the reserved key '{key}'.", data

        assert completion.get('success', True)

        if not isinstance(completion['usage'], dict):
            return False, f"Expected value of key 'usage' in {name} completion to be of type 'dict' but got '{type(completion['usage']).__name__}'.", data

        if not isinstance(completion['logs'], list):
            return False, f"Expected value of key 'logs' in {name} completion to be of type 'list' but got '{type(completion['logs']).__name__}'.", data
        for i, item in enumerate(completion['logs']):
            if not isinstance(item, str):
                return False, f"Expected value '{i}' in logs in {name} completion to be of type 'str' but got '{type(item).__name__}'.", data

        for key in required_keys:
            data[data_key][key] = data[data_key]['completion'].pop(key)

        assert 'text' in required_keys, required_keys
        data[data_key]['raw'] = data[data_key].pop('text')

        if len(data[data_key]['completion']) == 0:
            del data[data_key]['completion']
        else:
            data[data_key]['logs'].append(f"The {name} completion contains '{len(data[data_key]['completion'])}' excessive key{'' if len(data[data_key]['completion']) == 1 else 's'}: {list(data[data_key]['completion'].keys())}")
            self._logger.warn(data[data_key]['logs'][-1])

        data[data_key]['logs'].append(f"Validated {name} completion.")

        return True, data[data_key]['logs'][-1], data

    def parse_scene_description(self, data, stamp_local):
        description = copy.deepcopy(data['scene_description']['raw'])

        if not isinstance(description, str):
            return False, f"Expected scene description to be of type 'str' but got '{type(description).__name__}'.", data
        if len(description) == 0:
            return False, "Expected scene description to be a non-empty string.", data

        if data['scene_description']['raw'] == description:
            del data['scene_description']['raw']
            data['scene_description']['logs'].append("Removed raw data identical to validated data.")
            self._logger.debug(data['scene_description']['logs'][-1])
        data['scene_description']['data'] = description
        data['scene_description']['logs'].append("Validated scene description.")
        if stamp_local is not None:
            data['scene_description']['duration'] = time.perf_counter() - stamp_local

        return True, data['scene_description']['logs'][-1], data

    def generate_structured_description(self, data, settings, is_worker, stamp_global):
        if self._settings['structured_description']['skip']:
            return True, "Skipping structured description.", data
        else:
            stamp_local = time.perf_counter()
            data['structured_description'] = {'stamp': datetime.datetime.now().isoformat()}
            if not is_worker:
                data['structured_description']['settings'] = copy.deepcopy(settings['structured_description'])
            chat = ChatCompletions(settings=settings['structured_description']['chat_completions'])
            messages = [
                {'role': settings['structured_description']['system_prompt_role'], 'content': settings['structured_description']['system_prompt']},
                {'role': settings['structured_description']['image_prompt_role'], 'content': [{'type': "image_url", 'image_url': {'url': data['image']['data'], 'detail': settings['structured_description']['image_prompt_detail']}}]},
            ]
            if settings['structured_description']['use_scene_description']:
                messages.append({'role': settings['scene_description']['description_prompt_role'], 'content': [{'type': "text", 'text': settings['scene_description']['description_prompt']}]})
                messages.append({'role': "assistant", 'content': data['scene_description']['data']})
            messages.append({'role': settings['structured_description']['description_prompt_role'], 'content': [{'type': "text", 'text': settings['structured_description']['description_prompt']}]})
            data['structured_description']['success'], message, completion = chat.prompt(text=messages, reset_context=True, response_type=settings['structured_description']['response_type'])
            data['structured_description']['logs'] = [message]
            if isinstance(completion, dict):
                data['structured_description']['completion'] = completion
            if data['structured_description']['success']:
                required_keys = {'text', 'usage', 'logs'}
                reserved_keys = ['stamp', 'settings', 'raw', 'data', 'duration']
                success, message, data = self.parse_completion(completion=completion, required_keys=required_keys, reserved_keys=reserved_keys, data=data, data_key='structured_description', name="structured description")
                if not success:
                    return self.consolidate_error(key='structured_description', message=message, data=data, stamp_local=stamp_local, stamp_global=stamp_global)
                success, message, data = self.parse_structured_description(settings=settings, data=data, stamp_local=stamp_local)
                if success:
                    return True, message, data
                else:
                    # TODO trigger correction using message
                    return self.consolidate_error(key='structured_description', message=message, data=data, stamp_local=stamp_local, stamp_global=stamp_global)
            else:
                return self.consolidate_error(key='structured_description', message=None, data=data, stamp_local=stamp_local, stamp_global=stamp_global)

    def parse_structured_description(self, settings, data, stamp_local):
        description = copy.deepcopy(data['structured_description']['raw'])

        # extract structured description from string
        if isinstance(description, str):
            opening_indices = []
            opening_indices.extend([m.start() for m in re.finditer(r'[{\[]', description)])
            if len(opening_indices) > 0:
                closing_indices = []
                closing_indices.extend([m.start() for m in re.finditer(r'[}\]]', description)])
                if len(closing_indices) > 0:
                    options = []
                    for i in opening_indices:
                        for j in closing_indices:
                            if j < i:
                                continue
                            else:
                                options.append(description[i:j + 1])
                    options.sort(key=len, reverse=True)
                    for option in options:
                        try:
                            description = json.loads(option)
                        except json.JSONDecodeError:
                            pass
                        else:
                            data['structured_description']['logs'].append("Extracted possible structured description from string.")
                            self._logger.warn(data['structured_description']['logs'][-1])
                            break
                    else:
                        return False, "Failed to extract JSON from structured description received as string.", data

        # extract structured description nested in dictionary
        if isinstance(description, dict) and len(description) == 1:
            description = description[list(description.keys())[0]]
            data['structured_description']['logs'].append("Extracted possible structured description from dictionary.")
            self._logger.warn(data['structured_description']['logs'][-1])

        # structured description must be list
        if not isinstance(description, list):
            return False, f"Expected structured description to be of type 'list' but got '{type(description).__name__}'.", data

        # validate objects

        valid_description = []

        for i, obj in enumerate(description):
            # object must be dictionary
            if not isinstance(obj, dict):
                return False, f"Expected object '{i}' in structured description to be of type 'dict' but got '{type(obj).__name__}'.", data

            valid_obj = {}
            keys_left = list(obj.keys())

            # object can feature required keys
            for j, target_key in enumerate(settings['structured_description']['keys_required'] + settings['structured_description']['keys_optional']):
                target_key_norm = re.sub(r"[.,;:_\-\s]", "", target_key).lower().strip()
                for k, source_key in enumerate(copy.deepcopy(keys_left)):
                    source_key_norm = re.sub(r"[.,;:_\-\s]", "", source_key).lower().strip()
                    if source_key_norm == target_key_norm:
                        # matched key
                        if source_key != target_key:
                            data['structured_description']['logs'].append(f"Matched key '{source_key}' in object '{i}' of structured description to '{target_key}'.")
                            self._logger.warn(data['structured_description']['logs'][-1])

                        expected_type = (settings['structured_description']['keys_required_types'] + settings['structured_description']['keys_optional_types'])[j]
                        val = obj[source_key]
                        is_valid = False

                        if expected_type == "str":
                            if isinstance(val, str):
                                valid_obj[target_key] = val
                                is_valid = True

                        elif expected_type == "bool":
                            if isinstance(val, str):
                                if val.lower().strip() == "true":
                                    valid_obj[target_key] = True
                                    is_valid = True
                                    data['structured_description']['logs'].append(f"Matched value '{val}' (str) of key '{target_key}' in object '{i}' of structured description to value '{valid_obj[target_key]}' (bool).")
                                    self._logger.warn(data['structured_description']['logs'][-1])
                                elif val.lower().strip() == "false":
                                    valid_obj[target_key] = False
                                    is_valid = True
                                    data['structured_description']['logs'].append(f"Matched value '{val}' (str) of key '{target_key}' in object '{i}' of structured description to value '{valid_obj[target_key]}' (bool).")
                                    self._logger.warn(data['structured_description']['logs'][-1])
                            if not is_valid and isinstance(val, bool):
                                valid_obj[target_key] = val
                                is_valid = True

                        elif expected_type == "int":
                            if isinstance(val, int) and not isinstance(val, bool):
                                valid_obj[target_key] = val
                                is_valid = True

                        elif expected_type == "float":
                            if isinstance(val, float):
                                valid_obj[target_key] = val
                                is_valid = True

                        elif expected_type == "unit":
                            if isinstance(val, float) and 0.0 <= val <= 1.0:
                                valid_obj[target_key] = val
                                is_valid = True

                        elif expected_type == "list":
                            if isinstance(val, list):
                                valid_obj[target_key] = val
                                is_valid = True

                        elif expected_type == "bbox[int]":
                            if (isinstance(val, list) and len(val) == 4 and all(isinstance(x, int) and not isinstance(x, bool) and x >= 0 for x in val) and val[2] > val[0] and val[3] > val[1]):
                                valid_obj[target_key] = val
                                is_valid = True

                        elif expected_type == "bbox[float]":
                            if (isinstance(val, list) and len(val) == 4 and all(isinstance(x, float) and x >= 0 for x in val) and val[2] > val[0] and val[3] > val[1]):
                                valid_obj[target_key] = val
                                is_valid = True

                        else:
                            raise NotImplementedError(f"Unknown key type '{expected_type}'.")

                        if not is_valid:
                            is_required = j < len(settings['structured_description']['keys_required'])
                            req_opt_str = "required" if is_required else "optional"
                            err_msg = f"Expected valid format for {req_opt_str} key '{target_key}' in object '{i}' of structured description matching type '{expected_type}' but got invalid value '{val}' of type '{type(val).__name__}'."

                            if is_required:
                                return False, err_msg, data

                            data['structured_description']['logs'].append(err_msg)
                            self._logger.warn(data['structured_description']['logs'][-1])

                        keys_left.remove(source_key)
                        break
                else:
                    if j < len(settings['structured_description']['keys_required']):
                        return False, f"Expected object '{i}' in structured description to contain the key '{target_key}'.", data
                    data['structured_description']['logs'].append(f"Object '{i}' in structured description does not contain optional key '{target_key}'.")
                    self._logger.warn(data['structured_description']['logs'][-1])

            if len(keys_left) > 0:
                data['structured_description']['logs'].append(f"Ignored '{len(keys_left)}' excessive key{'' if len(keys_left) == 1 else 's'} in object '{i}' of structured description: {keys_left}")
                self._logger.warn(data['structured_description']['logs'][-1])

            valid_description.append(valid_obj)

        if data['structured_description']['raw'] == valid_description:
            del data['structured_description']['raw']
            data['structured_description']['logs'].append("Removed raw data identical to validated data.")
            self._logger.debug(data['structured_description']['logs'][-1])
        data['structured_description']['data'] = valid_description
        data['structured_description']['logs'].append("Validated structured description.")
        if stamp_local is not None:
            data['structured_description']['duration'] = time.perf_counter() - stamp_local

        return True, data['structured_description']['logs'][-1], data

    def generate_detection(self, data, settings, is_worker, stamp_global):
        if self._settings['detection']['skip'] or len(data['structured_description']['data']) == 0:
            return True, "Skipping detection.", data
        else:
            stamp_local = time.perf_counter()
            data['detection'] = {'stamp': datetime.datetime.now().isoformat()}
            if not is_worker:
                data['detection']['settings'] = copy.deepcopy(settings['detection'])
            if settings['detection']['extract_from_description']:
                for i, item in enumerate(settings['structured_description']['keys_required_types']):
                    if "bbox" in item:
                        bbox_key = settings['structured_description']['keys_required'][i]
                        break
                data['detection']['success'] = True
                data['detection']['logs'] = ["Extracted from structured description."]
                detection = []
                prompts = []
                for item in data['structured_description']['data']:
                    prompts.append(item[settings['detection']['prompt_key']])
                    detection.append({
                        # this will not work for unit coordinates, but fixing this would require knowing the image dimensions differentiating bbox[float] and bbox[unit]
                        'box_xyxy': [int(value) for value in item[bbox_key]],
                        'prompt': prompts[-1]
                    })
            else:
                detector = MmGroundingDino(settings=settings['detection']['mmgroundingdino'])
                prompts = [item[settings['detection']['prompt_key']] for item in data['structured_description']['data']]
                data['detection']['success'], message, detection = detector.get_detections(image=data['image']['data'], prompts=prompts)
                data['detection']['logs'] = [message]
            if data['detection']['success']:
                data['detection']['raw'] = detection
                success, message, data = self.parse_detection(data=data, prompts=prompts, stamp_local=stamp_local)
                if success:
                    return True, message, data
                else:
                    return self.consolidate_error(key='detection', message=message, data=data, stamp_local=stamp_local, stamp_global=stamp_global)
            else:
                return self.consolidate_error(key='detection', message=None, data=data, stamp_local=stamp_local, stamp_global=stamp_global)

    def parse_detection(self, data, prompts, stamp_local):
        detection = copy.deepcopy(data['detection']['raw'])

        if not isinstance(detection, list):
            return False, f"Expected detection to be of type 'list' but got '{type(detection).__name__}'.", data
        required_format = {'box_xyxy': list, 'confidence': float, 'prompt': str}
        required_keys = {'box_xyxy', 'prompt'}
        detection_prompts = []
        for i, item in enumerate(detection):
            if not isinstance(item, dict):
                return False, f"Expected detection '{i}' to be of type 'dict' but got '{type(item).__name__}'.", data
            detection_keys = set(item.keys())
            if not required_keys.issubset(detection_keys):
                return False, f"Expected detection '{i}' to contain the keys {required_keys} but got {detection_keys}.", data
            for key in required_keys:
                if not isinstance(item[key], required_format[key]):
                    return False, f"Expected key '{key}' of detection '{i}' to be of type '{required_format[key].__name__}' but got '{type(item[key]).__name__}'.", data
            if len(item['box_xyxy']) != 4:
                return False, f"Expected value of key 'box_xyxy' in detection '{i}' to be a list of length '4' but got '{len(item['box_xyxy'])}'.", data
            for j, sub_item in enumerate(item['box_xyxy']):
                if not isinstance(sub_item, int):
                    return False, f"Expected element '{j}' in value of key 'box_xyxy' in detection '{i}' to be of type 'int' but got '{type(sub_item).__name__}'.", data
                if sub_item < 0:
                    return False, f"Expected element '{j}' in value of key 'box_xyxy' in detection '{i}' to be a non-negative integer but got '{sub_item}'.", data
            if item['box_xyxy'][2] <= item['box_xyxy'][0] or item['box_xyxy'][3] <= item['box_xyxy'][1]:
                return False, f"Expected value of key 'box_xyxy' in detection '{i}' to be a valid bounding box (x0, y0, x1, y1) but got '{item['box_xyxy']}'.", data
            detection_prompts.append(item['prompt'])

        prompt_counts_initial = count_duplicates(iterable=prompts, include_unique=True)
        prompt_counts = copy.deepcopy(prompt_counts_initial)
        detection_counts = count_duplicates(iterable=detection_prompts, include_unique=True)
        for prompt in detection_counts:
            if prompt not in prompt_counts_initial:
                prompt_str = prompt.replace("\n", "\\n")
                return False, f"Detected unexpected prompt '{prompt_str}' which was not requested.", data
        for prompt in prompt_counts_initial:
            prompt_str = prompt.replace("\n", "\\n")
            prompt_counts[prompt] -= detection_counts.get(prompt, 0)
            if prompt_counts[prompt] > 0:
                return False, f"Failed to detect '{prompt_counts[prompt]}' instance{'' if prompt_counts[prompt] == 1 else 's'} of prompt '{prompt_str}' requested '{prompt_counts_initial[prompt]}' time{'' if prompt_counts_initial[prompt] == 1 else 's'}.", data
            elif prompt_counts[prompt] < 0:
                data['detection']['logs'].append(f"Over-detected '{-prompt_counts[prompt]}' instance{'' if prompt_counts[prompt] == -1 else 's'} of prompt '{prompt_str}' requested '{prompt_counts_initial[prompt]}' time{'' if prompt_counts_initial[prompt] == 1 else 's'}.")
                self._logger.debug(data['detection']['logs'])

        if data['detection']['raw'] == detection:
            del data['detection']['raw']
            data['detection']['logs'].append("Removed raw data identical to validated data.")
            self._logger.debug(data['detection']['logs'][-1])
        data['detection']['data'] = detection
        data['detection']['logs'].append("Validated detection.")
        if stamp_local is not None:
            data['detection']['duration'] = time.perf_counter() - stamp_local

        return True, data['detection']['logs'][-1], data

    def generate_segmentation(self, data, settings, is_worker, stamp_global):
        if self._settings['segmentation']['skip'] or len(data['detection']['data']) == 0:
            return True, "Skipping segmentation.", data
        else:
            stamp_local = time.perf_counter()
            data['segmentation'] = {'stamp': datetime.datetime.now().isoformat()}
            if not is_worker:
                data['segmentation']['settings'] = copy.deepcopy(settings['segmentation'])
            prompts = [{'object_id': i, 'bbox': item['box_xyxy']} for i, item in enumerate(data['detection']['data'])]
            segmenter = Sam2Realtime(settings=settings['segmentation']['sam2_realtime'])
            data['segmentation']['success'], message, segmentation = segmenter.get_response(image=data['image']['data'], prompts=prompts)
            data['segmentation']['logs'] = [message]
            if data['segmentation']['success']:
                if settings['segmentation']['track']:
                    data['segmentation']['duration_init'] = time.perf_counter() - stamp_local
                    data['segmentation']['success'], message, segmentation = segmenter.get_response(image=data['image']['data'])
                    data['segmentation']['logs'].append(message)
                    if not data['segmentation']['success']:
                        return self.consolidate_error(key='segmentation', message=None, data=data, stamp_local=stamp_local, stamp_global=stamp_global)
                data['segmentation']['raw'] = segmentation
                success, message, data = self.parse_segmentation(data=data, stamp_local=stamp_local)
                if success:
                    return True, message, data
                else:
                    return self.consolidate_error(key='segmentation', message=message, data=data, stamp_local=stamp_local, stamp_global=stamp_global)
            else:
                return self.consolidate_error(key='segmentation', message=None, data=data, stamp_local=stamp_local, stamp_global=stamp_global)

    def parse_segmentation(self, data, stamp_local):
        segmentation = copy.deepcopy(data['segmentation']['raw'])

        if not isinstance(segmentation, list):
            return False, f"Expected segmentation to be of type 'list' but got '{type(segmentation).__name__}'.", data
        required_format = {'track_id': int, 'box_xyxy': list, 'mask': str}
        required_keys = set(required_format.keys())
        track_ids_grouped = {}
        for i, item in enumerate(segmentation):
            if not isinstance(item, dict):
                return False, f"Expected segmentation '{i}' to be of type 'dict' but got '{type(item).__name__}'.", data
            segmentation_keys = set(item.keys())
            if not required_keys.issubset(segmentation_keys):
                return False, f"Expected segmentation '{i}' to contain the keys {required_keys} but got {segmentation_keys}.", data
            for key in required_format:
                if not isinstance(item[key], required_format[key]):
                    return False, f"Expected key '{key}' of segmentation '{i}' to be of type '{required_format[key].__name__}' but got '{type(item[key]).__name__}'.", data
            if len(item['box_xyxy']) != 4:
                return False, f"Expected value of key 'box_xyxy' in segmentation '{i}' to be a list of length '4' but got '{len(item['box_xyxy'])}'.", data
            for j, sub_item in enumerate(item['box_xyxy']):
                if not isinstance(sub_item, int):
                    return False, f"Expected element '{j}' in value of key 'box_xyxy' in segmentation '{i}' to be of type 'int' but got '{type(sub_item).__name__}'.", data
                if sub_item < 0:
                    return False, f"Expected element '{j}' in value of key 'box_xyxy' in segmentation '{i}' to be a non-negative integer but got '{sub_item}'.", data
            if item['box_xyxy'][2] <= item['box_xyxy'][0] or item['box_xyxy'][3] <= item['box_xyxy'][1]:
                return False, f"Expected value of key 'box_xyxy' in segmentation '{i}' to be a valid bounding box (x0, y0, x1, y1) but got '{item['box_xyxy']}'.", data
            track_ids_grouped.setdefault(item['track_id'], []).append(item)

        segmentation_filtered = []
        for i in range(len(data['detection']['data'])):
            if i not in track_ids_grouped:
                return False, f"Failed to segment detection '{i}' for prompt '{data['detection']['data'][i]['prompt']}'.", data
            items = track_ids_grouped[i]
            if len(items) > 1:
                excess = len(items) - 1
                items[0] = max(items, key=lambda x: (x['box_xyxy'][2] - x['box_xyxy'][0]) * (x['box_xyxy'][3] - x['box_xyxy'][1]))
                data['segmentation']['logs'].append(f"Discarded '{excess}' excessive segmentation{'' if excess == 1 else 's'} of detection '{i}' for prompt '{data['detection']['data'][i]['prompt']}' and kept the largest one.")
                self._logger.warn(data['segmentation']['logs'][-1])
            segmentation_filtered.append(items[0])

        if data['segmentation']['raw'] == segmentation:
            del data['segmentation']['raw']
            data['segmentation']['logs'].append("Removed raw data identical to validated data.")
            self._logger.debug(data['segmentation']['logs'][-1])
        data['segmentation']['data'] = segmentation
        data['segmentation']['logs'].append("Validated segmentation.")
        data['segmentation']['duration'] = time.perf_counter() - stamp_local

        return True, data['segmentation']['logs'][-1], data

    def consolidate_error(self, key, message, data, stamp_local, stamp_global):
        if not self._settings['include_image'] and 'image' in data and 'data' in data['image']:
            del data['image']['data']

        data[key]['success'] = False

        if message is not None:
            data[key]['logs'].append(message)

        if stamp_local is not None:
            data[key]['duration'] = time.perf_counter() - stamp_local
        else:
            data[key]['duration'] = 0.0

        data['run']['success'] = False
        data['run']['message'] = data[key]['logs'][-1]
        data['run']['duration'] = time.perf_counter() - stamp_global

        return False, data['run']['message'], data

    def finalize_result(self, data, settings, stamp_global):
        if not settings['include_image']:
            del data['image']['data']

        data['run']['success'] = True
        duration = time.perf_counter() - stamp_global

        suffix = "" if data['image']['path'] is None else f" for image '{data['image']['path']}'"
        if 'detection' in data:
            num_grounded = len(data['detection']['data'])
            data['run']['message'] = f"Generated '{num_grounded}' object grounding{'' if num_grounded == 1 else 's'}{suffix} in '{duration:.3f}s'."
            if settings['message_results'] and num_grounded > 0:
                data['run']['message'] = f"{data['run']['message'][:-1]}: {[item[settings['detection']['prompt_key']] for item in data['structured_description']['data']]}"
        elif 'structured_description' in data:
            num_grounded = len(data['structured_description']['data'])
            data['run']['message'] = f"Generated '{num_grounded}' object description{'' if num_grounded == 1 else 's'}{suffix} in '{duration:.3f}s'."
            if settings['message_results'] and num_grounded > 0:
                data['run']['message'] = f"{data['run']['message'][:-1]}: {[item[settings['detection']['prompt_key']] for item in data['structured_description']['data']]}"
        else:
            data['run']['message'] = f"Generated image description{suffix} in '{duration:.3f}s'."
            if settings['message_results']:
                data['run']['message'] = f"{data['run']['message'][:-1]}: {data['scene_description']['data']}"

        data['run']['duration'] = duration

        return True, data['run']['message'], data
