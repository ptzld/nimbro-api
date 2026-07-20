import os
import re
import copy
import json
import time
import datetime
import traceback
import concurrent.futures

from nimbro_api.api.openai import ChatCompletions
from nimbro_api.api.nimbro_vision_servers import MmGroundingDino, Sam2Realtime

from nimbro_api.client import ClientBase
from nimbro_api.utility.io import read_json, decode_b64, parse_image_b64
from nimbro_api.utility.misc import UnrecoverableError, assert_type_value, assert_log, assert_keys, count_duplicates, get_image_dimensions
from nimbro_api.utility.visual import IMPORT_ERROR, visualize_detections

class VlmGistBase(ClientBase):

    def __init__(self, settings, default_settings, **kwargs):
        super().__init__(settings=settings, default_settings=default_settings, **kwargs)
        self._logger.debug(f"Initialized '{type(self).__name__}' object.")
        self._initialized = True

    def set_settings(self, settings, mode="set"):
        settings = self._introduce_settings(settings=settings, mode=mode)

        # message_process
        assert_type_value(obj=settings['message_process'], type_or_value=bool, name="setting 'message_process'")

        # message_results
        assert_type_value(obj=settings['message_results'], type_or_value=bool, name="setting 'message_results'")

        # include_image
        assert_type_value(obj=settings['include_image'], type_or_value=bool, name="setting 'include_image'")

        # scene_description
        assert_type_value(obj=settings['scene_description'], type_or_value=dict, name="setting 'scene_description'")
        assert_keys(obj=settings['scene_description'], keys=['skip', 'message_process', 'message_results', 'chat_completions', 'system_prompt_role', 'system_prompt', 'image_prompt_role', 'image_prompt_detail', 'description_prompt_role', 'description_prompt'], mode="match", name="setting 'scene_description'")

        # scene_description.skip
        assert_type_value(obj=settings['scene_description']['skip'], type_or_value=bool, name="setting 'scene_description.skip'")

        # scene_description.message_process
        assert_type_value(obj=settings['scene_description']['message_process'], type_or_value=bool, name="setting 'scene_description.message_process'")

        # scene_description.message_results
        assert_type_value(obj=settings['scene_description']['message_results'], type_or_value=bool, name="setting 'scene_description.message_results'")

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
        assert_keys(obj=settings['structured_description'], keys=['skip', 'message_process', 'message_results', 'chat_completions', 'use_scene_description', 'system_prompt_role', 'system_prompt', 'image_prompt_role', 'image_prompt_detail', 'description_prompt_role', 'description_prompt', 'response_type', 'keys_required', 'keys_required_types', 'keys_optional', 'keys_optional_types'], mode="match", name="setting 'structured_description'")

        # structured_description.skip
        assert_type_value(obj=settings['structured_description']['skip'], type_or_value=bool, name="setting 'structured_description.skip'")

        # structured_description.message_process
        assert_type_value(obj=settings['structured_description']['message_process'], type_or_value=bool, name="setting 'structured_description.message_process'")

        # structured_description.message_results
        assert_type_value(obj=settings['structured_description']['message_results'], type_or_value=bool, name="setting 'structured_description.message_results'")

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

        types = ["str", "bool", "int", "likert5", "likert7", "float", "unit", "list", "point_xy[int]", "point_yx[int]", "point_xy[int1000]", "point_yx[int1000]", "box_xyxy[int]", "box_yxyx[int]", "box_xyxy[int1000]", "box_yxyx[int1000]"]
        box_types = "box_xyxy[int]", "box_yxyx[int]", "box_xyxy[int1000]", "box_yxyx[int1000]"

        # structured_description.keys_required_types
        assert_type_value(obj=settings['structured_description']['keys_required_types'], type_or_value=list, name="setting 'structured_description.keys_required_types'")
        assert_log(expression=len(settings['structured_description']['keys_required_types']) == len(settings['structured_description']['keys_required']), message=f"Expected setting 'structured_description.keys_required_types' to be a list of length '{len(settings['structured_description']['keys_required'])}' but got '{len(settings['structured_description']['keys_required_types'])}'.")
        for item in settings['structured_description']['keys_required_types']:
            assert_type_value(obj=item, type_or_value=types, name="all elements of setting 'structured_description.keys_required_types'")

        # structured_description.keys_optional
        assert_type_value(obj=settings['structured_description']['keys_optional'], type_or_value=list, name="setting 'structured_description.keys_optional'")
        for item in settings['structured_description']['keys_optional']:
            assert_type_value(obj=item, type_or_value=str, name="all elements of setting 'structured_description.keys_optional'")
            assert_log(expression=len(item), message="Expected all elements of setting 'structured_description.keys_optional' to be non-empty strings.")

        # structured_description.keys_optional_types
        assert_type_value(obj=settings['structured_description']['keys_optional_types'], type_or_value=list, name="setting 'structured_description.keys_optional_types'")
        assert_log(expression=len(settings['structured_description']['keys_optional_types']) == len(settings['structured_description']['keys_optional']), message=f"Expected setting 'structured_description.keys_optional_types' to be a list of length '{len(settings['structured_description']['keys_optional'])}' but got '{len(settings['structured_description']['keys_optional_types'])}'.")
        for item in settings['structured_description']['keys_optional_types']:
            assert_type_value(obj=item, type_or_value=types, name="all elements of setting 'structured_description.keys_optional_types'")

        keys = settings['structured_description']['keys_required'] + settings['structured_description']['keys_optional']
        keys_normalized = [re.sub(r"[.,;:_\-\s]", "", key).lower().strip() for key in keys]
        assert_log(expression=all(len(key) > 0 for key in keys_normalized), message="Expected settings 'structured_description.keys_required' and 'structured_description.keys_optional' to contain non-empty keys after normalization.")
        assert_log(expression=len(keys_normalized) == len(set(keys_normalized)), message="Expected settings 'structured_description.keys_required' and 'structured_description.keys_optional' to contain unique keys after normalization.")

        # detection
        assert_type_value(obj=settings['detection'], type_or_value=dict, name="setting 'detection'")
        assert_keys(obj=settings['detection'], keys=['skip', 'message_process', 'message_results', 'extract_from_description', 'prompt_key', 'mmgroundingdino', 'allow_incomplete', 'allow_excessive'], mode="match", name="setting 'detection'")

        # detection.skip
        assert_type_value(obj=settings['detection']['skip'], type_or_value=bool, name="setting 'detection.skip'")

        # detection.message_process
        assert_type_value(obj=settings['detection']['message_process'], type_or_value=bool, name="setting 'detection.message_process'")

        # detection.message_results
        assert_type_value(obj=settings['detection']['message_results'], type_or_value=bool, name="setting 'detection.message_results'")

        # detection.extract_from_description
        assert_type_value(obj=settings['detection']['extract_from_description'], type_or_value=bool, name="setting 'detection.extract_from_description'")
        if settings['detection']['extract_from_description']:
            num_required_bbox_keys = sum(t in box_types for t in settings['structured_description']['keys_required_types'])
            num_optional_bbox_keys = sum(t in box_types for t in settings['structured_description']['keys_optional_types'])
            assert_log(expression=num_required_bbox_keys == 1, message=f"Expected setting 'structured_description.keys_required_types' to contain exactly one box type when 'detection.extract_from_description' is 'True' but got '{num_required_bbox_keys}': {settings['structured_description']['keys_required_types']}")
            assert_log(expression=num_optional_bbox_keys == 0, message=f"Expected setting 'structured_description.keys_optional_types' to contain zero box types when 'detection.extract_from_description' is 'True' but got '{num_optional_bbox_keys}': {settings['structured_description']['keys_optional_types']}")

        # detection.prompt_key
        assert_type_value(obj=settings['detection']['prompt_key'], type_or_value=settings['structured_description']['keys_required'], name="setting 'detection.prompt_key'")
        prompt_key_index = settings['structured_description']['keys_required'].index(settings['detection']['prompt_key'])
        assert_log(expression=settings['structured_description']['keys_required_types'][prompt_key_index] == "str", message=f"Expected setting 'detection.prompt_key' to reference a required structured description key of type 'str' but key '{settings['detection']['prompt_key']}' has type '{settings['structured_description']['keys_required_types'][prompt_key_index]}'.")

        # detection.mmgroundingdino
        client = MmGroundingDino()
        success, message = client.set_settings(settings=settings['detection']['mmgroundingdino'], mute=True)
        assert_log(expression=success, message=message.replace("Unrecoverable error in 'set_settings()': ", ""))
        settings['detection']['mmgroundingdino'] = client.get_settings()

        # detection.allow_incomplete
        assert_type_value(obj=settings['detection']['allow_incomplete'], type_or_value=bool, name="setting 'detection.allow_incomplete'")

        # detection.allow_excessive
        assert_type_value(obj=settings['detection']['allow_excessive'], type_or_value=bool, name="setting 'detection.allow_excessive'")

        # segmentation
        assert_type_value(obj=settings['segmentation'], type_or_value=dict, name="setting 'segmentation'")
        assert_keys(obj=settings['segmentation'], keys=['skip', 'message_process', 'message_results', 'track', 'sam2_realtime', 'allow_incomplete', 'allow_excessive'], mode="match", name="setting 'segmentation'")

        # segmentation.skip
        assert_type_value(obj=settings['segmentation']['skip'], type_or_value=bool, name="setting 'segmentation.skip'")

        # segmentation.message_process
        assert_type_value(obj=settings['segmentation']['message_process'], type_or_value=bool, name="setting 'segmentation.message_process'")

        # segmentation.message_results
        assert_type_value(obj=settings['segmentation']['message_results'], type_or_value=bool, name="setting 'segmentation.message_results'")

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

        # segmentation.allow_incomplete
        assert_type_value(obj=settings['segmentation']['allow_incomplete'], type_or_value=bool, name="setting 'segmentation.allow_incomplete'")

        # segmentation.allow_excessive
        assert_type_value(obj=settings['segmentation']['allow_excessive'], type_or_value=bool, name="setting 'segmentation.allow_excessive'")

        # batch
        assert_type_value(obj=settings['batch'], type_or_value=dict, name="setting 'batch'")
        assert_keys(obj=settings['batch'], keys=['logger_severity', 'size', 'style', 'retry'], mode="match", name="setting 'batch'")

        # batch.logger_severity
        assert_type_value(obj=settings['batch']['logger_severity'], type_or_value=[0, 10, 20, 30, 40, 50, "off", "debug", "info", "warn", "error", "fatal", None], name="setting 'batch.logger_severity'")

        # batch.size
        assert_type_value(obj=settings['batch']['size'], type_or_value=int, name="setting 'batch.size'")

        assert_log(expression=settings['batch']['size'] >= 0, message=f"Expected setting 'batch.size' to be non-negative but got '{settings['batch']['size']}'.")

        # batch.style
        assert_type_value(obj=settings['batch']['style'], type_or_value=["threading", "multiprocessing"], name="setting 'batch.style'")

        # batch.retry
        assert_type_value(obj=settings['batch']['retry'], type_or_value=[int, bool], name="setting 'batch.retry'")
        if not isinstance(settings['batch']['retry'], bool):
            assert_log(
                expression=settings['batch']['retry'] >= 0,
                message=f"Expected setting 'retry' to be of type 'bool' or a non-negative 'int' but got '{settings['batch']['retry']}'."
            )
            if settings['batch']['retry'] == 0:
                settings['batch']['retry'] = False

        # do not skip all steps
        assert_log(expression=not (settings['scene_description']['skip'] and settings['structured_description']['skip'] and settings['detection']['skip'] and settings['segmentation']['skip']), message="Expected at least one setting 'skip' to be 'False'.")

        # apply settings
        return self._apply_settings(settings, mode)

    def visualize(self, result, image, output_dir, vis_args):
        # parse arguments
        assert_type_value(obj=output_dir, type_or_value=[None, str], name="argument 'output_dir'")
        if isinstance(output_dir, str):
            output_dir = os.path.realpath(output_dir)
            assert_log(expression=not os.path.isfile(output_dir), message="Expected argument 'output_dir' to not exist or point to an existing folder but it points to an existing file.")
        assert_log(expression=IMPORT_ERROR is None, message=f"Visual utilities are not available due to missing dependencies: {IMPORT_ERROR}")
        if isinstance(result, str):
            success, message, result = read_json(file_path=result, name="result", logger=self._logger)
            if not success:
                raise UnrecoverableError(message)
        assert_type_value(obj=result, type_or_value=dict, name="argument 'result'")
        assert_keys(obj=result, keys=['run'], mode="required", name="argument 'result'")
        assert_type_value(obj=result['run'], type_or_value=dict, name="value of key 'run' in argument 'result'")
        assert_keys(obj=result['run'], keys=['type'], mode="required", name="value of key 'run' in argument 'result'")
        assert_type_value(obj=result['run']['type'], type_or_value=["normal", "batch"], name="value of key 'run.type' in argument 'result'")
        num_choices = 1
        if result['run']['type'] == "batch":
            assert_keys(obj=result['run'], keys=['settings'], mode="required", name="value of key 'run' in argument 'result'")
            assert_type_value(obj=result['run']['settings'], type_or_value=dict, name="value of key 'run.settings' in argument 'result'")
            assert_keys(obj=result['run']['settings'], keys=['scene_description', 'structured_description', 'detection'], mode="required", name="value of key 'run.settings' in argument 'result'")
            assert_type_value(obj=result['run']['settings']['scene_description'], type_or_value=dict, name="value of key 'run.settings.scene_description' in argument 'result'")
            assert_type_value(obj=result['run']['settings']['structured_description'], type_or_value=dict, name="value of key 'run.settings.structured_description' in argument 'result'")
            assert_type_value(obj=result['run']['settings']['detection'], type_or_value=dict, name="value of key 'run.settings.detection' in argument 'result'")
            scene_description_settings = result['run']['settings']['scene_description']
            structured_description_settings = result['run']['settings']['structured_description']
            detection_settings = result['run']['settings']['detection']
            for settings_name, settings in [('scene_description', scene_description_settings), ('structured_description', structured_description_settings)]:
                assert_keys(obj=settings, keys=['skip'], mode="required", name=f"value of key 'run.settings.{settings_name}' in argument 'result'")
                assert_type_value(obj=settings['skip'], type_or_value=bool, name=f"value of key 'run.settings.{settings_name}.skip' in argument 'result'")
                if not settings['skip']:
                    assert_keys(obj=settings, keys=['chat_completions'], mode="required", name=f"value of key 'run.settings.{settings_name}' in argument 'result'")
                    assert_type_value(obj=settings['chat_completions'], type_or_value=dict, name=f"value of key 'run.settings.{settings_name}.chat_completions' in argument 'result'")
                    assert_keys(obj=settings['chat_completions'], keys=['choices'], mode="required", name=f"value of key 'run.settings.{settings_name}.chat_completions' in argument 'result'")
                    assert_type_value(obj=settings['chat_completions']['choices'], type_or_value=int, name=f"value of key 'run.settings.{settings_name}.chat_completions.choices' in argument 'result'")
                    assert_log(expression=not isinstance(settings['chat_completions']['choices'], bool) and settings['chat_completions']['choices'] > 0, message=f"Expected value of key 'run.settings.{settings_name}.chat_completions.choices' in argument 'result' to be a positive integer but got '{settings['chat_completions']['choices']}'.")
                    num_choices *= settings['chat_completions']['choices']
            assert_keys(obj=result, keys=['batch'], mode="required", name="argument 'result' of type 'batch'")
            assert_type_value(obj=result['batch'], type_or_value=list, name="value of key 'batch' in argument 'result'")
            batch = result['batch']
            assert_log(expression=len(batch) % num_choices == 0, message=f"Expected number of batch items '{len(batch)}' to be divisible by number of choices '{num_choices}'.")
        else:
            assert_keys(obj=result, keys=['structured_description', 'detection'], mode="required", name="argument 'result'")
            assert_type_value(obj=result['structured_description'], type_or_value=dict, name="value of key 'structured_description' in argument 'result'")
            assert_type_value(obj=result['detection'], type_or_value=dict, name="value of key 'detection' in argument 'result'")
            assert_keys(obj=result['structured_description'], keys=['settings'], mode="required", name="value of key 'structured_description' in argument 'result'")
            assert_keys(obj=result['detection'], keys=['settings'], mode="required", name="value of key 'detection' in argument 'result'")
            assert_type_value(obj=result['structured_description']['settings'], type_or_value=dict, name="value of key 'structured_description.settings' in argument 'result'")
            assert_type_value(obj=result['detection']['settings'], type_or_value=dict, name="value of key 'detection.settings' in argument 'result'")
            structured_description_settings = result['structured_description']['settings']
            detection_settings = result['detection']['settings']
            batch = [result]

        assert_type_value(obj=structured_description_settings, type_or_value=dict, name="structured description settings")
        attribute_settings = ['keys_required', 'keys_required_types', 'keys_optional', 'keys_optional_types']
        assert_keys(obj=structured_description_settings, keys=attribute_settings, mode="required", name="structured description settings")
        for key in attribute_settings:
            assert_type_value(obj=structured_description_settings[key], type_or_value=list, name=f"structured description setting '{key}'")
        assert_log(
            expression=len(structured_description_settings[attribute_settings[0]]) == len(structured_description_settings[attribute_settings[1]]),
            message=f"Expected structured descriptions settings '{attribute_settings[0]}' and '{attribute_settings[1]}' to contain the same number if items but got '{len(structured_description_settings[attribute_settings[0]])}' and '{len(structured_description_settings[attribute_settings[1]])}'.")
        assert_log(
            expression=len(structured_description_settings[attribute_settings[2]]) == len(structured_description_settings[attribute_settings[3]]),
            message=f"Expected structured descriptions settings '{attribute_settings[2]}' and '{attribute_settings[3]}' to contain the same number if items but got '{len(structured_description_settings[attribute_settings[2]])}' and '{len(structured_description_settings[attribute_settings[3]])}'.")
        point_attributes = []
        for attribute_name, attribute_type in zip(structured_description_settings['keys_required'] + structured_description_settings['keys_optional'], structured_description_settings['keys_required_types'] + structured_description_settings['keys_optional_types']):
            assert_type_value(obj=attribute_name, type_or_value=str, name="attribute name")
            assert_type_value(obj=attribute_type, type_or_value=str, name="attribute type")
            if 'point' in attribute_type:
                point_attributes.append((attribute_name, attribute_type))

        assert_type_value(obj=detection_settings, type_or_value=dict, name="detection settings")
        assert_keys(obj=detection_settings, keys=['prompt_key'], mode="required", name="detection settings")
        assert_type_value(obj=detection_settings['prompt_key'], type_or_value=str, name="detection setting 'prompt_key'")
        prompt_key = detection_settings['prompt_key']

        # import
        try:
            import cv2
            import numpy as np
        except Exception as e:
            raise UnrecoverableError("Failed to import visual dependencies (cv2, numpy).") from e

        # read image
        num_images = len(batch)
        if image is None:
            images = [None] * num_images
        else:
            if isinstance(image, list):
                image = list(image)
            elif isinstance(image, (str, os.PathLike)) and os.path.isdir(image):
                self._logger.debug("Argument 'image' provide as local path.")
                path = image
                image = sorted([os.path.join(image, item) for item in os.listdir(path) if os.path.isfile(os.path.join(path, item))])
                self._logger.debug(f"Files in '{path}': {image}")
            else:
                image = [image]

            if len(image) < num_images:
                num_input_images = num_images // num_choices
                if num_choices > 1 and len(image) <= num_input_images:
                    image_choices = []
                    for item in image:
                        for _ in range(num_choices):
                            image_choices.append(item)
                    image = image_choices
                if len(image) < num_images:
                    image += [None] * (num_images - len(image))
            elif len(image) > num_images:
                image = image[:num_images]

            images = [None] * num_images
            for i, item in enumerate(image):
                if item is not None:
                    success, message, images[i], _ = parse_image_b64(image=item, logger=self._logger)
                    if not success:
                        self._logger.error(f"Failed to visualize result '{i + 1}' of '{num_images}': {message}")

        # visualize

        self._logger.info(f"Visualizing '{num_images}' result{'' if num_images == 1 else 's'}.")

        visualizations = [None] * num_images
        paths = [None] * num_images

        for i, batch_item in enumerate(batch):
            # TODO parallelize using batch.size and batch.threading

            # validate data
            try:
                assert_type_value(obj=batch_item, type_or_value=dict, name=f"result '{i + 1}' of '{num_images}'")
                if images[i] is None:
                    assert_keys(obj=batch_item, keys=['image'], mode="required", name=f"result '{i + 1}' of '{num_images}'")
                    assert_type_value(obj=batch_item['image'], type_or_value=dict, name=f"value of key 'image' in result '{i + 1}' of '{num_images}'")
                    assert_keys(obj=batch_item['image'], keys=['data'], mode="required", name=f"value of key 'image' in result '{i + 1}' of '{num_images}'")
                    assert_type_value(obj=batch_item['image']['data'], type_or_value=str, name=f"value of key 'image.data' in result '{i + 1}' of '{num_images}'")
                    images[i] = copy.deepcopy(batch_item['image']['data'])

                # validate structured description structure
                if len(point_attributes) > 0:
                    assert_keys(obj=batch_item, keys=['structured_description'], mode="required", name=f"result '{i + 1}' of '{num_images}'")
                    assert_type_value(obj=batch_item['structured_description'], type_or_value=dict, name=f"value of key 'structured_description' in result '{i + 1}' of '{num_images}'")
                    assert_keys(obj=batch_item['structured_description'], keys=['success'], mode="required", name=f"value of key 'structured_description' in result '{i + 1}' of '{num_images}'")
                    assert_type_value(obj=batch_item['structured_description']['success'], type_or_value=True, name=f"value of key 'structured_description.success' in result '{i + 1}' of '{num_images}'")
                    assert_keys(obj=batch_item['structured_description'], keys=['data'], mode="required", name=f"value of key 'structured_description' in result '{i + 1}' of '{num_images}'")
                    assert_type_value(obj=batch_item['structured_description']['data'], type_or_value=list, name=f"value of key 'structured_description.data' in result '{i + 1}' of '{num_images}'")
                    for k, item in enumerate(batch_item['structured_description']['data']):
                        assert_type_value(obj=item, type_or_value=dict, name=f"item '{k + 1}' of value of key 'structured_description.data' in result '{i + 1}' of '{num_images}'")
                        if prompt_key in item:
                            assert_type_value(obj=item[prompt_key], type_or_value=str, name=f"value of prompt key '{prompt_key}' in item '{k + 1}' of value of key 'structured_description.data' in result '{i + 1}' of '{num_images}'")
                        for attribute_name, attribute_type in point_attributes:
                            if attribute_name in item:
                                assert_type_value(obj=item[attribute_name], type_or_value=list, name=f"value of point attribute '{attribute_name}' in item '{k + 1}' of value of key 'structured_description.data' in result '{i + 1}' of '{num_images}'")
                                assert_log(expression=len(item[attribute_name]) == 2, message=f"Expected value of point attribute '{attribute_name}' in item '{k + 1}' of value of key 'structured_description.data' in result '{i + 1}' of '{num_images}' to be a list of length '2' but got '{len(item[attribute_name])}'.")
                                assert_log(expression=all(isinstance(value, int) and not isinstance(value, bool) and value >= 0 for value in item[attribute_name]), message=f"Expected all elements of point attribute '{attribute_name}' in item '{k + 1}' of value of key 'structured_description.data' in result '{i + 1}' of '{num_images}' to be non-negative integers.")

                # validate detection structure
                assert_keys(obj=batch_item, keys=['detection'], mode="required", name=f"result '{i + 1}' of '{num_images}'")
                assert_type_value(obj=batch_item['detection'], type_or_value=dict, name=f"value of key 'detection' in result '{i + 1}' of '{num_images}'")
                assert_keys(obj=batch_item['detection'], keys=['success'], mode="required", name=f"value of key 'detection' in result '{i + 1}' of '{num_images}'")
                assert_type_value(obj=batch_item['detection']['success'], type_or_value=bool, name=f"value of key 'detection.success' in result '{i + 1}' of '{num_images}'")
                if not batch_item['detection']['success']:
                    assert_keys(obj=batch_item['detection'], keys=['logs'], mode="required", name=f"value of key 'detection' in result '{i + 1}' of '{num_images}'")
                    assert_type_value(obj=batch_item['detection']['logs'], type_or_value=list, name=f"value of key 'detection.logs' in result '{i + 1}' of '{num_images}'")
                    assert_log(expression=len(batch_item['detection']['logs']) > 0, message=f"Expected value of key 'detection.logs' in result '{i + 1}' of '{num_images}' to be non-empty.")
                else:
                    assert_keys(obj=batch_item['detection'], keys=['data'], mode="required", name=f"value of key 'detection' in result '{i + 1}' of '{num_images}'")
                    assert_type_value(obj=batch_item['detection']['data'], type_or_value=list, name=f"value of key 'detection.data' in result '{i + 1}' of '{num_images}'")
                    for k, item in enumerate(batch_item['detection']['data']):
                        assert_type_value(obj=item, type_or_value=dict, name=f"item '{k + 1}' of value of key 'detection.data' in result '{i + 1}' of '{num_images}'")
                        assert_keys(obj=item, keys=['prompt', 'box_xyxy'], mode="required", name=f"item '{k + 1}' of value of key 'detection.data' in result '{i + 1}' of '{num_images}'")
                        assert_type_value(obj=item['prompt'], type_or_value=str, name=f"value of key 'prompt' in item '{k + 1}' of value of key 'detection.data' in result '{i + 1}' of '{num_images}'")
                        assert_type_value(obj=item['box_xyxy'], type_or_value=list, name=f"value of key 'box_xyxy' in item '{k + 1}' of value of key 'detection.data' in result '{i + 1}' of '{num_images}'")
                        assert_log(expression=len(item['box_xyxy']) == 4, message=f"Expected value of key 'box_xyxy' in item '{k + 1}' of value of key 'detection.data' in result '{i + 1}' of '{num_images}' to be a list of length '4' but got '{len(item['box_xyxy'])}'.")
                        assert_log(expression=all(isinstance(value, int) and not isinstance(value, bool) and value >= 0 for value in item['box_xyxy']), message=f"Expected all elements of value of key 'box_xyxy' in item '{k + 1}' of value of key 'detection.data' in result '{i + 1}' of '{num_images}' to be non-negative integers.")
                        assert_log(expression=item['box_xyxy'][2] > item['box_xyxy'][0] and item['box_xyxy'][3] > item['box_xyxy'][1], message=f"Expected value of key 'box_xyxy' in item '{k + 1}' of value of key 'detection.data' in result '{i + 1}' of '{num_images}' to be a valid bounding box but got '{item['box_xyxy']}'.")

                # validate segmentation structure (only if present)
                if 'segmentation' in batch_item:
                    assert_type_value(obj=batch_item['segmentation'], type_or_value=dict, name=f"value of key 'segmentation' in result '{i + 1}' of '{num_images}'")
                    assert_keys(obj=batch_item['segmentation'], keys=['success'], mode="required", name=f"value of key 'segmentation' in result '{i + 1}' of '{num_images}'")
                    assert_type_value(obj=batch_item['segmentation']['success'], type_or_value=bool, name=f"value of key 'segmentation.success' in result '{i + 1}' of '{num_images}'")
                    if not batch_item['segmentation']['success']:
                        assert_keys(obj=batch_item['segmentation'], keys=['logs'], mode="required", name=f"value of key 'segmentation' in result '{i + 1}' of '{num_images}'")
                        assert_type_value(obj=batch_item['segmentation']['logs'], type_or_value=list, name=f"value of key 'segmentation.logs' in result '{i + 1}' of '{num_images}'")
                        assert_log(expression=len(batch_item['segmentation']['logs']) > 0, message=f"Expected value of key 'segmentation.logs' in result '{i + 1}' of '{num_images}' to be non-empty.")
                    else:
                        assert_type_value(obj=batch_item['detection']['success'], type_or_value=True, name=f"value of key 'detection.success' when segmentation succeeded in result '{i + 1}' of '{num_images}'")
                        assert_keys(obj=batch_item['segmentation'], keys=['data'], mode="required", name=f"value of key 'segmentation' in result '{i + 1}' of '{num_images}'")
                        assert_type_value(obj=batch_item['segmentation']['data'], type_or_value=list, name=f"value of key 'segmentation.data' in result '{i + 1}' of '{num_images}'")
                        for k, item in enumerate(batch_item['segmentation']['data']):
                            assert_type_value(obj=item, type_or_value=dict, name=f"item '{k + 1}' of value of key 'segmentation.data' in result '{i + 1}' of '{num_images}'")
                            assert_keys(obj=item, keys=['track_id', 'box_xyxy', 'mask'], mode="required", name=f"item '{k + 1}' of value of key 'segmentation.data' in result '{i + 1}' of '{num_images}'")
                            assert_type_value(obj=item['track_id'], type_or_value=int, name=f"value of key 'track_id' in item '{k + 1}' of value of key 'segmentation.data' in result '{i + 1}' of '{num_images}'")
                            assert_log(expression=not isinstance(item['track_id'], bool) and 0 <= item['track_id'] < len(batch_item['detection']['data']), message=f"Expected value of key 'track_id' in item '{k + 1}' of value of key 'segmentation.data' in result '{i + 1}' of '{num_images}' to reference an existing detection but got '{item['track_id']}'.")
                            assert_type_value(obj=item['box_xyxy'], type_or_value=list, name=f"value of key 'box_xyxy' in item '{k + 1}' of value of key 'segmentation.data' in result '{i + 1}' of '{num_images}'")
                            assert_log(expression=len(item['box_xyxy']) == 4, message=f"Expected value of key 'box_xyxy' in item '{k + 1}' of value of key 'segmentation.data' in result '{i + 1}' of '{num_images}' to be a list of length '4' but got '{len(item['box_xyxy'])}'.")
                            assert_log(expression=all(isinstance(value, int) and not isinstance(value, bool) and value >= 0 for value in item['box_xyxy']), message=f"Expected all elements of value of key 'box_xyxy' in item '{k + 1}' of value of key 'segmentation.data' in result '{i + 1}' of '{num_images}' to be non-negative integers.")
                            assert_log(expression=item['box_xyxy'][2] > item['box_xyxy'][0] and item['box_xyxy'][3] > item['box_xyxy'][1], message=f"Expected value of key 'box_xyxy' in item '{k + 1}' of value of key 'segmentation.data' in result '{i + 1}' of '{num_images}' to be a valid bounding box but got '{item['box_xyxy']}'.")
                            assert_type_value(obj=item['mask'], type_or_value=str, name=f"value of key 'mask' in item '{k + 1}' of value of key 'segmentation.data' in result '{i + 1}' of '{num_images}'")

            except UnrecoverableError as e:
                self._logger.error(f"Failed to visualize result '{i + 1}' of '{num_images}': {e}")
                continue

            if not batch_item['detection']['success']:
                self._logger.error(f"Failed to visualize result '{i + 1}' of '{num_images}': {batch_item['detection']['logs'][-1]}")
                continue

            # collect data
            detection_data = batch_item['detection']['data']
            detection_labels = [item['prompt'] for item in detection_data]
            detection_points = [None] * len(detection_labels)

            detection_indices_by_label = {}
            for detection_index, detection_label in enumerate(detection_labels):
                if detection_label not in detection_indices_by_label:
                    detection_indices_by_label[detection_label] = []
                detection_indices_by_label[detection_label].append(detection_index)

            detection_cursor_by_label = {}

            if len(point_attributes) > 0:
                for item_index, item in enumerate(batch_item['structured_description']['data']):
                    if prompt_key not in item:
                        self._logger.warn(f"Item '{item_index}' in structured description of result '{i + 1}' of '{num_images}' misses prompt key '{prompt_key}'.")
                        continue

                    label = item[prompt_key]
                    if label not in detection_indices_by_label:
                        self._logger.warn(f"Result '{i + 1}' of '{num_images}' misses detection for item '{item_index}' in structured description due to missing prompt key '{label}'.")
                        continue

                    detection_indices = detection_indices_by_label[label]
                    detection_cursor = detection_cursor_by_label.get(label, 0)
                    detection_index = detection_indices[detection_cursor]
                    detection_cursor_by_label[label] = (detection_cursor + 1) % len(detection_indices)

                    for key in item:
                        for attribute in point_attributes:
                            if key == attribute[0]:
                                if "xy" in attribute[1]:
                                    detection_points[detection_index] = (item[key][0], item[key][1])
                                else:
                                    detection_points[detection_index] = (item[key][1], item[key][0])
                                break
                        if detection_points[detection_index] is not None:
                            break

            if 'segmentation' not in batch_item or not batch_item['segmentation']['success']:
                if 'segmentation' in batch_item and not batch_item['segmentation']['success']:
                    self._logger.warn(f"There is no valid segmentation for result '{i + 1}' of '{num_images}': {batch_item['segmentation']['logs'][-1]}")
                labels = detection_labels
                points = detection_points
                boxes = [item['box_xyxy'] for item in detection_data]
                masks = None
            else:
                segmentation_data = batch_item['segmentation']['data']
                labels = [detection_data[item['track_id']]['prompt'] for item in segmentation_data]
                points = [detection_points[item['track_id']] for item in segmentation_data]
                boxes = [item['box_xyxy'] for item in segmentation_data]
                masks = [item['mask'] for item in segmentation_data]
                for j, mask in enumerate(masks):
                    success, message, masks[j] = decode_b64(string=mask, logger=self._logger)
                    if success:
                        try:
                            masks[j] = np.frombuffer(masks[j], np.uint8)
                            masks[j] = cv2.imdecode(masks[j], cv2.IMREAD_GRAYSCALE)
                            masks[j] = masks[j].astype(bool)
                        except Exception as e:
                            masks[j] = None
                            self._logger.warn(f"Failed to visualize mask of result '{i + 1}' of '{num_images}': {repr(e)}")
                    else:
                        masks[j] = None
                        self._logger.warn(f"Failed to visualize mask of result '{i + 1}' of '{num_images}': {message}")

            if sum(item is None for item in points) == len(points):
                points = None

            # draw
            self._logger.debug(f"Visualizing result '{i + 1}' of '{num_images}'.")
            tic = time.perf_counter()
            if vis_args is None:
                vis_args = {}
            try:
                visual = visualize_detections(
                    image=images[i],
                    boxes=boxes,
                    masks=masks,
                    points=points,
                    labels=labels,
                    box_format="xyxy_absolute",
                    mask_format="box_local",
                    point_format="xy_absolute",
                    **vis_args
                )
            except Exception:
                self._logger.error(f"Failed to visualize result '{i + 1}' of '{num_images}':\n{traceback.format_exc()}")
                continue
            else:
                self._logger.debug(f"Visualized result '{i + 1}' of '{num_images}' in '{time.perf_counter() - tic:.3f}s'.")
                visualizations[i] = visual

            # save
            if output_dir is not None:
                success, visual = cv2.imencode('.png', visual)
                if success:
                    visual = visual.tobytes()
                    out_path = os.path.join(output_dir, f"{i}_{datetime.datetime.now().isoformat()[:23].replace('-', '_').replace(':', '_').replace('.', '_')}.png")
                    os.makedirs(output_dir, exist_ok=True)
                    with open(out_path, "wb") as file:
                        file.write(visual)
                    paths[i] = out_path
                    self._logger.info(f"Saved visualization for result '{i + 1}' of '{num_images}' to '{out_path}'.")
                else:
                    visualizations[i] = None
                    self._logger.error(f"Failed to encode visualization for result '{i + 1}' of '{num_images}'.")
            else:
                self._logger.info(f"Visualized result '{i + 1}' of '{num_images}'.")

        # consolidate
        num_success = sum(item is not None for item in visualizations)
        # success = num_success == num_images
        success = True # to prevent any retry attempts
        if num_success == 0:
            visualizations = None
            paths = None
        elif sum(item is not None for item in paths) == 0:
            paths = None
        message = f"Visualized '{num_success}' of '{num_images}' result{'' if num_images == 1 else 's'}."

        return success, message, visualizations, paths

    def run(self, image, scene_description, structured_description, detection, is_worker=False):
        stamp_global = time.perf_counter()
        data = {'run': {'stamp': datetime.datetime.now().isoformat(), 'type': "normal"}}

        if is_worker:
            data['run']['type'] = "worker"

        image, data, settings = self.parse_arguments(image=image, scene_description=scene_description, structured_description=structured_description, detection=detection, data=data, stamp_global=stamp_global)
        if isinstance(image, list):
            assert_log(expression=not is_worker, message="Expected a batch worker to receive a single image.")
            return self.batch_orchestrator(image=image, data=data, settings=settings, scene_description=scene_description, structured_description=structured_description, detection=detection, stamp_global=stamp_global)
        scene_choices = 1 if settings['scene_description']['skip'] or scene_description is not None else settings['scene_description']['chat_completions']['choices']
        structured_choices = 1 if settings['structured_description']['skip'] or structured_description is not None else settings['structured_description']['chat_completions']['choices']
        if scene_choices * structured_choices > 1:
            return self.choices_orchestrator(image=image, data=data, settings=settings, scene_description=scene_description, structured_description=structured_description, detection=detection, scene_choices=scene_choices, structured_choices=structured_choices, is_worker=is_worker, stamp_global=stamp_global)
        if not is_worker:
            data['run']['settings'] = {name: settings[name] for name in ['logger_severity', 'logger_name', 'message_results', 'include_image', 'retry']}

        success, message, data = self.read_image(image=image, settings=settings, data=data, stamp_global=stamp_global)
        if not success:
            return False, message, data

        success, message, data = self.generate_scene_description(data=data, settings=settings, is_worker=is_worker, stamp_global=stamp_global, choices=1)
        if not success:
            return False, message, data

        success, message, data = self.generate_structured_description(data=data, settings=settings, is_worker=is_worker, stamp_global=stamp_global, choices=1)
        if not success:
            return False, message, data

        success, message, data = self.generate_detection(data=data, settings=settings, is_worker=is_worker, stamp_global=stamp_global)
        if not success:
            return False, message, data

        success, message, data = self.generate_segmentation(data=data, settings=settings, is_worker=is_worker, stamp_global=stamp_global)
        if not success:
            return False, message, data

        return self.finalize_result(data=data, settings=settings, is_worker=is_worker, stamp_global=stamp_global)

    # pipeline

    def parse_arguments(self, image, scene_description, structured_description, detection, data, stamp_global):
        if isinstance(image, list):
            assert_log(expression=len(image) > 0, message="Expected argument 'image' provided as 'list' to contain at least one element.")

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
                    settings[name] = copy.deepcopy(arg['settings'])
                    settings[name]['skip'] = self._settings[name]['skip']
                    validate_settings = True

        if validate_settings:
            from ..client.vlm_gist import VlmGist
            client = VlmGist(logger_severity=50)
            temp = copy.deepcopy(settings)
            temp['logger_severity'] = "off"
            success, message = client.set_settings(settings=temp)
            assert_log(expression=success, message=message)

        if isinstance(image, list):
            return image, data, settings

        for arg, name in zip([scene_description, structured_description, detection], ["scene_description", "structured_description", "detection"]):
            if arg is None:
                continue
            if isinstance(arg, dict):
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
                success, message, data = self.read_image(image=image, settings=settings, data=data, stamp_global=stamp_global)
                if not success:
                    raise UnrecoverableError(message)
                dummy_data['image'] = data['image']
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
                success, message, dummy_data = self.parse_detection(data=dummy_data, settings=settings, prompts=prompts, stamp_local=None)
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

    def batch_orchestrator(self, image, data, settings, scene_description, structured_description, detection, stamp_global):
        # log
        message = (
            f"Processing batch with '{len(image)}' image{'' if len(image) == 1 else 's'} using "
            f"'{settings['batch']['size'] if settings['batch']['size'] > 0 else len(image)}' "
            f"{('thread' if settings['batch']['style'] == 'threading' else 'process')}"
            f"{'' if (settings['batch']['size'] if settings['batch']['size'] > 0 else len(image)) == 1 else ('s' if settings['batch']['style'] == 'threading' else 'es')}."
        )
        if settings['message_process']:
            self._logger.info(message)
        else:
            self._logger.debug(message)

        # full settings here instead of individual results
        data['run']['type'] = "batch"
        data['run']['settings'] = copy.deepcopy(settings)
        for key in ['keys_required_types', 'keys_optional_types']:
            for i, _type in enumerate(settings['structured_description'][key]):
                data['run']['settings']['structured_description'][key][i] = _type.replace("[int1000]", "[int]")
        for arg, name in zip([scene_description, structured_description, detection], ["scene_description", "structured_description", "detection"]):
            if arg is not None and (not isinstance(arg, dict) or 'settings' not in arg):
                del data['run']['settings'][name]

        # configure workers
        max_workers = settings['batch']['size'] if settings['batch']['size'] > 0 else len(image)
        if settings['batch']['style'] == "multiprocessing":
            executor_class = concurrent.futures.ProcessPoolExecutor
        else:
            executor_class = concurrent.futures.ThreadPoolExecutor

        # execute parallel batch
        with executor_class(max_workers=max_workers) as executor:
            futures = []
            for i, img in enumerate(image):
                worker_settings = copy.deepcopy(settings)
                worker_settings['logger_severity'] = worker_settings['batch']['logger_severity']
                worker_settings['logger_name'] = f"{settings['logger_name']}-{i}"
                del worker_settings['batch']['logger_severity']
                del worker_settings['batch']['size']
                del worker_settings['batch']['style']
                worker_settings['retry'] = worker_settings['batch'].pop('retry')
                futures.append(executor.submit(VlmGistBase.batch_worker, img, scene_description, structured_description, detection, worker_settings))

            # evaluate in original submission order
            results = [future.result() for future in futures]

        # extract batch results
        successes = [res[0] for res in results]
        failures = len(image) - sum(successes)
        data['run']['success'] = failures == 0
        duration = time.perf_counter() - stamp_global
        data['run']['message'] = (
            f"Processed batch with '{len(image)}' image{'' if len(image) == 1 else 's'} using "
            f"'{settings['batch']['size'] if settings['batch']['size'] > 0 else len(image)}' "
            f"{('thread' if settings['batch']['style'] == 'threading' else 'process')}"
            f"{'' if (settings['batch']['size'] if settings['batch']['size'] > 0 else len(image)) == 1 else ('s' if settings['batch']['style'] == 'threading' else 'es')} "
            f"with '{failures}' failure{'' if failures == 1 else 's'} "
            f"in '{duration:.3f}s'."
        )
        data['run']['duration'] = duration
        data['batch'] = [item for res in results for item in res[2]]
        return data['run']['success'], data['run']['message'], data

    @staticmethod
    def batch_worker(image, scene_description, structured_description, detection, settings):
        from ..client.vlm_gist import VlmGist
        client = VlmGist(settings=settings)
        success, message, data = client.run(image=image, scene_description=scene_description, structured_description=structured_description, detection=detection, is_worker=True)
        if not isinstance(data, list):
            data = [data]
        return success, message, data

    def choices_orchestrator(self, image, data, settings, scene_description, structured_description, detection, scene_choices, structured_choices, is_worker, stamp_global):
        # log
        num_results = scene_choices * structured_choices
        if scene_choices > 1 and structured_choices > 1:
            message = f"Processing image with '{scene_choices}' scene description choices and '{structured_choices}' structured description choices producing '{num_results}' results."
        elif scene_choices > 1:
            message = f"Processing image with '{scene_choices}' scene description choices producing '{num_results}' results."
        else:
            message = f"Processing image with '{structured_choices}' structured description choices producing '{num_results}' results."
        if settings['message_process']:
            self._logger.info(message)
        else:
            self._logger.debug(message)

        # branches carry parsed arguments while the envelope only retains run metadata
        branch_base = copy.deepcopy(data)
        branch_base['run']['type'] = "worker"

        # full settings here instead of individual results
        if not is_worker:
            for key in ['image', 'scene_description', 'structured_description', 'detection']:
                if key in data:
                    del data[key]
            data['run']['type'] = "batch"
            data['run']['settings'] = copy.deepcopy(settings)
            for key in ['keys_required_types', 'keys_optional_types']:
                for i, _type in enumerate(settings['structured_description'][key]):
                    data['run']['settings']['structured_description'][key][i] = _type.replace("[int1000]", "[int]")
            for arg, name in zip([scene_description, structured_description, detection], ["scene_description", "structured_description", "detection"]):
                if arg is not None and (not isinstance(arg, dict) or 'settings' not in arg):
                    del data['run']['settings'][name]

        # execute sequential branches
        success, message, branch_base = self.read_image(image=image, settings=settings, data=branch_base, stamp_global=stamp_global)
        if not success:
            items = [copy.deepcopy(branch_base) for _ in range(num_results)]
        else:
            if scene_choices > 1:
                success, message, scene_branches = self.generate_scene_description(data=branch_base, settings=settings, is_worker=True, stamp_global=stamp_global, choices=scene_choices)
            else:
                success, message, scene_branch = self.generate_scene_description(data=branch_base, settings=settings, is_worker=True, stamp_global=stamp_global, choices=1)
                scene_branches = [scene_branch]
            items = []
            for scene_branch in scene_branches:
                if scene_branch['run'].get('success') is False:
                    items.extend(
                        copy.deepcopy(scene_branch)
                        for _ in range(structured_choices)
                    )
                    continue
                if structured_choices > 1:
                    success, message, structured_branches = self.generate_structured_description(data=scene_branch, settings=settings, is_worker=True, stamp_global=stamp_global, choices=structured_choices)
                else:
                    success, message, structured_branch = self.generate_structured_description(data=scene_branch, settings=settings, is_worker=True, stamp_global=stamp_global, choices=1)
                    structured_branches = [structured_branch]
                for structured_branch in structured_branches:
                    if structured_branch['run'].get('success') is False:
                        items.append(structured_branch)
                        continue
                    success, message, structured_branch = self.generate_detection(data=structured_branch, settings=settings, is_worker=True, stamp_global=stamp_global)
                    if not success:
                        items.append(structured_branch)
                        continue
                    success, message, structured_branch = self.generate_segmentation(data=structured_branch, settings=settings, is_worker=True, stamp_global=stamp_global)
                    if not success:
                        items.append(structured_branch)
                        continue
                    success, message, structured_branch = self.finalize_result(data=structured_branch, settings=settings, is_worker=True, stamp_global=stamp_global)
                    items.append(structured_branch)

        # extract batch results
        successes = [item['run']['success'] for item in items]
        failures = len(items) - sum(successes)
        failure_messages = [f"Failed in choice '{i + 1}' of '{num_results}' after '{item['run']['duration']:.3f}s': {item['run']['message']}" for i, item in enumerate(items) if not item['run']['success']]
        if failures == 1:
            failure_messages = failure_messages[0]
        success = failures == 0
        duration = time.perf_counter() - stamp_global

        if failures == 0:
            if scene_choices > 1 and structured_choices > 1:
                message = f"Processed image with '{scene_choices}' scene description choices and '{structured_choices}' structured description choices producing '{num_results}' results with '{failures}' failure{'' if failures == 1 else 's'} in '{duration:.3f}s'."
            elif scene_choices > 1:
                message = f"Processed image with '{scene_choices}' scene description choices producing '{num_results}' results with '{failures}' failure{'' if failures == 1 else 's'} in '{duration:.3f}s'."
            else:
                message = f"Processed image with '{structured_choices}' structured description choices producing '{num_results}' results with '{failures}' failure{'' if failures == 1 else 's'} in '{duration:.3f}s'."
        else:
            if scene_choices > 1 and structured_choices > 1:
                message = f"Processed image with '{scene_choices}' scene description choices and '{structured_choices}' structured description choices producing '{num_results}' results with '{failures}' failure{'' if failures == 1 else 's'} in '{duration:.3f}s': {failure_messages}"
            elif scene_choices > 1:
                message = f"Processed image with '{scene_choices}' scene description choices producing '{num_results}' results with '{failures}' failure{'' if failures == 1 else 's'} in '{duration:.3f}s': {failure_messages}"
            else:
                message = f"Processed image with '{structured_choices}' structured description choices producing '{num_results}' results with '{failures}' failure{'' if failures == 1 else 's'} in '{duration:.3f}s': {failure_messages}"

        if is_worker:
            return success, message, items
        data['run']['success'] = success
        data['run']['message'] = message
        data['run']['duration'] = duration
        data['batch'] = items
        return data['run']['success'], data['run']['message'], data

    def read_image(self, image, data, settings, stamp_global):
        if 'image' in data:
            # Skipping when `parse_arguments()` already read the image when a structured description was passed as an argument to allow `parse_structured_description()` to obtain image dimensions.
            return data['image']['success'], data['image']['logs'][-1], data

        if isinstance(image, dict):
            assert_keys(obj=image, keys=['stamp', 'success', 'logs', 'path', 'duration'], mode="blacklist", name="image provided as 'dict'")
            assert_keys(obj=image, keys=['data'], mode="required", name="image provided as 'dict'")
            width = image.get('width', 1)
            assert_type_value(obj=width, type_or_value=int, name="key 'width' in image provided as 'dict'")
            assert_log(expression=not isinstance(width, bool), message="Expected value of key 'width' in image provided as 'dict' to be of type 'int' but got 'bool'.")
            assert_log(expression=width > 0, message=f"Expected value of key 'width' in image provided as 'dict' to greater zero but got '{width}'.")
            height = image.get('height', 1)
            assert_type_value(obj=height, type_or_value=int, name="key 'height' in image provided as 'dict'")
            assert_log(expression=not isinstance(height, bool), message="Expected value of key 'height' in image provided as 'dict' to be of type 'int' but got 'bool'.")
            assert_log(expression=height > 0, message=f"Expected value of key 'height' in image provided as 'dict' to greater zero but got '{height}'.")
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
            if settings['message_process']:
                if data['image']['path'] is not None:
                    self._logger.info(f"Processing image '{data['image']['path']}'.")
                else:
                    self._logger.info("Processing image provided as object.")
            return True, message, data

        return self.consolidate_error(key='image', message=None, data=data, stamp_local=stamp_local, stamp_global=stamp_global)

    def generate_scene_description(self, data, settings, is_worker, stamp_global, choices):
        if settings['scene_description']['skip']:
            return True, "Skipping scene description.", data

        # log
        log_message = "Generating scene description." if choices == 1 else f"Generating '{choices}' scene descriptions."
        if settings['scene_description']['message_process']:
            self._logger.info(log_message)
        else:
            self._logger.debug(log_message)

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
            if choices > 1:
                success, message, completions = self.split_completion(completion=completion, choices=choices, name="scene description")
                if not success:
                    success, message, data = self.consolidate_error(key='scene_description', message=message, data=data, stamp_local=stamp_local, stamp_global=stamp_global)
                    return success, message, [copy.deepcopy(data) for _ in range(choices)]
                del data['scene_description']['completion']
                branches = [None] * choices
                num_success = 0
                for k, choice_completion in enumerate(completions):
                    branch = copy.deepcopy(data)
                    branch['scene_description']['completion'] = choice_completion
                    success, message, branch = self.parse_completion(completion=choice_completion, required_keys=required_keys, reserved_keys=reserved_keys, data=branch, data_key='scene_description', name="scene description")
                    if success:
                        success, message, branch = self.parse_scene_description(data=branch, stamp_local=stamp_local)
                    if success:
                        num_success += 1
                    else:
                        _, _, branch = self.consolidate_error(key='scene_description', message=message, data=branch, stamp_local=stamp_local, stamp_global=stamp_global)
                    branches[k] = branch
                # log
                failures = choices - num_success
                if failures == 0:
                    log_message = f"Generated '{choices}' scene descriptions in '{branches[-1]['scene_description']['duration']:.3f}s'."
                else:
                    log_message = f"Failed to generate '{failures}' of '{choices}' scene descriptions in '{branches[-1]['scene_description']['duration']:.3f}s'."
                if settings['scene_description']['message_process']:
                    if settings['scene_description']['message_results'] and num_success > 0:
                        self._logger.info(f"{log_message[:-1]}: '\n{json.dumps([branch['scene_description']['data'] for branch in branches if branch['scene_description']['success']], indent=4)}'")
                    else:
                        self._logger.info(log_message)
                else:
                    self._logger.debug(log_message)
                if failures > 0:
                    message = log_message
                return failures == 0, message, branches
            success, message, data = self.parse_completion(completion=completion, required_keys=required_keys, reserved_keys=reserved_keys, data=data, data_key='scene_description', name="scene description")
            if not success:
                return self.consolidate_error(key='scene_description', message=message, data=data, stamp_local=stamp_local, stamp_global=stamp_global)
            success, message, data = self.parse_scene_description(data=data, stamp_local=stamp_local)
            if success:
                # log
                if settings['scene_description']['message_process']:
                    if settings['scene_description']['message_results']:
                        self._logger.info(f"Generated scene description in '{data['scene_description']['duration']:.3f}s': '{data['scene_description']['data']}'")
                    else:
                        self._logger.info(f"Generated scene description in '{data['scene_description']['duration']:.3f}s'.")
                else:
                    self._logger.debug(f"Generated scene description in '{data['scene_description']['duration']:.3f}s'.")
                return True, message, data
            return self.consolidate_error(key='scene_description', message=message, data=data, stamp_local=stamp_local, stamp_global=stamp_global)

        if choices > 1:
            success, message, data = self.consolidate_error(key='scene_description', message=None, data=data, stamp_local=stamp_local, stamp_global=stamp_global)
            return success, message, [copy.deepcopy(data) for _ in range(choices)]

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
            self._logger.debug(data[data_key]['logs'][-1])

        data[data_key]['logs'].append(f"Validated {name} completion.")

        return True, data[data_key]['logs'][-1], data

    def split_completion(self, completion, choices, name):
        if not isinstance(completion, dict):
            return False, f"Expected {name} completion to be of type 'dict' but got '{type(completion).__name__}'.", None

        if 'choices' not in completion:
            return False, f"Expected {name} completion with '{choices}' choices to contain the key 'choices'.", None
        if not isinstance(completion['choices'], list):
            return False, f"Expected value of key 'choices' in {name} completion to be of type 'list' but got '{type(completion['choices']).__name__}'.", None
        if len(completion['choices']) != choices:
            return False, f"Expected value of key 'choices' in {name} completion to be a list of length '{choices}' but got '{len(completion['choices'])}'.", None

        completions = [None] * choices
        for i, choice in enumerate(completion['choices']):
            if not isinstance(choice, dict):
                return False, f"Expected choice '{i}' in {name} completion to be of type 'dict' but got '{type(choice).__name__}'.", None
            item = copy.deepcopy({key: value for key, value in completion.items() if key != 'choices'})
            for key, value in choice.items():
                if key == 'logs' and isinstance(value, list) and isinstance(item.get('logs'), list):
                    item['logs'] = item['logs'] + copy.deepcopy(value)
                else:
                    item[key] = copy.deepcopy(value)
            completions[i] = item

        return True, f"Split {name} completion into '{choices}' completions.", completions

    def parse_scene_description(self, data, stamp_local):
        description = copy.deepcopy(data['scene_description']['raw'])

        if not isinstance(description, str):
            return False, f"Expected scene description to be of type 'str' but got '{type(description).__name__}'.", data
        if len(description) == 0:
            return False, "Expected scene description to be a non-empty string.", data

        if data['scene_description']['raw'] == description:
            del data['scene_description']['raw']
            data['scene_description']['logs'].append("Removed raw data identical to validated data.")
            self._logger.debug(f"{data['scene_description']['logs'][-1]} (scene description)")
        data['scene_description']['data'] = description
        data['scene_description']['logs'].append("Validated scene description.")
        if stamp_local is not None:
            data['scene_description']['duration'] = time.perf_counter() - stamp_local

        return True, data['scene_description']['logs'][-1], data

    def generate_structured_description(self, data, settings, is_worker, stamp_global, choices):
        if settings['structured_description']['skip']:
            return True, "Skipping structured description.", data

        # log
        log_message = "Generating structured description." if choices == 1 else f"Generating '{choices}' structured descriptions."
        if settings['structured_description']['message_process']:
            self._logger.info(log_message)
        else:
            self._logger.debug(log_message)

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
            if choices > 1:
                success, message, completions = self.split_completion(completion=completion, choices=choices, name="structured description")
                if not success:
                    success, message, data = self.consolidate_error(key='structured_description', message=message, data=data, stamp_local=stamp_local, stamp_global=stamp_global)
                    return success, message, [copy.deepcopy(data) for _ in range(choices)]
                del data['structured_description']['completion']
                branches = [None] * choices
                num_success = 0
                for k, choice_completion in enumerate(completions):
                    branch = copy.deepcopy(data)
                    branch['structured_description']['completion'] = choice_completion
                    success, message, branch = self.parse_completion(completion=choice_completion, required_keys=required_keys, reserved_keys=reserved_keys, data=branch, data_key='structured_description', name="structured description")
                    if success:
                        success, message, branch = self.parse_structured_description(settings=settings, data=branch, stamp_local=stamp_local)
                    if success:
                        num_success += 1
                    else:
                        _, _, branch = self.consolidate_error(key='structured_description', message=message, data=branch, stamp_local=stamp_local, stamp_global=stamp_global)
                    branches[k] = branch
                # log
                failures = choices - num_success
                if failures == 0:
                    log_message = f"Generated '{choices}' structured descriptions in '{branches[-1]['structured_description']['duration']:.3f}s'."
                else:
                    log_message = f"Failed to generate '{failures}' of '{choices}' structured descriptions in '{branches[-1]['structured_description']['duration']:.3f}s'."
                if settings['structured_description']['message_process']:
                    if settings['structured_description']['message_results'] and num_success > 0:
                        self._logger.info(f"{log_message[:-1]}: '\n{json.dumps([branch['structured_description']['data'] for branch in branches if branch['structured_description']['success']], indent=4)}'")
                    else:
                        self._logger.info(log_message)
                else:
                    self._logger.debug(log_message)
                if failures > 0:
                    message = log_message
                return failures == 0, message, branches
            success, message, data = self.parse_completion(completion=completion, required_keys=required_keys, reserved_keys=reserved_keys, data=data, data_key='structured_description', name="structured description")
            if not success:
                return self.consolidate_error(key='structured_description', message=message, data=data, stamp_local=stamp_local, stamp_global=stamp_global)
            success, message, data = self.parse_structured_description(settings=settings, data=data, stamp_local=stamp_local)
            if success:
                # log
                num_described = len(data['structured_description']['data'])
                if settings['structured_description']['message_process']:
                    if settings['structured_description']['message_results'] and num_described > 0:
                        self._logger.info(f"Generated structured description with '{num_described}' object{'' if num_described == 1 else 's'}in '{data['structured_description']['duration']:.3f}s': '\n{json.dumps(data['structured_description']['data'], indent=4)}'")
                    else:
                        self._logger.info(f"Generated structured description with '{num_described}' object{'' if num_described == 1 else 's'} in '{data['structured_description']['duration']:.3f}s'.")
                else:
                    self._logger.debug(f"Generated structured description with '{num_described}' object{'' if num_described == 1 else 's'} in '{data['structured_description']['duration']:.3f}s'.")
                return True, message, data
            # TODO trigger correction using message
            return self.consolidate_error(key='structured_description', message=message, data=data, stamp_local=stamp_local, stamp_global=stamp_global)

        if choices > 1:
            success, message, data = self.consolidate_error(key='structured_description', message=None, data=data, stamp_local=stamp_local, stamp_global=stamp_global)
            return success, message, [copy.deepcopy(data) for _ in range(choices)]

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
                            options.append(description[i:j + 1])
                    options.sort(key=len, reverse=True)
                    for option in options:
                        try:
                            description = json.loads(option)
                        except json.JSONDecodeError:
                            pass
                        else:
                            data['structured_description']['logs'].append("Parsed structured description from data of type 'str' as JSON.")
                            self._logger.debug(data['structured_description']['logs'][-1])
                            break
                    else:
                        return False, "Failed to extract JSON from structured description received as string.", data

        # extract structured description nested in dictionary
        if isinstance(description, dict) and len(description) == 1:
            candidate = description[list(description.keys())[0]]
            if isinstance(candidate, list):
                description = candidate
                data['structured_description']['logs'].append("Extracted structured description from data of type 'dict' containing '1' item with value of type 'list'.")
                self._logger.warn(data['structured_description']['logs'][-1])

        # structured description must be list
        if not isinstance(description, list):
            try:
                return False, f"Expected structured description to be of type 'list' but got '{type(description).__name__}': {description}", data
            except Exception:
                return False, f"Expected structured description to be of type 'list' but got '{type(description).__name__}'.", data

        # obtain image dimensions if required
        dimensions = None
        for key in ["point_xy[int1000]", "point_yx[int1000]", "box_xyxy[int1000]", "box_yxyx[int1000]"]:
            for setting in ['keys_required_types', 'keys_optional_types']:
                if key in settings['structured_description'][setting]:
                    if 'width' in data['image'] and 'height' in data['image']:
                        break
                    success, message, dimensions = get_image_dimensions(image=data['image']['data'], logger=self._logger)
                    if success:
                        data['image']['width'] = dimensions[0]
                        data['image']['height'] = dimensions[1]
                        break
                    raise UnrecoverableError(message)
            if dimensions is not None:
                break

        # validate objects

        valid_description = []

        for i, obj in enumerate(description):
            # object must be dictionary
            if not isinstance(obj, dict):
                return False, f"Expected object '{i}' in structured description to be of type 'dict' but got '{type(obj).__name__}'.", data

            valid_obj = {}
            keys_left = list(obj.keys())
            for key in keys_left:
                if not isinstance(key, str):
                    return False, f"Expected all keys in object '{i}' of structured description to be of type 'str' but got key '{key}' of type '{type(key).__name__}'.", data

            # object can feature required keys
            for j, target_key in enumerate(settings['structured_description']['keys_required'] + settings['structured_description']['keys_optional']):
                target_key_norm = re.sub(r"[.,;:_\-\s]", "", target_key).lower().strip()
                for source_key in copy.deepcopy(keys_left):
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

                        elif expected_type == "likert5":
                            if isinstance(val, int) and not isinstance(val, bool) and val in [1, 2, 3, 4, 5]:
                                valid_obj[target_key] = val
                                is_valid = True

                        elif expected_type == "likert7":
                            if isinstance(val, int) and not isinstance(val, bool) and val in [1, 2, 3, 4, 5, 6, 7]:
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

                        elif expected_type in ["point_xy[int]", "point_yx[int]"]:
                            if isinstance(val, list) and len(val) == 1:
                                val = val[0]
                            if isinstance(val, list) and len(val) == 2 and all(isinstance(x, int) and not isinstance(x, bool) and x >= 0 for x in val):
                                valid_obj[target_key] = val
                                is_valid = True

                        elif expected_type in ["point_xy[int1000]", "point_yx[int1000]"]:
                            if isinstance(val, list) and len(val) == 1:
                                val = val[0]
                            if isinstance(val, list) and len(val) == 2 and all(isinstance(x, int) and not isinstance(x, bool) and x >= 0 and x <= 1000 for x in val):
                                if expected_type == "point_xy[int1000]":
                                    x = min(max(int(round(val[0] / 1000 * data['image']['width'])), 0), data['image']['width'])
                                    y = min(max(int(round(val[1] / 1000 * data['image']['height'])), 0), data['image']['height'])
                                    valid_obj[target_key] = [x, y]
                                else:
                                    x = min(max(int(round(val[1] / 1000 * data['image']['width'])), 0), data['image']['width'])
                                    y = min(max(int(round(val[0] / 1000 * data['image']['height'])), 0), data['image']['height'])
                                    valid_obj[target_key] = [y, x]
                                data['structured_description']['logs'].append(f"Unnormalized point {val} of object '{i}' to '{valid_obj[target_key]}' according to image width '{data['image']['width']}' and height '{data['image']['height']}'.")
                                self._logger.debug(data['structured_description']['logs'][-1])
                                is_valid = True

                        elif expected_type in ["box_xyxy[int]", "box_yxyx[int]"]:
                            if isinstance(val, list) and len(val) == 1:
                                val = val[0]
                            if isinstance(val, list) and len(val) == 4 and all(isinstance(x, int) and not isinstance(x, bool) and x >= 0 for x in val) and val[2] > val[0] and val[3] > val[1]:
                                valid_obj[target_key] = val
                                is_valid = True

                        elif expected_type in ["box_xyxy[int1000]", "box_yxyx[int1000]"]:
                            if isinstance(val, list) and len(val) == 1:
                                val = val[0]
                            if isinstance(val, list) and len(val) == 4 and all(isinstance(x, int) and not isinstance(x, bool) and x >= 0 and x <= 1000 for x in val) and val[2] > val[0] and val[3] > val[1]:
                                if expected_type == "box_xyxy[int1000]":
                                    x_min = min(max(int(round(val[0] / 1000 * data['image']['width'])), 0), data['image']['width'])
                                    y_min = min(max(int(round(val[1] / 1000 * data['image']['height'])), 0), data['image']['height'])
                                    x_max = min(max(int(round(val[2] / 1000 * data['image']['width'])), 0), data['image']['width'])
                                    y_max = min(max(int(round(val[3] / 1000 * data['image']['height'])), 0), data['image']['height'])
                                else:
                                    x_min = min(max(int(round(val[1] / 1000 * data['image']['width'])), 0), data['image']['width'])
                                    y_min = min(max(int(round(val[0] / 1000 * data['image']['height'])), 0), data['image']['height'])
                                    x_max = min(max(int(round(val[3] / 1000 * data['image']['width'])), 0), data['image']['width'])
                                    y_max = min(max(int(round(val[2] / 1000 * data['image']['height'])), 0), data['image']['height'])
                                if x_min < x_max and y_min < y_max:
                                    if expected_type == "box_xyxy[int1000]":
                                        valid_obj[target_key] = [x_min, y_min, x_max, y_max]
                                    else:
                                        valid_obj[target_key] = [y_min, x_min, y_max, x_max]
                                    data['structured_description']['logs'].append(f"Unnormalized bounding box {val} of object '{i}' to '{valid_obj[target_key]}' according to image width '{data['image']['width']}' and height '{data['image']['height']}'.")
                                    self._logger.debug(data['structured_description']['logs'][-1])
                                    is_valid = True

                        else:
                            raise NotImplementedError(f"Unknown key type '{expected_type}'.")

                        if not is_valid:
                            is_required = j < len(settings['structured_description']['keys_required'])
                            req_opt_str = "required" if is_required else "optional"
                            err_msg = f"Expected format of {req_opt_str} key '{target_key}' in object '{i}' of structured description to match type '{expected_type}' but got invalid value '{val}' of type '{type(val).__name__}'."

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
            self._logger.debug(f"{data['structured_description']['logs'][-1]} (structured description)")
        data['structured_description']['data'] = valid_description
        data['structured_description']['logs'].append("Validated structured description.")
        if stamp_local is not None:
            data['structured_description']['duration'] = time.perf_counter() - stamp_local

        return True, data['structured_description']['logs'][-1], data

    def generate_detection(self, data, settings, is_worker, stamp_global):
        if settings['detection']['skip']:
            return True, "Skipping detection.", data

        # log
        num_described = len(data['structured_description']['data'])
        if settings['detection']['extract_from_description']:
            self._logger.debug(f"Extracting '{num_described}' detection{'' if num_described == 1 else 's'} from structured description.")
        elif settings['detection']['message_process']:
            self._logger.info(f"Detecting '{num_described}' described object{'' if num_described == 1 else 's'}.")
        else:
            self._logger.debug(f"Detecting '{num_described}' described object{'' if num_described == 1 else 's'}.")

        stamp_local = time.perf_counter()
        data['detection'] = {'stamp': datetime.datetime.now().isoformat()}
        if not is_worker:
            data['detection']['settings'] = copy.deepcopy(settings['detection'])

        if len(data['structured_description']['data']) == 0:
            data['detection']['success'] = True
            data['detection']['logs'] = ["Structured description was empty."]
            prompts = []
            detection = []
        elif settings['detection']['extract_from_description']:
            for i, item in enumerate(settings['structured_description']['keys_required_types']):
                if "box" in item:
                    bbox_key = settings['structured_description']['keys_required'][i]
                    bbox_type = item
                    break
            data['detection']['success'] = True
            data['detection']['logs'] = ["Extracted from structured description."]
            prompts = []
            detection = []
            for item in data['structured_description']['data']:
                prompts.append(item[settings['detection']['prompt_key']])
                detection.append({
                    'box_xyxy': [int(value) for value in item[bbox_key]],
                    'prompt': prompts[-1]
                })
                if "yxyx" in bbox_type:
                    detection[-1]['box_xyxy'] = [detection[-1]['box_xyxy'][1], detection[-1]['box_xyxy'][0], detection[-1]['box_xyxy'][3], detection[-1]['box_xyxy'][2]]
        else:
            detector = MmGroundingDino(settings=settings['detection']['mmgroundingdino'])
            prompts = [item[settings['detection']['prompt_key']] for item in data['structured_description']['data']]
            data['detection']['success'], message, detection = detector.get_detections(image=data['image']['data'], prompts=prompts)
            data['detection']['logs'] = [message]
        if data['detection']['success']:
            data['detection']['raw'] = detection
            success, message, data = self.parse_detection(data=data, settings=settings, prompts=prompts, stamp_local=stamp_local)
            if success:
                # log
                num_detected = len(data['detection']['data'])
                if settings['detection']['extract_from_description']:
                    self._logger.debug(f"Extracted '{num_detected}' detection{'' if num_detected == 1 else 's'} from '{num_described}' described object{'' if num_described == 1 else 's'} in '{data['detection']['duration']:.3f}s'.")
                elif settings['detection']['message_process']:
                    if settings['detection']['message_results'] and num_detected > 0:
                        self._logger.info(f"Obtained '{num_detected}' detection{'' if num_detected == 1 else 's'} from '{num_described}' described object{'' if num_described == 1 else 's'} in '{data['detection']['duration']:.3f}s': '\n{json.dumps(data['detection']['data'], indent=4)}'")
                    else:
                        self._logger.info(f"Obtained '{num_detected}' detection{'' if num_detected == 1 else 's'} from '{num_described}' described object{'' if num_described == 1 else 's'} in '{data['detection']['duration']:.3f}s'.")
                else:
                    self._logger.debug(f"Obtained '{num_detected}' detection{'' if num_detected == 1 else 's'} from '{num_described}' described object{'' if num_described == 1 else 's'} in '{data['detection']['duration']:.3f}s'.")
                return True, message, data
            return self.consolidate_error(key='detection', message=message, data=data, stamp_local=stamp_local, stamp_global=stamp_global)

        return self.consolidate_error(key='detection', message=None, data=data, stamp_local=stamp_local, stamp_global=stamp_global)

    def parse_detection(self, data, settings, prompts, stamp_local):
        detection = copy.deepcopy(data['detection']['raw'])

        if not isinstance(detection, list):
            return False, f"Expected detection to be of type 'list' but got '{type(detection).__name__}'.", data
        required_format = {'box_xyxy': list, 'confidence': float, 'prompt': str}
        required_keys = {'box_xyxy', 'prompt'}
        for i, item in enumerate(detection):
            if not isinstance(item, dict):
                return False, f"Expected detection '{i}' to be of type 'dict' but got '{type(item).__name__}'.", data
            detection_keys = set(item.keys())
            if not required_keys.issubset(detection_keys):
                return False, f"Expected detection '{i}' to contain the keys {required_keys} but got {detection_keys}.", data
            for key in required_keys:
                if not isinstance(item[key], required_format[key]):
                    return False, f"Expected key '{key}' of detection '{i}' to be of type '{required_format[key].__name__}' but got '{type(item[key]).__name__}'.", data
            if 'confidence' in item and not isinstance(item['confidence'], float):
                return False, f"Expected value of optional key 'confidence' of detection '{i}' to be of type 'float' but got '{type(item['confidence']).__name__}'.", data
            if len(item['box_xyxy']) != 4:
                return False, f"Expected value of key 'box_xyxy' in detection '{i}' to be a list of length '4' but got '{len(item['box_xyxy'])}'.", data
            for j, sub_item in enumerate(item['box_xyxy']):
                if not isinstance(sub_item, int) or isinstance(sub_item, bool):
                    return False, f"Expected element '{j}' in value of key 'box_xyxy' in detection '{i}' to be of type 'int' but got '{type(sub_item).__name__}'.", data
                if sub_item < 0:
                    return False, f"Expected element '{j}' in value of key 'box_xyxy' in detection '{i}' to be a non-negative integer but got '{sub_item}'.", data
            if item['box_xyxy'][2] <= item['box_xyxy'][0] or item['box_xyxy'][3] <= item['box_xyxy'][1]:
                return False, f"Expected value of key 'box_xyxy' in detection '{i}' to be a valid bounding box (x0, y0, x1, y1) but got '{item['box_xyxy']}'.", data

        prompt_counts_initial = count_duplicates(iterable=prompts, include_unique=True)

        if settings['detection']['allow_excessive']:
            detection_filtered = []
            for item in detection:
                prompt = item['prompt']
                if prompt in prompt_counts_initial:
                    detection_filtered.append(item)
                else:
                    prompt_str = prompt.replace("\n", "\\n")
                    data['detection']['logs'].append(f"Discarded detection with unexpected prompt '{prompt_str}'.")
                    self._logger.warn(data['detection']['logs'][-1])
        else:
            items_by_prompt = {prompt: [] for prompt in prompt_counts_initial}
            for i, item in enumerate(detection):
                prompt = item['prompt']
                if prompt in prompt_counts_initial:
                    items_by_prompt[prompt].append((i, item))
                else:
                    prompt_str = prompt.replace("\n", "\\n")
                    data['detection']['logs'].append(f"Discarded detection with unexpected prompt '{prompt_str}'.")
                    self._logger.warn(data['detection']['logs'][-1])
            keep_indices = set()
            for prompt, items in items_by_prompt.items():
                requested_count = prompt_counts_initial[prompt]
                selected = sorted(items, key=lambda pair: pair[1].get('confidence', float('-inf')), reverse=True)[:requested_count]

                selected_indices = {i for i, _ in selected}
                keep_indices.update(selected_indices)
                if len(items) > requested_count:
                    prompt_str = prompt.replace("\n", "\\n")
                    for i, item in items:
                        if i not in selected_indices:
                            data['detection']['logs'].append(f"Discarded excessive detection with prompt '{prompt_str}'.")
                            self._logger.warn(data['detection']['logs'][-1])
            detection_filtered = [item for i, item in enumerate(detection) if i in keep_indices]

        detection = detection_filtered
        detection_prompts = [item['prompt'] for item in detection]
        prompt_counts = copy.deepcopy(prompt_counts_initial)
        detection_counts = count_duplicates(iterable=detection_prompts, include_unique=True)

        for prompt in prompt_counts_initial:
            prompt_counts[prompt] -= detection_counts.get(prompt, 0)
            if prompt_counts[prompt] > 0:
                prompt_str = prompt.replace("\n", "\\n")
                message = f"Failed to detect '{prompt_counts[prompt]}' instance{'' if prompt_counts[prompt] == 1 else 's'} of prompt '{prompt_str}' requested '{prompt_counts_initial[prompt]}' time{'' if prompt_counts_initial[prompt] == 1 else 's'}."
                if not settings['detection']['allow_incomplete']:
                    return False, message, data
                data['detection']['logs'].append(message)
                self._logger.warn(data['detection']['logs'][-1])
            if prompt_counts[prompt] < 0:
                prompt_str = prompt.replace("\n", "\\n")
                data['detection']['logs'].append(f"Over-detected '{-prompt_counts[prompt]}' instance{'' if prompt_counts[prompt] == -1 else 's'} of prompt '{prompt_str}' requested '{prompt_counts_initial[prompt]}' time{'' if prompt_counts_initial[prompt] == 1 else 's'}.")
                self._logger.debug(data['detection']['logs'][-1])

        if data['detection']['raw'] == detection:
            del data['detection']['raw']
            data['detection']['logs'].append("Removed raw data identical to validated data.")
            self._logger.debug(f"{data['detection']['logs'][-1]} (detection)")

        data['detection']['data'] = detection
        data['detection']['logs'].append("Validated detection.")

        if stamp_local is not None:
            data['detection']['duration'] = time.perf_counter() - stamp_local

        return True, data['detection']['logs'][-1], data

    def generate_segmentation(self, data, settings, is_worker, stamp_global):
        if settings['segmentation']['skip']:
            return True, "Skipping segmentation.", data

        # log
        num_detected = len(data['detection']['data'])
        if settings['segmentation']['message_process']:
            self._logger.info(f"Segmenting '{num_detected}' detected object{'' if num_detected == 1 else 's'}.")
        else:
            self._logger.debug(f"Segmenting '{num_detected}' detected object{'' if num_detected == 1 else 's'}.")

        stamp_local = time.perf_counter()
        data['segmentation'] = {'stamp': datetime.datetime.now().isoformat()}
        if not is_worker:
            data['segmentation']['settings'] = copy.deepcopy(settings['segmentation'])
        if len(data['detection']['data']) == 0:
            data['segmentation']['success'] = True
            data['segmentation']['logs'] = ["There were no detections."]
            segmentation = []
        else:
            prompts = [{'object_id': i, 'bbox': item['box_xyxy']} for i, item in enumerate(data['detection']['data'])]
            segmenter = Sam2Realtime(settings=settings['segmentation']['sam2_realtime'])
            data['segmentation']['success'], message, segmentation = segmenter.get_response(image=data['image']['data'], prompts=prompts)
            data['segmentation']['logs'] = [message]
            if data['segmentation']['success']:
                if settings['segmentation']['track']:
                    data['segmentation']['duration_init'] = time.perf_counter() - stamp_local
                    data['segmentation']['success'], message, segmentation = segmenter.get_response(image=data['image']['data'])
                    data['segmentation']['logs'].append(message)
        if data['segmentation']['success']:
            data['segmentation']['raw'] = segmentation
            success, message, data = self.parse_segmentation(data=data, settings=settings, stamp_local=stamp_local)
            if success:
                # log
                num_segmented = len(data['segmentation']['data'])
                if settings['segmentation']['message_process']:
                    if settings['segmentation']['message_results'] and num_segmented > 0:
                        self._logger.info(f"Segmented '{num_segmented}' detected object{'' if num_segmented == 1 else 's'} in '{data['segmentation']['duration']:.3f}s': '\n{json.dumps(data['segmentation']['data'], indent=4)}'")
                    else:
                        self._logger.info(f"Segmented '{num_segmented}' detected object{'' if num_segmented == 1 else 's'} in '{data['segmentation']['duration']:.3f}s'.")
                else:
                    self._logger.debug(f"Segmented '{num_segmented}' detected object{'' if num_segmented == 1 else 's'} in '{data['segmentation']['duration']:.3f}s'.")
                return True, message, data
            return self.consolidate_error(key='segmentation', message=message, data=data, stamp_local=stamp_local, stamp_global=stamp_global)

        return self.consolidate_error(key='segmentation', message=None, data=data, stamp_local=stamp_local, stamp_global=stamp_global)

    def parse_segmentation(self, data, settings, stamp_local):
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
            if isinstance(item['track_id'], bool):
                return False, f"Expected key 'track_id' of segmentation '{i}' to be of type 'int' but got 'bool'.", data
            if len(item['box_xyxy']) != 4:
                return False, f"Expected value of key 'box_xyxy' in segmentation '{i}' to be a list of length '4' but got '{len(item['box_xyxy'])}'.", data
            for j, sub_item in enumerate(item['box_xyxy']):
                if not isinstance(sub_item, int) or isinstance(sub_item, bool):
                    return False, f"Expected element '{j}' in value of key 'box_xyxy' in segmentation '{i}' to be of type 'int' but got '{type(sub_item).__name__}'.", data
                if sub_item < 0:
                    return False, f"Expected element '{j}' in value of key 'box_xyxy' in segmentation '{i}' to be a non-negative integer but got '{sub_item}'.", data
            if item['box_xyxy'][2] <= item['box_xyxy'][0] or item['box_xyxy'][3] <= item['box_xyxy'][1]:
                return False, f"Expected value of key 'box_xyxy' in segmentation '{i}' to be a valid bounding box (x0, y0, x1, y1) but got '{item['box_xyxy']}'.", data
            track_ids_grouped.setdefault(item['track_id'], []).append(item)

        segmentation_filtered = []
        num_detections = len(data['detection']['data'])

        for track_id, items in track_ids_grouped.items():
            if track_id < 0 or track_id >= num_detections:
                for item in items:
                    data['segmentation']['logs'].append(f"Discarded segmentation with unexpected track ID '{track_id}'.")
                    self._logger.warn(data['segmentation']['logs'][-1])

        for i in range(num_detections):
            if i not in track_ids_grouped:
                prompt_str = data['detection']['data'][i]['prompt'].replace("\n", "\\n")
                message = f"Failed to segment detection '{i}' for prompt '{prompt_str}'."
                if not settings['segmentation']['allow_incomplete']:
                    return False, message, data
                data['segmentation']['logs'].append(message)
                self._logger.warn(data['segmentation']['logs'][-1])
                continue
            items = track_ids_grouped[i]
            if len(items) > 1:
                if settings['segmentation']['allow_excessive']:
                    segmentation_filtered.extend(items)
                    excess = len(items) - 1
                    prompt_str = data['detection']['data'][i]['prompt'].replace("\n", "\\n")
                    data['segmentation']['logs'].append(f"Over-segmented '{excess}' instance{'' if excess == 1 else 's'} of detection '{i}' for prompt '{prompt_str}'.")
                    self._logger.debug(data['segmentation']['logs'][-1])
                else:
                    selected = max(items, key=lambda item: (item['box_xyxy'][2] - item['box_xyxy'][0]) * (item['box_xyxy'][3] - item['box_xyxy'][1]))
                    for item in items:
                        if item is not selected:
                            prompt_str = data['detection']['data'][i]['prompt'].replace("\n", "\\n")
                            data['segmentation']['logs'].append(f"Discarded excessive segmentation of detection '{i}' for prompt '{prompt_str}'.")
                            self._logger.warn(data['segmentation']['logs'][-1])
                    segmentation_filtered.append(selected)
            else:
                segmentation_filtered.append(items[0])

        if data['segmentation']['raw'] == segmentation_filtered:
            del data['segmentation']['raw']
            data['segmentation']['logs'].append("Removed raw data identical to validated data.")
            self._logger.debug(f"{data['segmentation']['logs'][-1]} (segmentation)")
        data['segmentation']['data'] = segmentation_filtered
        data['segmentation']['logs'].append("Validated segmentation.")
        if stamp_local is not None:
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

    def finalize_result(self, data, settings, is_worker, stamp_global):
        if not settings['include_image']:
            del data['image']['data']

        if not is_worker and 'structured_description' in data and 'settings' in data['structured_description']:
            for key in ['keys_required_types', 'keys_optional_types']:
                for i, _type in enumerate(settings['structured_description'][key]):
                    data['structured_description']['settings'][key][i] = _type.replace("[int1000]", "[int]")

        data['run']['success'] = True
        duration = time.perf_counter() - stamp_global

        suffix = "" if data['image']['path'] is None else f" for image '{data['image']['path']}'"
        if 'detection' in data:
            num_grounded = len(data['detection']['data'])
            data['run']['message'] = f"Generated '{num_grounded}' object grounding{'' if num_grounded == 1 else 's'}{suffix} in '{duration:.3f}s'."
            if settings['message_results'] and num_grounded > 0:
                data['run']['message'] = f"{data['run']['message'][:-1]}: '\n{json.dumps(data['structured_description']['data'], indent=4)}'"
        elif 'structured_description' in data:
            num_grounded = len(data['structured_description']['data'])
            data['run']['message'] = f"Generated '{num_grounded}' object description{'' if num_grounded == 1 else 's'}{suffix} in '{duration:.3f}s'."
            if settings['message_results'] and num_grounded > 0:
                data['run']['message'] = f"{data['run']['message'][:-1]}: '\n{json.dumps(data['structured_description']['data'], indent=4)}'"
        else:
            data['run']['message'] = f"Generated image description{suffix} in '{duration:.3f}s'."
            if settings['message_results']:
                data['run']['message'] = f"{data['run']['message'][:-1]}: {data['scene_description']['data']}"

        data['run']['duration'] = duration

        return True, data['run']['message'], data
