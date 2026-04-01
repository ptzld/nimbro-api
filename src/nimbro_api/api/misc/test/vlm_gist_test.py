import os
import json
import datetime

import nimbro_api
from nimbro_api.utility.io import parse_image_b64, decode_b64
from nimbro_api.utility.misc import assert_type_value, assert_log, assert_keys, format_obj
from ..client.vlm_gist import VlmGist

def assert_result(success, message, result, settings):
    assert_type_value(obj=result, type_or_value=dict, name="result")
    assert_keys(obj=result, keys=['run', 'image', 'scene_description', 'structured_description', 'detection', 'segmentation', 'batch'], mode="whitelist", name="result")
    assert_log('run' in result, message="Expected result to contain key 'run'.")
    assert_type_value(obj=result['run'], type_or_value=dict, name="key 'run' in result")
    assert_log('type' in result['run'], message="Expected result to contain key 'type' in key 'run'.")
    assert_type_value(obj=result['run']['type'], type_or_value=["normal", "batch", "worker"], name="key 'type' in key 'run' in result")
    assert_keys(obj=result['run'], keys=['stamp', 'type', 'success', 'settings', 'message', 'duration', 'scene_description', 'structured_description', 'detection'], mode="whitelist", name="key 'run' in result")
    if result['run']['type'] == "worker":
        assert_keys(obj=result['run'], keys=['stamp', 'type', 'success', 'message', 'duration'], mode="required", name="key 'run' in result")
        assert_keys(obj=result['run'], keys=['settings'], mode="blacklist", name="key 'run' in result")
    else:
        assert_keys(obj=result['run'], keys=['stamp', 'type', 'settings', 'success', 'message', 'duration'], mode="required", name="key 'run' in result")
    assert_type_value(obj=result['run']['stamp'], type_or_value=str, name="key 'stamp' in key 'run' in result")
    try:
        datetime.datetime.fromisoformat(result['run']['stamp'])
    except Exception as e:
        assert_log(expression=False, message=f"Expected value of key 'stamp' in key 'run' in result to be ISO 8601: {repr(e)}")
    assert_type_value(obj=result['run'].get('settings', {}), type_or_value=dict, name="key 'settings' in key 'run' in result")
    assert_type_value(obj=result['run']['success'], type_or_value=bool, name="key 'success' in key 'run' in result")
    assert_log(expression=result['run']['success'] == success, message=f"Expected key 'success' in key 'run' in result to be '{success}' but got '{result['run']['success']}'.")
    assert_type_value(obj=result['run']['message'], type_or_value=str, name="key 'message' in key 'run' in result")
    assert_log(expression=result['run']['message'] == message, message=f"Expected key 'message' in key 'run' in result to be '{message}' but got '{result['run']['message']}'.")
    assert_type_value(obj=result['run']['duration'], type_or_value=float, name="key 'duration' in key 'run' in result")
    assert_log(expression=result['run']['duration'] >= 0, message=f"Expected key 'duration' in key 'run' in result to be non-negative but got '{result['run']['duration']}'.")
    if result['run']['type'] == "batch":
        assert_keys(obj=result, keys=['run', 'batch'], mode="match", name="result")
        assert_log(expression=result['run']['settings'] == settings, message=f"Expected full settings in key 'run' in result for batch type but got {format_obj(result['run']['settings'])} instead of {format_obj(settings)}.")
        assert_type_value(obj=result['batch'], type_or_value=list, name="key 'batch' in result")
        assert_log(expression=len(result['batch']) > 0, message=f"Expected key 'batch' in result to be non-empty but got length '{len(result['batch'])}'.")
        for i, item in enumerate(result['batch']):
            assert_result(success=item['run']['success'], message=item['run']['message'], result=item, settings=settings)
        return
    if result['run']['type'] == "normal":
        assert_keys(obj=result['run']['settings'], keys=['logger_severity', 'logger_name', 'message_results', 'include_image', 'retry'], mode="match", name="key 'settings' in key 'run' in result")
        for key in ['logger_severity', 'logger_name', 'message_results', 'include_image', 'retry']:
            assert_log(expression=result['run']['settings'][key] == settings[key], message=f"Expected setting '{key}' in key 'run' in result to be '{settings[key]}' but got '{result['run']['settings'][key]}'.")
    for key in ['image', 'scene_description', 'structured_description', 'detection', 'segmentation']:
        if key not in result:
            if key != 'image':
                assert_log(expression=settings[key]['skip'], message=f"Expected key '{key}' in result because setting 'skip' in '{key}' is 'False'.")
            continue
        assert_type_value(obj=result[key], type_or_value=dict, name=f"key '{key}' in result")
        assert_keys(obj=result[key], keys=['success', 'logs'], mode="required", name=f"key '{key}' in result")
        assert_type_value(obj=result[key]['success'], type_or_value=bool, name=f"key 'success' in key '{key}' in result")
        assert_type_value(obj=result[key]['logs'], type_or_value=list, name=f"key 'logs' in key '{key}' in result")
        for i, log in enumerate(result[key]['logs']):
            assert_type_value(obj=log, type_or_value=str, name=f"log '{i}' in key '{key}' in result")
            assert_log(expression=len(log) > 0, message=f"Expected log '{i}' in key '{key}' in result to be a non-empty string.")
        if 'stamp' in result[key]:
            assert_type_value(obj=result[key]['stamp'], type_or_value=str, name=f"key 'stamp' in key '{key}' in result")
            try:
                datetime.datetime.fromisoformat(result[key]['stamp'])
            except Exception as e:
                assert_log(expression=False, message=f"Expected key 'stamp' in key '{key}' in result to be ISO 8601: {repr(e)}")
        if 'duration' in result[key]:
            assert_type_value(obj=result[key]['duration'], type_or_value=float, name=f"key 'duration' in key '{key}' in result")
            assert_log(expression=result[key]['duration'] >= 0, message=f"Expected key 'duration' in key '{key}' in result to be non-negative but got '{result[key]['duration']}'.")
        if result[key]['success']:
            if key == 'image':
                assert_keys(obj=result[key], keys=['stamp', 'success', 'logs', 'duration', 'path'], mode="required", name="key 'image' in result")
                if settings['include_image']:
                    assert_log(expression='data' in result[key], message="Expected key 'data' in key 'image' because setting 'include_image' is 'True'.")
                    assert_type_value(obj=result[key]['data'], type_or_value=str, name="key 'data' in key 'image' in result")
                    assert_log(expression=len(result[key]['data']) > 0, message="Expected key 'data' in key 'image' in result to be a non-empty string.")
                assert_type_value(obj=result[key].get('path'), type_or_value=[None, str], name="key 'path' in key 'image' in result")
            else:
                if key == 'scene_description':
                    if result['run']['type'] != "worker" and not settings[key]['skip']:
                        assert_log(expression='settings' in result[key], message="Expected key 'settings' in key 'scene_description' in result for non-worker result.")
                    if 'settings' in result[key]:
                        assert_type_value(obj=result[key]['settings'], type_or_value=dict, name="key 'settings' in key 'scene_description' in result")
                        for s_name in ['skip', 'system_prompt_role', 'system_prompt', 'image_prompt_role', 'image_prompt_detail', 'description_prompt_role', 'description_prompt']:
                            if s_name == 'skip' and result[key]['settings'][s_name] is True:
                                continue
                            assert_log(expression=result[key]['settings'][s_name] == settings[key][s_name], message=f"Expected setting '{s_name}' in key 'scene_description' in result to be '{settings[key][s_name]}' but got '{result[key]['settings'][s_name]}'.")
                    assert_type_value(obj=result[key]['data'], type_or_value=str, name="key 'data' in key 'scene_description' in result")
                    assert_log(expression=len(result[key]['data']) > 0, message="Expected key 'data' in key 'scene_description' in result to be a non-empty string.")
                elif key == 'structured_description':
                    if result['run']['type'] != "worker" and not settings[key]['skip']:
                        assert_log(expression='settings' in result[key], message="Expected key 'settings' in key 'structured_description' in result for non-worker result.")
                    if 'settings' in result[key]:
                        assert_type_value(obj=result[key]['settings'], type_or_value=dict, name="key 'settings' in key 'structured_description' in result")
                        for s_name in ['skip', 'use_scene_description', 'system_prompt_role', 'system_prompt', 'image_prompt_role', 'image_prompt_detail', 'description_prompt_role', 'description_prompt', 'response_type', 'keys_required', 'keys_required_types', 'keys_optional', 'keys_optional_types']:
                            if s_name == 'skip' and result[key]['settings'][s_name] is True:
                                continue
                            assert_log(expression=result[key]['settings'][s_name] == settings[key][s_name], message=f"Expected setting '{s_name}' in key 'structured_description' in result to be '{settings[key][s_name]}' but got '{result[key]['settings'][s_name]}'.")
                    assert_type_value(obj=result[key]['data'], type_or_value=list, name="key 'data' in key 'structured_description' in result")
                    for i, item in enumerate(result[key]['data']):
                        assert_type_value(obj=item, type_or_value=dict, name=f"item '{i}' in key 'data' in key 'structured_description'")
                        for j, k_req in enumerate(settings[key]['keys_required']):
                            assert_log(expression=k_req in item, message=f"Missing required key '{k_req}' in item '{i}'")
                        for j, k in enumerate(settings[key]['keys_required'] + settings[key]['keys_optional']):
                            if k not in item:
                                continue
                            val = item[k]
                            t = (settings[key]['keys_required_types'] + settings[key]['keys_optional_types'])[j]
                            ok = False
                            if t == "str":
                                ok = isinstance(val, str) and len(val) > 0
                            elif t == "bool":
                                ok = isinstance(val, bool)
                            elif t == "int":
                                ok = isinstance(val, int) and not isinstance(val, bool)
                            elif t == "float":
                                ok = isinstance(val, float)
                            elif t == "unit":
                                ok = isinstance(val, float) and 0.0 <= val <= 1.0
                            elif t == "list":
                                ok = isinstance(val, list)
                            elif t == "bbox[int]":
                                ok = isinstance(val, list) and len(val) == 4 and all(isinstance(x, int) and not isinstance(x, bool) and x >= 0 for x in val) and val[2] > val[0] and val[3] > val[1]
                            elif t == "bbox[float]":
                                ok = isinstance(val, list) and len(val) == 4 and all(isinstance(x, float) and x >= 0 for x in val) and val[2] > val[0] and val[3] > val[1]
                            else:
                                raise NotImplementedError(t)
                            if j < len(settings[key]['keys_required']):
                                assert_log(expression=ok, message=f"Invalid value for required key '{k}' in item '{i}': {val} ({type(val).__name__})")
                            else:
                                assert_log(expression=ok, message=f"Invalid value for optional key '{k}' in item '{i}': {val} ({type(val).__name__})")
                        unexpected = set(item.keys()) - set(settings[key]['keys_required'] + settings[key]['keys_optional'])
                        assert_log(len(unexpected) == 0, f"Unexpected keys in item '{i}': {unexpected}")
                elif key == 'detection':
                    if result['run']['type'] != "worker" and not settings[key]['skip']:
                        assert_log(expression='settings' in result[key], message="Expected key 'settings' in key 'detection' in result for non-worker result.")
                    if 'settings' in result[key]:
                        assert_type_value(obj=result[key]['settings'], type_or_value=dict, name="key 'settings' in key 'detection' in result")
                        if result[key]['settings']['skip'] is False:
                            assert_log(expression=result[key]['settings']['skip'] == settings[key]['skip'], message=f"Expected setting 'skip' in key 'detection' in result to be '{settings[key]['skip']}' but got '{result[key]['settings']['skip']}'.")
                        assert_log(expression=result[key]['settings']['prompt_key'] == settings[key]['prompt_key'], message=f"Expected setting 'prompt_key' in key 'detection' in result to be '{settings[key]['prompt_key']}' but got '{result[key]['settings']['prompt_key']}'.")
                    assert_type_value(obj=result[key]['data'], type_or_value=list, name="key 'data' in key 'detection' in result")
                    prompt_key = settings['detection']['prompt_key']
                    expected_prompts = [o[prompt_key] for o in result['structured_description']['data']]
                    for i, item in enumerate(result[key]['data']):
                        assert_keys(obj=item, keys=['box_xyxy', 'confidence', 'prompt'], mode="whitelist", name=f"detection item '{i}'")
                        assert_keys(obj=item, keys=['box_xyxy', 'prompt'], mode="required", name=f"detection item '{i}'")
                        assert_type_value(obj=item['box_xyxy'], type_or_value=list, name=f"key 'box_xyxy' in detection item '{i}'")
                        assert_log(expression=len(item['box_xyxy']) == 4, message=f"Expected key 'box_xyxy' in detection item '{i}' to have length '4' but got '{len(item['box_xyxy'])}'.")
                        for j, coord in enumerate(item['box_xyxy']):
                            assert_type_value(obj=coord, type_or_value=int, name=f"coordinate '{j}' in box_xyxy in detection item '{i}'")
                            assert_log(expression=coord >= 0, message=f"Expected coordinate '{j}' in detection item '{i}' to be non-negative but got '{coord}'.")
                        assert_log(expression=item['box_xyxy'][2] > item['box_xyxy'][0], message=f"Expected coordinate '2' (x1) to be greater than coordinate '0' (x0) in detection item '{i}' but got x0='{item['box_xyxy'][0]}' and x1='{item['box_xyxy'][2]}'.")
                        assert_log(expression=item['box_xyxy'][3] > item['box_xyxy'][1], message=f"Expected coordinate '3' (y1) to be greater than coordinate '1' (y0) in detection item '{i}' but got y0='{item['box_xyxy'][1]}' and y1='{item['box_xyxy'][3]}'.")
                        if 'confidence' in item:
                            assert_type_value(obj=item['confidence'], type_or_value=float, name=f"key 'confidence' in detection item '{i}'")
                            assert_log(expression=0.0 <= item['confidence'] <= 1.0, message=f"Expected key 'confidence' in detection item '{i}' to be between 0 and 1 but got '{item['confidence']}'.")
                        assert_type_value(obj=item['prompt'], type_or_value=str, name=f"key 'prompt' in detection item '{i}'")
                        assert_log(expression=item['prompt'] in expected_prompts, message=f"Expected prompt '{item['prompt']}' in detection item '{i}' to be one of the requested prompts from the structured description: {expected_prompts}")
                    for p in expected_prompts:
                        assert_log(expression=any(d['prompt'] == p for d in result[key]['data']), message=f"Expected prompt '{p}' from structured description to be present in the detections but it was not found.")
                elif key == 'segmentation':
                    if result['run']['type'] != "worker":
                        assert_type_value(obj=result[key]['settings'], type_or_value=dict, name="key 'settings' in key 'segmentation' in result")
                        if result[key]['settings']['skip'] is False:
                            assert_log(expression=result[key]['settings']['skip'] == settings[key]['skip'], message=f"Expected setting 'skip' in key 'segmentation' in result to be '{settings[key]['skip']}' but got '{result[key]['settings']['skip']}'.")
                        assert_log(expression=result[key]['settings']['track'] == settings[key]['track'], message=f"Expected setting 'track' in key 'segmentation' in result to be '{settings[key]['track']}' but got '{result[key]['settings']['track']}'.")
                    assert_type_value(obj=result[key]['data'], type_or_value=list, name="key 'data' in key 'segmentation' in result")
                    assert_log(expression=len(result[key]['data']) == len(result['detection']['data']), message=f"Expected number of segmentations to be '{len(result['detection']['data'])}' (matching detections) but got '{len(result[key]['data'])}'.")
                    if settings[key]['track']:
                        assert_log(expression='duration_init' in result[key], message="Expected key 'duration_init' in key 'segmentation' in result because setting 'track' is 'True'.")
                        assert_type_value(obj=result[key]['duration_init'], type_or_value=float, name="key 'duration_init' in key 'segmentation' in result")
                        assert_log(expression=result[key]['duration_init'] > 0, message=f"Expected key 'duration_init' in key 'segmentation' in result to be greater zero but got '{result[key]['duration_init']}'.")
                    else:
                        assert_log(expression='duration_init' not in result[key], message="Expected key 'duration_init' not in key 'segmentation' in result because setting 'track' is 'False'.")
                    for i, item in enumerate(result[key]['data']):
                        assert_keys(obj=item, keys=['track_id', 'box_xyxy', 'mask'], mode="required", name=f"segmentation item '{i}'")
                        assert_type_value(obj=item['track_id'], type_or_value=int, name=f"key 'track_id' in segmentation item '{i}'")
                        assert_log(expression=item['track_id'] == i, message=f"Expected key 'track_id' in segmentation item '{i}' to match the index '{i}' but got '{item['track_id']}'.")
                        assert_type_value(obj=item['box_xyxy'], type_or_value=list, name=f"key 'box_xyxy' in segmentation item '{i}'")
                        assert_log(expression=len(item['box_xyxy']) == 4, message=f"Expected key 'box_xyxy' in segmentation item '{i}' to have length '4' but got '{len(item['box_xyxy'])}'.")
                        assert_type_value(obj=item['mask'], type_or_value=str, name=f"key 'mask' in segmentation item '{i}'")
                        assert_log(expression=len(item['mask']) > 0, message=f"Expected key 'mask' in segmentation item '{i}' to be a non-empty base64 string.")

def log_result(result):
    from nimbro_api.utility.misc import print_lines
    threshold = 100000
    for key in result:
        print()
        if isinstance(result[key], dict):
            string = json.dumps(result[key], indent=4)
        else:
            string = str(result[key])
        print_lines(string=string if len(string) < threshold else f"{string[:threshold]}...", prefix_first_line=key, prefix_next_lines=" " * len(key), line_length=200, style="")
    print(f"\nKeys: {result.keys()}")

def test_01_image_path():
    client = VlmGist(settings={
        'scene_description.skip': False,
        'structured_description.skip': False,
        'structured_description.use_scene_description': False,
        'detection.skip': False,
        'segmentation.skip': False,
        'segmentation.track': False,
        'include_image': False
    })
    success, message, result = client.run(image=os.path.join(nimbro_api.__path__[0], "test", "assets", "test.png"))
    assert_log(expression=success, message=message)
    assert_result(success=success, message=message, result=result, settings=client.get_settings())
    # log_result(result=result)

def test_02_image_bytes():
    client = VlmGist(settings={
        'scene_description.skip': False,
        'structured_description.skip': False,
        'structured_description.use_scene_description': True,
        'detection.skip': False,
        'segmentation.skip': False,
        'segmentation.track': False,
        'include_image': False
    })
    success, message, image_data, _ = parse_image_b64(image=os.path.join(nimbro_api.__path__[0], "test", "assets", "test.png"))
    assert_log(expression=success, message=message)
    assert_type_value(obj=image_data, type_or_value=str, name="parsed image")
    success, message, image_data = decode_b64(string=image_data, name="image")
    assert_log(expression=success, message=message)
    assert_type_value(obj=image_data, type_or_value=bytes, name="decoded image")
    success, message, result = client.run(image=image_data)
    assert_log(expression=success, message=message)
    assert_result(success=success, message=message, result=result, settings=client.get_settings())
    # log_result(result=result)

def test_03_image_dict():
    client = VlmGist(settings={
        'scene_description.skip': False,
        'structured_description.skip': False,
        'structured_description.use_scene_description': False,
        'detection.skip': False,
        'segmentation.skip': False,
        'segmentation.track': False,
        'include_image': False
    })

    image = {
        'data': os.path.join(nimbro_api.__path__[0], "test", "assets", "test.png"),
        'metadata': "Hello World!"
    }
    success, message, result = client.run(image=image)
    assert_log(expression=success, message=message)
    assert_result(success=success, message=message, result=result, settings=client.get_settings())
    assert_log(result['image'].get('metadata', None) == image['metadata'], f"Expected key 'image' in result to contain metadata but got {format_obj(result['image'])}")
    # log_result(result=result)

def test_04_scene_description_raw():
    client = VlmGist(settings={
        'scene_description.skip': True,
        'structured_description.skip': False,
        'structured_description.use_scene_description': False,
        'detection.skip': False,
        'segmentation.skip': True,
        'segmentation.track': False,
        'include_image': True
    })
    description = "There is a lot going on in this image."
    success, message, result = client.run(
        image=os.path.join(nimbro_api.__path__[0], "test", "assets", "test.png"),
        scene_description=description
    )
    assert_log(expression=success, message=message)
    assert_result(success=success, message=message, result=result, settings=client.get_settings())
    assert_log(expression=result['scene_description']['data'] == description, message=f"Expected key 'data' in key 'scene_description' to match input '{description}' but got '{result['scene_description']['data']}'.")
    # log_result(result=result)

def test_05_scene_description_dict():
    client = VlmGist(settings={
        'scene_description.skip': True,
        'structured_description.skip': False,
        'structured_description.use_scene_description': True,
        'detection.skip': True,
        'segmentation.skip': True,
        'segmentation.track': False
    })
    description = {
        'success': True,
        'logs': ["Manually injected scene description."],
        'data': "There is a lot going on in this image.",
        'stamp': datetime.datetime.now().isoformat(),
        'duration': 0.123
    }
    success, message, result = client.run(
        image=os.path.join(nimbro_api.__path__[0], "test", "assets", "test.png"),
        scene_description=description
    )
    assert_log(expression=success, message=message)
    assert_result(success=success, message=message, result=result, settings=client.get_settings())
    assert_log(expression=result['scene_description'] == description, message=f"Expected key 'data' in key 'scene_description' to match input '{description}' but got '{result['scene_description']}'.")
    # log_result(result=result)

def test_06_structured_description_raw():
    client = VlmGist(settings={
        'scene_description.skip': True,
        'structured_description.skip': True,
        'structured_description.use_scene_description': True,
        'detection.skip': False,
        'segmentation.skip': True,
        'segmentation.track': False
    })
    description = [{"object_name": "robot", "description": "A white humanoid robot with two arms and an omni-directional base."}]
    success, message, result = client.run(
        image=os.path.join(nimbro_api.__path__[0], "test", "assets", "test.png"),
        structured_description=description
    )
    assert_log(expression=success, message=message)
    assert_result(success=success, message=message, result=result, settings=client.get_settings())
    assert_log(expression=result['structured_description']['data'] == description, message=f"Expected key 'data' in key 'structured_description' to match input '{description}' but got '{result['structured_description']['data']}'.")
    # log_result(result=result)

def test_07_structured_description_dict():
    client = VlmGist(settings={
        'scene_description.skip': True,
        'structured_description.skip': True,
        'structured_description.use_scene_description': True,
        'detection.skip': False,
        'segmentation.skip': False,
        'segmentation.track': False
    })
    description = {
        'success': True,
        'logs': ["Injected list of objects."],
        'data': [{"object_name": "robot", "description": "A white humanoid robot with two arms and an omni-directional base."}],
        'stamp': datetime.datetime.now().isoformat(),
        'duration': 0.05
    }
    success, message, result = client.run(
        image=os.path.join(nimbro_api.__path__[0], "test", "assets", "test.png"),
        structured_description=description
    )
    assert_log(expression=success, message=message)
    assert_result(success=success, message=message, result=result, settings=client.get_settings())
    assert_log(expression=result['structured_description'] == description, message=f"Expected key 'data' in key 'structured_description' to match input '{description}' but got '{result['structured_description']}'.")
    # log_result(result=result)

def test_08_scene_and_structured_description():
    client = VlmGist(settings={
        'scene_description.skip': True,
        'structured_description.skip': True
    })
    scene = {
        'success': True,
        'logs': ["Manually injected scene description."],
        'data': "There is a lot going on in this image.",
        'stamp': datetime.datetime.now().isoformat(),
        'duration': 0.123
    }
    description = {
        'success': True,
        'logs': ["Injected list of objects."],
        'data': [{"object_name": "robot", "description": "A white humanoid robot with two arms and an omni-directional base."}],
        'stamp': datetime.datetime.now().isoformat(),
        'duration': 0.05
    }
    success, message, result = client.run(
        image=os.path.join(nimbro_api.__path__[0], "test", "assets", "test.png"),
        scene_description=scene,
        structured_description=description
    )
    assert_log(expression=success, message=message)
    assert_result(success=success, message=message, result=result, settings=client.get_settings())
    assert_log(expression=result['scene_description'] == scene, message=f"Expected key 'data' in key 'scene_description' to match input '{scene}' but got '{result['scene_description']}'.")
    assert_log(expression=result['structured_description'] == description, message=f"Expected key 'data' in key 'structured_description' to match input '{description}' but got '{result['structured_description']}'.")
    # log_result(result=result)

def test_09_detection_raw():
    client = VlmGist(settings={
        'scene_description.skip': True,
        'structured_description.skip': True,
        'detection.skip': True,
        'segmentation.skip': False
    })
    description = {
        'success': True,
        'logs': ["Injected list of objects."],
        'data': [{"object_name": "robot", "description": "A white humanoid robot with two arms and an omni-directional base."}],
        'stamp': datetime.datetime.now().isoformat(),
        'duration': 0.05
    }
    detection = [{"box_xyxy": [100, 200, 350, 280], "confidence": 0.98, "prompt": description['data'][0]['description']}]
    success, message, result = client.run(
        image=os.path.join(nimbro_api.__path__[0], "test", "assets", "test.png"),
        structured_description=description,
        detection=detection
    )
    assert_log(expression=success, message=message)
    assert_result(success=success, message=message, result=result, settings=client.get_settings())
    assert_log(expression=result['structured_description'] == description, message=f"Expected key 'data' in key 'structured_description' to match input '{description}' but got '{result['structured_description']}'.")
    assert_log(expression=result['detection']['data'] == detection, message=f"Expected key 'data' in key 'detection' to match input '{detection}' but got '{result['detection']['data']}'.")
    # log_result(result=result)

def test_10_detection_dict():
    client = VlmGist(settings={
        'scene_description.skip': True,
        'structured_description.skip': True,
        'detection.skip': True,
        'segmentation.skip': False
    })
    description = {
        'success': True,
        'logs': ["Injected list of objects."],
        'data': [{"object_name": "robot", "description": "A white humanoid robot with two arms and an omni-directional base."}],
        'stamp': datetime.datetime.now().isoformat(),
        'duration': 0.05
    }
    detection = {
        'success': True,
        'logs': ["Manual bounding box for monitor."],
        'data': [{"box_xyxy": [100, 200, 350, 280], "confidence": 0.98, "prompt": description['data'][0]['description']}],
        'stamp': datetime.datetime.now().isoformat(),
        'duration': 0.01
    }
    success, message, result = client.run(
        image=os.path.join(nimbro_api.__path__[0], "test", "assets", "test.png"),
        structured_description=description,
        detection=detection
    )
    assert_log(expression=success, message=message)
    assert_result(success=success, message=message, result=result, settings=client.get_settings())
    assert_log(expression=result['structured_description'] == description, message=f"Expected key 'data' in key 'structured_description' to match input '{description}' but got '{result['structured_description']}'.")
    assert_log(expression=result['detection'] == detection, message=f"Expected key 'data' in key 'detection' to match input '{detection}' but got '{result['detection']}'.")
    # log_result(result=result)

def test_11_tracking():
    client = VlmGist(settings={
        'scene_description.skip': True,
        'structured_description.skip': True,
        'detection.skip': True,
        'segmentation.skip': False,
        'segmentation.track': False
    })
    description = {
        'success': True,
        'logs': ["Injected list of objects."],
        'data': [{"object_name": "robot", "description": "A white humanoid robot with two arms and an omni-directional base."}],
        'stamp': datetime.datetime.now().isoformat(),
        'duration': 0.05
    }
    detection = {
        'success': True,
        'logs': ["Manual bounding box for monitor."],
        'data': [{"box_xyxy": [100, 200, 350, 280], "confidence": 0.98, "prompt": description['data'][0]['description']}],
        'stamp': datetime.datetime.now().isoformat(),
        'duration': 0.01
    }
    success, message, result = client.run(
        image=os.path.join(nimbro_api.__path__[0], "test", "assets", "test.png"),
        structured_description=description,
        detection=detection
    )
    assert_log(expression=success, message=message)
    assert_result(success=success, message=message, result=result, settings=client.get_settings())
    assert_log(expression=result['structured_description'] == description, message=f"Expected key 'data' in key 'structured_description' to match input '{description}' but got '{result['structured_description']}'.")
    assert_log(expression=result['detection'] == detection, message=f"Expected key 'data' in key 'detection' to match input '{detection}' but got '{result['detection']}'.")
    # log_result(result=result)

def test_12_structured_description_bbox():
    client = VlmGist(settings={
        'scene_description.skip': True,
        'structured_description.skip': False,
        'structured_description.use_scene_description': False,
        # 'structured_description.chat_completions.logger_severity': "info",
        'structured_description.description_prompt':
            "Provide a list in JSON format that contains each object (including furniture, persons, and animals) visible in the image above. "
            "Explicitly include each object instance as an individual list element, and never group multiple instances that are clearly distinct from one another. "
            "Each list element must be a dictionary with the fields label and description. "
            "The label of all humans must be person."
            "The description must be a single short sentence (max. 10 words, starting with 'A' or 'An'), "
            "that differs from the other descriptions and summarizes the most important information about the type, color, and appearance of the object, "
            "allowing for a visual identification of the object without knowing any of the descriptions generated for the other objects. "
            "In addition, also provided the key box_2d containing a bounding box if the object in [x1, y1, x2, y2] format.",
        'structured_description.keys_required': ['label', 'description'],
        'structured_description.keys_required_types': ['str', 'str'],
        'structured_description.keys_optional': ['box_2d'],
        'structured_description.keys_optional_types': ['bbox[int]'],
        'detection.skip': False,
        'detection.extract_from_description': False,
        'segmentation.skip': False,
        'segmentation.track': False,
        'include_image': False
    })
    success, message, result = client.run(image=os.path.join(nimbro_api.__path__[0], "test", "assets", "test.png"))
    assert_log(expression=success, message=message)
    assert_result(success=success, message=message, result=result, settings=client.get_settings())
    # log_result(result=result)

def test_13_structured_description_bbox_as_detection():
    client = VlmGist(settings={
        'scene_description.skip': True,
        'structured_description.skip': False,
        'structured_description.use_scene_description': False,
        # 'structured_description.chat_completions.logger_severity': "info",
        'structured_description.description_prompt':
            "Provide a list in JSON format that contains each object (including furniture, persons, and animals) visible in the image above. "
            "Explicitly include each object instance as an individual list element, and never group multiple instances that are clearly distinct from one another. "
            "Each list element must be a dictionary with the fields label and description. "
            "The label of all humans must be person."
            "The description must be a single short sentence (max. 10 words, starting with 'A' or 'An'), "
            "that differs from the other descriptions and summarizes the most important information about the type, color, and appearance of the object, "
            "allowing for a visual identification of the object without knowing any of the descriptions generated for the other objects. "
            "In addition, also provided the key box_2d containing a bounding box if the object in [x1, y1, x2, y2] format.",
        'structured_description.keys_required': ['label', 'description', 'box_2d'],
        'structured_description.keys_required_types': ['str', 'str', 'bbox[int]'],
        'structured_description.keys_optional': [],
        'structured_description.keys_optional_types': [],
        'detection.skip': False,
        'detection.extract_from_description': True,
        'segmentation.skip': False,
        'segmentation.track': False,
        'include_image': False
    })
    success, message, result = client.run(image=os.path.join(nimbro_api.__path__[0], "test", "assets", "test.png"))
    assert_log(expression=success, message=message)
    assert_result(success=success, message=message, result=result, settings=client.get_settings())
    # log_result(result=result)

def test_13_parallel_threads(n=10):
    client = VlmGist(settings={
        'scene_description.skip': True,
        'structured_description.skip': False,
        'structured_description.use_scene_description': False,
        'detection.skip': False,
        'segmentation.skip': False,
        'segmentation.track': False,
        'batch_size': n,
        'batch_style': "threading",
        'include_image': True
    })
    images = [os.path.join(nimbro_api.__path__[0], "test", "assets", "test.png")] * n
    success, message, result = client.run(image=images)
    assert_log(expression=success, message=message)
    assert_result(success=success, message=message, result=result, settings=client.get_settings())
    assert_log(expression=len(result['batch']) == len(images), message=f"Expected key 'batch' in result to contain '{len(images)}' elements but got '{len(result['batch'])}'.")
    # log_result(result=result)
    return message

def test_14_parallel_multiprocessing(n=10):
    client = VlmGist(settings={
        'scene_description.skip': True,
        'structured_description.skip': False,
        'structured_description.use_scene_description': False,
        'detection.skip': False,
        'segmentation.skip': False,
        'segmentation.track': False,
        'batch_size': n,
        'batch_style': "multiprocessing",
        'include_image': False
    })
    images = [os.path.join(nimbro_api.__path__[0], "test", "assets", "test.png")] * n
    success, message, result = client.run(image=images)
    assert_log(expression=success, message=message)
    assert_result(success=success, message=message, result=result, settings=client.get_settings())
    assert_log(expression=len(result['batch']) == len(images), message=f"Expected key 'batch' in result to contain '{len(images)}' elements but got '{len(result['batch'])}'.")
    # log_result(result=result)
    return message
