import os

import nimbro_api
from nimbro_api.utility.misc import assert_type_value, assert_keys, assert_log
from ..client.sam2_realtime import Sam2Realtime

def test_1_utilities():
    client = Sam2Realtime()

    success, message, api_key = client.get_api_key()
    assert_type_value(obj=success, type_or_value=bool, name="success")
    assert_type_value(obj=message, type_or_value=str, name="message")
    assert_log(expression=success, message=message)
    assert_type_value(obj=api_key, type_or_value=str, name="message")

def test_2_endpoint():
    client = Sam2Realtime(validate_health=True)

    success, message, flavor, init = client.get_status()
    assert_type_value(obj=success, type_or_value=bool, name="success")
    assert_type_value(obj=message, type_or_value=str, name="message")
    assert_log(expression=success, message=message)
    assert_type_value(obj=flavor, type_or_value=[str, None], name="result of get_status()")
    assert_type_value(obj=init, type_or_value=[bool, None], name="initialization result of get_health()")

    success, message, is_healthy = client.get_health()
    assert_type_value(obj=success, type_or_value=bool, name="success")
    assert_type_value(obj=message, type_or_value=str, name="message")
    assert_log(expression=success, message=message)
    assert_type_value(obj=is_healthy, type_or_value=bool, name="result of get_health()")

    success, message, flavors = client.get_flavors()
    assert_type_value(obj=success, type_or_value=bool, name="success")
    assert_type_value(obj=message, type_or_value=str, name="message")
    assert_log(expression=success, message=message)
    assert_type_value(obj=flavors, type_or_value=list, name="result of get_flavors()")

    success, message = client.load()
    assert_type_value(obj=success, type_or_value=bool, name="success")
    assert_type_value(obj=message, type_or_value=str, name="message")
    assert_log(expression=success, message=message)

    success, message = client.unload()
    assert_type_value(obj=success, type_or_value=bool, name="success")
    assert_type_value(obj=message, type_or_value=str, name="message")
    assert_log(expression=success, message=message)

def test_3_inference():
    client = Sam2Realtime()

    success, message, result = client.get_response(
        image=os.path.join(nimbro_api.__path__[0], "test", "assets", "test.png"),
        prompts=[{'object_id': 0, 'points': [[50, 50], [60, 60]], 'labels': [1, 1]}],
    )
    assert_type_value(obj=success, type_or_value=bool, name="success")
    assert_type_value(obj=message, type_or_value=str, name="message")
    assert_log(expression=success, message=message)
    assert_type_value(obj=result, type_or_value=list, name="result")
    for i, item in enumerate(result):
        assert_type_value(obj=item, type_or_value=dict, name=f"element '{i}' in result")
        assert_keys(obj=item, keys=['box_xyxy', 'mask', 'track_id'], mode="match", name=f"element '{i}' in result")

    success, message, result = client.get_response(
        image=os.path.join(nimbro_api.__path__[0], "test", "assets", "test.png"),
    )
    assert_type_value(obj=success, type_or_value=bool, name="success")
    assert_type_value(obj=message, type_or_value=str, name="message")
    assert_log(expression=success, message=message)
    assert_type_value(obj=result, type_or_value=list, name="result")
    for i, item in enumerate(result):
        assert_type_value(obj=item, type_or_value=dict, name=f"element '{i}' in result")
        assert_keys(obj=item, keys=['box_xyxy', 'mask', 'track_id'], mode="match", name=f"element '{i}' in result")

    success, message, result = client.get_response(
        image=os.path.join(nimbro_api.__path__[0], "test", "assets", "test.png"),
        prompts=[{'object_id': 0, 'bbox': [50, 50, 100, 100]}, {'object_id': 1, 'bbox': [1000, 1000, 1200, 1200]}],
    )
    assert_type_value(obj=success, type_or_value=bool, name="success")
    assert_type_value(obj=message, type_or_value=str, name="message")
    assert_log(expression=success, message=message)
    assert_type_value(obj=result, type_or_value=list, name="result")
    for i, item in enumerate(result):
        assert_type_value(obj=item, type_or_value=dict, name=f"element '{i}' in result")
        assert_keys(obj=item, keys=['box_xyxy', 'mask', 'track_id'], mode="match", name=f"element '{i}' in result")

    success, message, result = client.get_response(
        image=os.path.join(nimbro_api.__path__[0], "test", "assets", "test.png"),
    )
    assert_type_value(obj=success, type_or_value=bool, name="success")
    assert_type_value(obj=message, type_or_value=str, name="message")
    assert_log(expression=success, message=message)
    assert_type_value(obj=result, type_or_value=list, name="result")
    for i, item in enumerate(result):
        assert_type_value(obj=item, type_or_value=dict, name=f"element '{i}' in result")
        assert_keys(obj=item, keys=['box_xyxy', 'mask', 'track_id'], mode="match", name=f"element '{i}' in result")

    success, message = client.unload()
    assert_log(expression=success, message=message)
    assert_type_value(obj=success, type_or_value=bool, name="success")
    assert_type_value(obj=message, type_or_value=str, name="message")
