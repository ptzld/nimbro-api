import os

import nimbro_api
from nimbro_api.utility.misc import assert_type_value, assert_log
from ..client.florence2 import Florence2

def test_1_utilities():
    client = Florence2()

    settings = client.get_settings()
    assert_type_value(obj=settings['endpoint']['key_type'], type_or_value="environment", name="key 'key_type' of default endpoint")
    key_name = settings['endpoint']['key_value']
    key_before = os.getenv(key_name)
    key_now = "supersecretkey"
    os.environ[key_name] = key_now

    success, message, api_key = client.get_api_key()
    assert_type_value(obj=success, type_or_value=bool, name="success")
    assert_type_value(obj=message, type_or_value=str, name="message")
    assert_log(expression=success, message=message)
    assert_type_value(obj=api_key, type_or_value=key_now, name="API key")

    assert_type_value(obj=os.getenv(key_name), type_or_value=key_now, name=f"environment variable '{key_name}' (pre)")

    if key_before is None:
        key_before = key_now

    success, message = nimbro_api.set_api_key(name=key_name, key=key_before)
    assert_type_value(obj=success, type_or_value=bool, name="success")
    assert_type_value(obj=message, type_or_value=str, name="message")
    assert_log(expression=success, message=message)

    assert_type_value(obj=os.getenv(key_name), type_or_value=key_before, name=f"environment variable '{key_name}' (post)")

def test_2_endpoint():
    client = Florence2(validate_health=True)

    success, message, flavor = client.get_status()
    assert_type_value(obj=success, type_or_value=bool, name="success")
    assert_type_value(obj=message, type_or_value=str, name="message")
    assert_log(expression=success, message=message)
    assert_type_value(obj=flavor, type_or_value=[str, None], name="result of get_status()")

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

    # success, message = client.load()
    # assert_type_value(obj=success, type_or_value=bool, name="success")
    # assert_type_value(obj=message, type_or_value=str, name="message")
    # assert_log(expression=success, message=message)

    # success, message = client.unload()
    # assert_type_value(obj=success, type_or_value=bool, name="success")
    # assert_type_value(obj=message, type_or_value=str, name="message")
    # assert_log(expression=success, message=message)

def test_3_inference():
    client = Florence2()

    success, message, result = client.get_response(
        image=os.path.join(nimbro_api.__path__[0], "test", "assets", "test.png"),
    )
    assert_type_value(obj=success, type_or_value=bool, name="success")
    assert_type_value(obj=message, type_or_value=str, name="message")
    assert_log(expression=success, message=message)
    assert_type_value(obj=result, type_or_value=dict, name="result")

    # success, message = client.unload()
    # assert_type_value(obj=success, type_or_value=bool, name="success")
    # assert_type_value(obj=message, type_or_value=str, name="message")
    # assert_log(expression=success, message=message)
