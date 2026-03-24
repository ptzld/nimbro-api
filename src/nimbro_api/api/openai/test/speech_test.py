import os

import nimbro_api
from nimbro_api.utility.misc import assert_type_value, assert_log
from ..client.speech import Speech

def test_1_utilities():
    client = Speech()

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
    client = Speech()

    success, message, models = client.get_models()
    assert_type_value(obj=success, type_or_value=bool, name="success")
    assert_type_value(obj=message, type_or_value=str, name="message")
    assert_log(expression=success, message=message)
    assert_type_value(obj=models, type_or_value=list, name="result of get_models()")

def test_3_inference():
    client = Speech(settings={'cache_read': False})

    success, message, audio = client.get_speech(text="Hallo! Ich bin ein autonomer Haushaltsroboter der Arbeitsgruppe für Autonome Intelligente Systeme an der Universität Bonn.")
    assert_type_value(obj=success, type_or_value=bool, name="success")
    assert_type_value(obj=message, type_or_value=str, name="message")
    assert_log(expression=success, message=message)
    assert_type_value(obj=audio, type_or_value=str, name="result of get_speech()")
    assert_log(expression=os.path.exists(audio), message=f"Expected generated file '{audio}' to exist.")
