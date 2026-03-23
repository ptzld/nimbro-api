import os

from nimbro_api.utility.misc import assert_type_value, assert_log
from ..client.speech import Speech

def test_1_utilities():
    client = Speech()

    success, message, api_key = client.get_api_key()
    assert_type_value(obj=success, type_or_value=bool, name="success")
    assert_type_value(obj=message, type_or_value=str, name="message")
    assert_log(expression=success, message=message)
    assert_type_value(obj=api_key, type_or_value=str, name="message")

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
