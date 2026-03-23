import os

import nimbro_api
from nimbro_api.utility.misc import assert_type_value, assert_log
from ..client.translations import Translations

def test_1_utilities():
    client = Translations()

    success, message, api_key = client.get_api_key()
    assert_type_value(obj=success, type_or_value=bool, name="success")
    assert_type_value(obj=message, type_or_value=str, name="message")
    assert_log(expression=success, message=message)
    assert_type_value(obj=api_key, type_or_value=str, name="message")

def test_2_endpoint():
    client = Translations()

    success, message, models = client.get_models()
    assert_type_value(obj=success, type_or_value=bool, name="success")
    assert_type_value(obj=message, type_or_value=str, name="message")
    assert_log(expression=success, message=message)
    assert_type_value(obj=models, type_or_value=list, name="result of get_models()")

def test_3_inference():
    client = Translations()

    success, message, translation = client.get_translation(os.path.join(nimbro_api.__path__[0], "test/assets/test.wav"))
    assert_type_value(obj=success, type_or_value=bool, name="success")
    assert_type_value(obj=message, type_or_value=str, name="message")
    assert_log(expression=success, message=message)
    assert_type_value(obj=translation, type_or_value=dict, name="result of get_translation()")
