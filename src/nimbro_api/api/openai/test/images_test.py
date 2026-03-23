import os

from nimbro_api.utility.misc import assert_type_value, assert_log
from ..client.images import Images

def test_1_utilities():
    client = Images()

    success, message, api_key = client.get_api_key()
    assert_type_value(obj=success, type_or_value=bool, name="success")
    assert_type_value(obj=message, type_or_value=str, name="message")
    assert_log(expression=success, message=message)
    assert_type_value(obj=api_key, type_or_value=str, name="message")

def test_2_endpoint():
    client = Images()

    success, message, models = client.get_models()
    assert_type_value(obj=success, type_or_value=bool, name="success")
    assert_type_value(obj=message, type_or_value=str, name="message")
    assert_log(expression=success, message=message)
    assert_type_value(obj=models, type_or_value=list, name="result of get_models()")

def test_3_inference():
    client = Images(settings={'cache_read': True})

    success, message, image = client.get_image("A cat named 'The evil power in Mirkwood'.", model="dall-e-2", quality="", style="", size="512x512")
    assert_type_value(obj=success, type_or_value=bool, name="success")
    assert_type_value(obj=message, type_or_value=str, name="message")
    assert_log(expression=success, message=message)
    assert_type_value(obj=image, type_or_value=str, name="result of get_image()")
    assert_log(expression=os.path.exists(image), message=f"Expected generated file '{image}' to exist.")
