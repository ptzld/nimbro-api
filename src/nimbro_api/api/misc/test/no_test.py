from nimbro_api.utility.misc import assert_type_value, assert_log
from ..client.no import No

def test_1_run():
    client = No()

    success, message, response = client.no()
    assert_type_value(obj=success, type_or_value=bool, name="success")
    assert_type_value(obj=message, type_or_value=str, name="message")
    assert_log(expression=success, message=message)
    assert_type_value(obj=response, type_or_value=str, name="response")
