from nimbro_api.utility.misc import assert_type_value, assert_keys, assert_log
from ..client.searxng import SearXNG

def test_1_inference():
    client = SearXNG(endpoint="localhost")

    success, message, response = client.search(query="Python documentation")
    assert_type_value(obj=success, type_or_value=bool, name="success")
    assert_type_value(obj=message, type_or_value=str, name="message")
    assert_log(expression=success, message=message)
    assert_type_value(obj=response, type_or_value=dict, name="result of search()")
    assert_keys(obj=response, keys=['query', 'results', 'unresponsive_engines'], mode="required", name="search response")
    assert_type_value(obj=response['query'], type_or_value="Python documentation", name="response query")
    assert_type_value(obj=response['results'], type_or_value=list, name="search results")
    assert_log(expression=len(response['results']) > 0, message=f"Expected at least one search result but got '{len(response['results'])}'.")
    for index, result in enumerate(response['results']):
        assert_type_value(obj=result, type_or_value=dict, name=f"result at index '{index}'")
        assert_keys(obj=result, keys=['title', 'url'], mode="required", name=f"result at index '{index}'")
        assert_type_value(obj=result['title'], type_or_value=str, name=f"title of result at index '{index}'")
        assert_type_value(obj=result['url'], type_or_value=str, name=f"URL of result at index '{index}'")
        assert_log(expression=result['url'].startswith(("http://", "https://")), message=f"Expected an HTTP or HTTPS result URL but got '{result['url']}'.")
        if 'content' in result and result['content'] is not None:
            assert_type_value(obj=result['content'], type_or_value=str, name=f"snippet of result at index '{index}'")
    assert_type_value(obj=response['unresponsive_engines'], type_or_value=list, name="unresponsive engines")

def test_2_inference():
    client = SearXNG(endpoint="localhost", language="en", safesearch=0, time_range="year")

    success, message, response = client.search(query="Python release", page=2)
    assert_type_value(obj=success, type_or_value=bool, name="success")
    assert_type_value(obj=message, type_or_value=str, name="message")
    assert_log(expression=success, message=message)
    assert_type_value(obj=response, type_or_value=dict, name="result of search()")
    assert_keys(obj=response, keys=['query', 'results', 'unresponsive_engines'], mode="required", name="search response")
    assert_type_value(obj=response['query'], type_or_value="Python release", name="response query")
    assert_type_value(obj=response['results'], type_or_value=list, name="search results")
    assert_type_value(obj=response['unresponsive_engines'], type_or_value=list, name="unresponsive engines")
