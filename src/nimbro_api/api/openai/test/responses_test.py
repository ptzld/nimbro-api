import os
import copy
import json
import time
import threading
import multiprocessing

import nimbro_api
from nimbro_api.utility.misc import assert_type_value, assert_keys, assert_log
from ..client.responses import Responses

def test_001_utilities():
    client = Responses()
    settings = client.get_settings()
    success, message = client.set_settings(settings)
    assert_type_value(obj=success, type_or_value=bool, name="success")
    assert_type_value(obj=message, type_or_value=str, name="message")
    assert_log(expression=success, message=message)
    assert_log(expression=client.get_settings() == settings, message="Expected settings to round-trip.")
    assert_type_value(obj=settings['endpoint']['key_type'], type_or_value="environment", name="key 'key_type' of default endpoint")
    key_name = settings['endpoint']['key_value']
    key_before = os.getenv(key_name)
    cached, _, cache = nimbro_api.query_cache(category="keys", identifier=key_name, mute=True)
    cache_keys = nimbro_api.get_settings('keys_cache')
    try:
        nimbro_api.clear_cache(category="keys", identifier=key_name, mute=True)
        nimbro_api.set_settings(keys_cache=False, mute=True)
        os.environ[key_name] = "supersecretkey"
        success, message, api_key = client.get_api_key()
        assert_type_value(obj=success, type_or_value=bool, name="success")
        assert_type_value(obj=message, type_or_value=str, name="message")
        assert_log(expression=success, message=message)
        assert_type_value(obj=api_key, type_or_value="supersecretkey", name="API key")
        key_after = key_before or "supersecretkey"
        success, message = nimbro_api.set_api_key(name=key_name, key=key_after)
        assert_type_value(obj=success, type_or_value=bool, name="success")
        assert_type_value(obj=message, type_or_value=str, name="message")
        assert_log(expression=success, message=message)
        assert_type_value(obj=os.getenv(key_name), type_or_value=key_after, name=f"environment variable '{key_name}'")
    finally:
        if key_before is None:
            os.environ.pop(key_name, None)
        else:
            os.environ[key_name] = key_before
        if cached:
            nimbro_api.update_cache(category="keys", identifier=key_name, data=cache['data'], mute=True)
        nimbro_api.set_settings(keys_cache=cache_keys, mute=True)
    for update in [
        {'temperature': True},
        {'temperature': float("nan")},
        {'timeout_read': 0},
        {'timeout_chunk_first': 0},
        {'timeout_chunk_first': True},
        {'timeout_chunk_first': float("inf")},
        {'timeout_chunk_next': -1},
        {'timeout_chunk_next': False},
        {'timeout_chunk_next': float("nan")},
        {'max_tokens': 0},
        {'parallel_tool_calls': None},
        {'parallel_tool_calls': 1},
        {'reasoning_effort': None},
        {'reasoning_effort': "invalid"},
        {'reasoning_summary': False},
        {'reasoning_summary': "invalid"},
        {'send_reasoning': None},
        {'send_reasoning': 1},
        {'verbosity': "invalid"},
        {'verbosity': False},
        {'endpoint': "vLLM", 'top_k': -1},
        {'endpoint': "OpenRouter", 'top_k': -1},
        {'endpoint': "vLLM", 'stop': ["a", "b", "c", "d", "e"]},
        {'endpoint': "OpenRouter", 'stop': ["a", "b", "c", "d", "e"]}
    ]:
        success, message = client.set_settings(update)
        assert_type_value(obj=success, type_or_value=bool, name="success")
        assert_type_value(obj=message, type_or_value=str, name="message")
        assert_log(expression=not success, message=f"Expected invalid settings to fail: {update}")
        assert_log(expression=client.get_settings() == settings, message="Expected failed validation to preserve settings.")
    success, message = client.set_settings()
    assert_type_value(obj=success, type_or_value=bool, name="success")
    assert_type_value(obj=message, type_or_value=str, name="message")
    assert_log(expression=success, message=message)
    assert_log(expression=client.get_settings() == settings, message="Expected initialization settings on reset.")

def test_002_endpoint():
    client = Responses()
    success, message, models = client.get_models()
    assert_type_value(obj=success, type_or_value=bool, name="success")
    assert_type_value(obj=message, type_or_value=str, name="message")
    assert_log(expression=success, message=message)
    assert_type_value(obj=models, type_or_value=list, name="models")

# context configuration

def test_003_utilities_context_config():
    client = Responses(retry=False)
    expected = []
    for mode, messages, index, reverse, valid in [
        ("remove", None, 0, True, False),
        ("replace", {'role': "user", 'content': "Hello"}, 0, True, False),
        ("insert", {'role': "user", 'content': "Hello"}, 0, True, True),
        ("insert", {'role': "assistant", 'content': "Hi"}, 0, True, True),
        ("insert", {'role': "system", 'content': "Be helpful"}, 0, False, True),
        ("replace", [{'role': "user", 'content': "New"}, {'role': "assistant", 'content': "Answer"}], 2, False, True),
        ("remove", None, 0, False, True),
        ("replace", [{}], 0, True, False),
        ("reset", None, 0, True, True)
    ]:
        items = [messages] if isinstance(messages, dict) else messages or []
        items = [dict(item, type="message") if 'role' in item else item for item in items]
        if valid:
            position = len(expected) - index - (mode != "insert") if reverse else index
            if mode == "reset":
                expected = copy.deepcopy(items)
            elif mode == "insert":
                expected[position:position] = copy.deepcopy(items)
            elif mode == "replace":
                expected[position:position + len(items)] = copy.deepcopy(items)
            else:
                del expected[position]
        success, message = client.set_context(mode=mode, messages=messages, index=index, reverse_indexing=reverse)
        assert_type_value(obj=success, type_or_value=bool, name="success")
        assert_type_value(obj=message, type_or_value=str, name="message")
        assert_log(expression=success == valid, message=message)
        success, message, context = client.get_context()
        assert_type_value(obj=success, type_or_value=bool, name="success")
        assert_type_value(obj=message, type_or_value=str, name="message")
        assert_log(expression=success, message=message)
        assert_type_value(obj=context, type_or_value=list, name="context")
        assert_log(expression=context == expected, message="Expected exact context after editing.")
        context.clear()
    success, message, result = client.prompt("Remember 372", response_type="none", temperature=0)
    assert_type_value(obj=success, type_or_value=bool, name="success")
    assert_type_value(obj=message, type_or_value=str, name="message")
    assert_log(expression=success, message=message)
    assert_type_value(obj=result, type_or_value=None, name="result")
    assert_type_value(obj=client.get_settings('temperature'), type_or_value=None, name="restored temperature")

def test_004_utilities_tool_config():
    client = Responses(retry=False)
    tools = [{
        'type': "function",
        'name': name,
        'description': None,
        'parameters': {
            'type': "object",
            'properties': {
                'options': {
                    'type': "object",
                    'properties': {'text': {'type': "string"}},
                    'required': ['text'],
                    'additionalProperties': False
                }
            },
            'required': ['options'],
            'additionalProperties': False
        },
        'strict': True
    } for name in ["speak", "translate"]]
    for definitions in [tools, tools[0], [], None]:
        success, message = client.set_tools(definitions)
        assert_type_value(obj=success, type_or_value=bool, name="success")
        assert_type_value(obj=message, type_or_value=str, name="message")
        assert_log(expression=success, message=message)
        expected = [definitions] if isinstance(definitions, dict) else definitions or []
        success, message, result = client.get_tools()
        assert_type_value(obj=success, type_or_value=bool, name="success")
        assert_type_value(obj=message, type_or_value=str, name="message")
        assert_log(expression=success, message=message)
        assert_type_value(obj=result, type_or_value=list, name="tools")
        assert_log(expression=result == expected, message="Expected native tool definitions to round-trip.")
        result.clear()
        success, message, result = client.get_tools()
        assert_type_value(obj=success, type_or_value=bool, name="success")
        assert_type_value(obj=message, type_or_value=str, name="message")
        assert_log(expression=success, message=message)
        assert_type_value(obj=result, type_or_value=list, name="tools")
        assert_log(expression=result == expected, message="Expected defensive tool returns.")
    success, message = client.set_tools([tools[0], tools[0]])
    assert_type_value(obj=success, type_or_value=bool, name="success")
    assert_type_value(obj=message, type_or_value=str, name="message")
    assert_log(expression=not success, message="Expected duplicate names to fail.")
    success, message, result = client.get_tools()
    assert_type_value(obj=success, type_or_value=bool, name="success")
    assert_type_value(obj=message, type_or_value=str, name="message")
    assert_log(expression=success, message=message)
    assert_type_value(obj=result, type_or_value=list, name="tools")
    assert_log(expression=result == [], message="Expected failed tool validation to preserve definitions.")
    call = {
        'type': "function_call",
        'call_id': "call_number",
        'name': "get_number",
        'arguments': "{}"
    }
    success, message = client.set_context(messages=call)
    assert_type_value(obj=success, type_or_value=bool, name="success")
    assert_type_value(obj=message, type_or_value=str, name="message")
    assert_log(expression=success, message=message)
    for supplied, expected in [(None, ["call_number"]), ("372", [])]:
        if supplied is not None:
            success, message, result = client.prompt(supplied, response_type="none")
            assert_type_value(obj=success, type_or_value=bool, name="success")
            assert_type_value(obj=message, type_or_value=str, name="message")
            assert_log(expression=success, message=message)
            assert_type_value(obj=result, type_or_value=None, name="result")
        success, message, awaited = client.get_awaited_tools()
        assert_type_value(obj=success, type_or_value=bool, name="success")
        assert_type_value(obj=message, type_or_value=str, name="message")
        assert_log(expression=success, message=message)
        assert_type_value(obj=awaited, type_or_value=list, name="awaited tools")
        assert_log(expression=awaited == expected, message="Expected pending call IDs to reflect supplied outputs.")

def test_005_utilities_media():
    client = Responses()
    assets = os.path.join(nimbro_api.__path__[0], "test", "assets")
    messages = [{
        'role': "user",
        'content': [
            {'type': "input_image", 'image_url': os.path.join(assets, "test.png")},
            {'type': "input_file", 'file_data': os.path.join(assets, "test.pdf")},
            {'type': "input_audio", 'input_audio': {'data': os.path.join(assets, "test.wav"), 'format': "wav"}},
            {'type': "input_video", 'video_url': os.path.join(assets, "test.mp4")}
        ]
    }]
    before = copy.deepcopy(messages)
    success, message = client.set_context(messages=messages)
    assert_type_value(obj=success, type_or_value=bool, name="success")
    assert_type_value(obj=message, type_or_value=str, name="message")
    assert_log(expression=success, message=message)
    success, message, context = client.get_context()
    assert_type_value(obj=success, type_or_value=bool, name="success")
    assert_type_value(obj=message, type_or_value=str, name="message")
    assert_log(expression=success, message=message)
    assert_type_value(obj=context, type_or_value=list, name="context")
    parts = context[0]['content']
    assert_log(expression=parts[0]['image_url'].startswith("data:image/png;base64,"), message="Expected PNG data URL.")
    assert_log(expression=parts[1]['file_data'].startswith("data:application/pdf;base64,"), message="Expected PDF data URL.")
    assert_log(expression=parts[2]['input_audio']['data'] != before[0]['content'][2]['input_audio']['data'], message="Expected encoded audio.")
    assert_log(expression=parts[3]['video_url'].startswith("data:video/mp4;base64,"), message="Expected MP4 data URL.")
    assert_log(expression=messages == before, message="Expected conversion not to mutate caller input.")
    for kind in ["function_call_output", "custom_tool_call_output", "computer_call_output"]:
        part = {
            'type': "computer_screenshot" if kind == "computer_call_output" else "input_image",
            'image_url': os.path.join(assets, "test.png")
        }
        item = {
            'type': kind,
            'call_id': "call_image",
            'output': part if kind == "computer_call_output" else [part]
        }
        before = copy.deepcopy(item)
        success, message = client.set_context(messages=item)
        assert_type_value(obj=success, type_or_value=bool, name="success")
        assert_type_value(obj=message, type_or_value=str, name="message")
        assert_log(expression=success, message=message)
        success, message, context = client.get_context()
        assert_type_value(obj=success, type_or_value=bool, name="success")
        assert_type_value(obj=message, type_or_value=str, name="message")
        assert_log(expression=success, message=message)
        assert_type_value(obj=context, type_or_value=list, name="tool context")
        output = context[0]['output']
        converted = output if isinstance(output, dict) else output[0]
        assert_log(expression=converted['type'] == part['type'] and converted['image_url'].startswith("data:image/png;base64,"), message="Expected embedded tool media with its native type.")
        assert_log(expression=item == before, message="Expected caller-owned tool output not to change.")

# text completions

def _vllm_model():
    client = Responses(endpoint="vLLM")
    success, message, models = client.get_models(age=0, retry=False, mute=True, timeout_connect=1, timeout_read=2)
    assert_type_value(obj=success, type_or_value=bool, name="success")
    assert_type_value(obj=message, type_or_value=str, name="message")
    if success:
        assert_type_value(obj=models, type_or_value=list, name="models")
    if not success or not models:
        return False, "vLLM endpoint is unavailable or is not hosting any models.", None
    return True, message, models[-1]

def _switch_context(client, settings, text, response_type="text"):
    success, message, before = client.get_context()
    assert_type_value(obj=success, type_or_value=bool, name="success")
    assert_type_value(obj=message, type_or_value=str, name="message")
    assert_log(expression=success, message=message)
    assert_type_value(obj=before, type_or_value=list, name="context before switching")
    assert_log(expression=bool(before), message="Expected populated history before switching providers.")
    success, message = client.set_settings(settings)
    assert_type_value(obj=success, type_or_value=bool, name="success")
    assert_type_value(obj=message, type_or_value=str, name="message")
    assert_log(expression=success, message=message)
    for name, value in settings.items():
        assert_log(expression=client.get_settings(name) == value, message=f"Expected switched setting '{name}' to be '{value}'.")
    success, message, context = client.get_context()
    assert_type_value(obj=success, type_or_value=bool, name="success")
    assert_type_value(obj=message, type_or_value=str, name="message")
    assert_log(expression=success, message=message)
    assert_type_value(obj=context, type_or_value=list, name="context after switching")
    assert_log(expression=context == before, message="Expected provider switching to preserve native history.")
    success, message, completion = client.prompt(text, response_type=response_type)
    assert_type_value(obj=success, type_or_value=bool, name="success")
    assert_type_value(obj=message, type_or_value=str, name="message")
    assert_log(expression=success, message=message)
    assert_type_value(obj=completion, type_or_value=dict, name="completion after switching")
    success, message, context = client.get_context()
    assert_type_value(obj=success, type_or_value=bool, name="success")
    assert_type_value(obj=message, type_or_value=str, name="message")
    assert_log(expression=success, message=message)
    assert_type_value(obj=context, type_or_value=list, name="context after continuation")
    user_message = {'type': "message", 'role': "user", 'content': text}
    assert_log(expression=context == before + [user_message] + completion['response']['output'], message="Expected cross-provider continuation to append native output without rewriting prior history.")
    return completion

def _text_completions(switch_settings=None, **kwargs):
    client = Responses(kwargs)
    for text, reset in [("Remember the number 372. Reply briefly.", False), ("What number did I give you?", False), ("Say hello.", True)]:
        success, message, before = client.get_context()
        assert_type_value(obj=success, type_or_value=bool, name="success")
        assert_type_value(obj=message, type_or_value=str, name="message")
        assert_log(expression=success, message=message)
        assert_type_value(obj=before, type_or_value=list, name="context before generation")
        if reset:
            before = []
        success, message, completion = client.prompt(text=text, reset_context=reset, response_type="text")
        assert_type_value(obj=success, type_or_value=bool, name="success")
        assert_type_value(obj=message, type_or_value=str, name="message")
        assert_log(expression=success, message=message)
        assert_type_value(obj=completion, type_or_value=dict, name="completion")
        assert_keys(obj=completion, keys=['text', 'refusal', 'reasoning', 'tools', 'usage', 'response'], mode="match", name="completion")
        assert_type_value(obj=completion['text'], type_or_value=str, name="completion text")
        assert_log(expression=bool(completion['text']), message="Expected assistant text.")
        if text.startswith("What"):
            assert_log(expression="372" in completion['text'], message="Expected retained conversation history.")
        for key in ['tokens_input', 'tokens_input_cached', 'tokens_input_uncached', 'tokens_output', 'tokens_output_reasoning', 'tokens_total']:
            assert_type_value(obj=completion['usage'][key], type_or_value=[int, None], name=f"usage '{key}'")
        success, message, context = client.get_context()
        assert_type_value(obj=success, type_or_value=bool, name="success")
        assert_type_value(obj=message, type_or_value=str, name="message")
        assert_log(expression=success, message=message)
        assert_type_value(obj=context, type_or_value=list, name="context")
        user_message = {
            'type': "message",
            'role': "user",
            'content': text
        }
        assert_log(expression=context == before + [user_message] + completion['response']['output'], message="Expected complete native history.")
        if switch_settings is not None and text.startswith("What"):
            completion = _switch_context(client, switch_settings, "What number did I give you? Reply using digits.")
            assert_log(expression="372" in completion['text'], message=f"Expected retained number after switching but got '{completion['text']}'.")
    success, message = client.set_context(messages={'role': "user", 'content': "Say goodbye."})
    assert_type_value(obj=success, type_or_value=bool, name="success")
    assert_type_value(obj=message, type_or_value=str, name="message")
    assert_log(expression=success, message=message)
    success, message, before = client.get_context()
    assert_type_value(obj=success, type_or_value=bool, name="success")
    assert_type_value(obj=message, type_or_value=str, name="message")
    assert_log(expression=success, message=message)
    assert_type_value(obj=before, type_or_value=list, name="context before generation")
    success, message, completion = client.prompt()
    assert_type_value(obj=success, type_or_value=bool, name="success")
    assert_type_value(obj=message, type_or_value=str, name="message")
    assert_log(expression=success, message=message)
    assert_type_value(obj=completion, type_or_value=dict, name="completion")
    success, message, context = client.get_context()
    assert_type_value(obj=success, type_or_value=bool, name="success")
    assert_type_value(obj=message, type_or_value=str, name="message")
    assert_log(expression=success, message=message)
    assert_type_value(obj=context, type_or_value=list, name="context")
    assert_log(expression=context == before + completion['response']['output'], message="Expected generation without new input.")

def test_006_openai_text_completions():
    return _text_completions(endpoint="OpenAI", model="gpt-6-luna", stream=False)

def test_007_openai_text_completions_stream():
    return _text_completions(endpoint="OpenAI", model="gpt-6-luna", stream=True)

def test_008_openrouter_text_completions():
    return _text_completions(endpoint="OpenRouter", model="~google/gemini-flash-latest", stream=False)

def test_009_openrouter_text_completions_stream():
    return _text_completions(endpoint="OpenRouter", model="~google/gemini-flash-latest", stream=True)

def test_010_vllm_text_completions():
    success, message, model = _vllm_model()
    if not success:
        return message
    return _text_completions(endpoint="vLLM", model=model, stream=False)

def test_011_vllm_text_completions_stream():
    success, message, model = _vllm_model()
    if not success:
        return message
    return _text_completions(endpoint="vLLM", model=model, stream=True)

def test_012_openai_openrouter_text_switch():
    for stream in [False, True]:
        _text_completions(endpoint="OpenAI", model="gpt-6-luna", stream=stream, switch_settings={'endpoint': "OpenRouter", 'model': "~google/gemini-flash-latest"})

def test_013_openrouter_openai_text_switch():
    for stream in [False, True]:
        _text_completions(endpoint="OpenRouter", model="~google/gemini-flash-latest", stream=stream, switch_settings={'endpoint': "OpenAI", 'model': "gpt-6-luna"})

def test_014_openrouter_stop():
    client = Responses(endpoint="OpenRouter", model="~google/gemini-flash-latest")
    for stop in ["red", ["red", "green", "blue"]]:
        for word in [stop] if isinstance(stop, str) else stop:
            success, message, completion = client.prompt(
                f"Copy exactly the following text without quotes, formatting, or commentary: alpha {word} omega",
                reset_context=True, response_type="text", stop=stop
            )
            assert_type_value(obj=success, type_or_value=bool, name="success")
            assert_type_value(obj=message, type_or_value=str, name="message")
            assert_log(expression=success, message=message)
            assert_type_value(obj=completion, type_or_value=dict, name="completion")
            assert_type_value(obj=completion['text'], type_or_value=str, name="completion text")
            assert_log(expression=completion['text'].strip() == "alpha", message=f"Expected generation to stop before '{word}' but got '{completion['text']}'.")

# structured output

def _json_completions(schema=False, switch_settings=None, **kwargs):
    client = Responses(kwargs)
    config = "json"
    if schema:
        config = {
            'name': "answer",
            'schema': {
                'type': "object",
                'properties': {'number': {'type': "integer"}},
                'required': ['number'],
                'additionalProperties': False
            },
            'strict': True
        }
    text = (
        "Return exactly one JSON object with the single key \"number\" and the integer value 372: "
        "{\"number\": 372}. Do not add other keys."
    )
    success, message, completion = client.prompt(text, response_type=config)
    assert_type_value(obj=success, type_or_value=bool, name="success")
    assert_type_value(obj=message, type_or_value=str, name="message")
    assert_log(expression=success, message=message)
    assert_type_value(obj=completion, type_or_value=dict, name="completion")
    assert_type_value(obj=completion['text'], type_or_value=dict, name="parsed JSON")
    expected = {'number': 372}
    assert_log(expression=completion['text'] == expected, message=f"Expected parsed structured output '{expected}' but got '{completion['text']}' with status '{completion['response']['status']}'.")
    native_text = "".join(part['text'] for item in completion['response']['output'] if item.get('type') == "message" for part in item['content'] if part.get('type') == "output_text")
    assert_log(expression=json.loads(native_text) == completion['text'], message="Expected native response text to match parsed JSON.")
    if switch_settings is not None:
        completion = _switch_context(client, switch_settings, "Return the same JSON object as before, using the same number and no other keys.", response_type=config)
        assert_type_value(obj=completion['text'], type_or_value=dict, name="parsed JSON after switching")
        assert_log(expression=completion['text'] == expected, message=f"Expected structured history reuse after switching but got '{completion['text']}'.")
        native_text = "".join(part['text'] for item in completion['response']['output'] if item.get('type') == "message" for part in item['content'] if part.get('type') == "output_text")
        assert_log(expression=json.loads(native_text) == expected, message="Expected native structured output after switching.")

def test_015_openai_json():
    return _json_completions(endpoint="OpenAI", model="gpt-6-luna", stream=False)

def test_016_openai_json_stream():
    return _json_completions(endpoint="OpenAI", model="gpt-6-luna", stream=True)

def test_017_openrouter_json():
    return _json_completions(endpoint="OpenRouter", model="~google/gemini-flash-latest", stream=False)

def test_018_openrouter_json_stream():
    return _json_completions(endpoint="OpenRouter", model="~google/gemini-flash-latest", stream=True)

def test_019_vllm_json():
    success, message, model = _vllm_model()
    if not success:
        return message
    return _json_completions(endpoint="vLLM", model=model, stream=False)

def test_020_vllm_json_stream():
    success, message, model = _vllm_model()
    if not success:
        return message
    return _json_completions(endpoint="vLLM", model=model, stream=True)

def test_021_openai_json_schema():
    return _json_completions(schema=True, endpoint="OpenAI", model="gpt-6-luna", stream=False)

def test_022_openai_json_schema_stream():
    return _json_completions(schema=True, endpoint="OpenAI", model="gpt-6-luna", stream=True)

def test_023_openrouter_json_schema():
    return _json_completions(schema=True, endpoint="OpenRouter", model="~google/gemini-flash-latest", stream=False)

def test_024_openrouter_json_schema_stream():
    return _json_completions(schema=True, endpoint="OpenRouter", model="~google/gemini-flash-latest", stream=True)

def test_025_vllm_json_schema():
    success, message, model = _vllm_model()
    if not success:
        return message
    return _json_completions(schema=True, endpoint="vLLM", model=model, stream=False)

def test_026_vllm_json_schema_stream():
    success, message, model = _vllm_model()
    if not success:
        return message
    return _json_completions(schema=True, endpoint="vLLM", model=model, stream=True)

def test_027_openai_openrouter_json_switch():
    for stream in [False, True]:
        _json_completions(endpoint="OpenAI", model="gpt-6-luna", stream=stream, switch_settings={'endpoint': "OpenRouter", 'model': "~google/gemini-flash-latest"})

def test_028_openrouter_openai_json_switch():
    for stream in [False, True]:
        _json_completions(endpoint="OpenRouter", model="~google/gemini-flash-latest", stream=stream, switch_settings={'endpoint': "OpenAI", 'model': "gpt-6-luna"})

def test_029_openai_openrouter_json_schema_switch():
    for stream in [False, True]:
        _json_completions(schema=True, endpoint="OpenAI", model="gpt-6-luna", stream=stream, switch_settings={'endpoint': "OpenRouter", 'model': "~google/gemini-flash-latest"})

def test_030_openrouter_openai_json_schema_switch():
    for stream in [False, True]:
        _json_completions(schema=True, endpoint="OpenRouter", model="~google/gemini-flash-latest", stream=stream, switch_settings={'endpoint': "OpenAI", 'model': "gpt-6-luna"})

# reasoning

def _reasoning_completion(switch_settings=None, **kwargs):
    client = Responses(kwargs)
    success, message, completion = client.prompt(
        "Find the smallest positive integer n with n modulo 17 = 5 and n modulo 19 = 7. Reply with only n."
    )
    assert_type_value(obj=success, type_or_value=bool, name="success")
    assert_type_value(obj=message, type_or_value=str, name="message")
    assert_log(expression=success, message=message)
    assert_type_value(obj=completion, type_or_value=dict, name="completion")
    assert_log(expression="311" in completion['text'], message=f"Expected arithmetic answer '311' but got '{completion['text']}'.")
    reasoning = [item for item in completion['response']['output'] if item.get('type') == "reasoning"]
    assert_log(expression=bool(reasoning) and completion['reasoning'] == reasoning, message=f"Expected preserved native reasoning but got output types '{[item.get('type') for item in completion['response']['output']]}' with usage '{completion['usage']}' and status '{completion['response']['status']}'.")
    if kwargs.get('endpoint') == "OpenAI":
        assert_log(expression=any(item.get('encrypted_content') for item in reasoning), message="Expected encrypted reasoning to exercise replay.")
    success, message, context = client.get_context()
    assert_type_value(obj=success, type_or_value=bool, name="success")
    assert_type_value(obj=message, type_or_value=str, name="message")
    assert_log(expression=success, message=message)
    assert_type_value(obj=context, type_or_value=list, name="context")
    assert_log(expression=context[1:] == completion['response']['output'], message="Expected replayable reasoning history.")
    if switch_settings is not None:
        success, message = client.set_settings(switch_settings)
        assert_type_value(obj=success, type_or_value=bool, name="success")
        assert_type_value(obj=message, type_or_value=str, name="message")
        assert_log(expression=success, message=message)
        success, message, switched_context = client.get_context()
        assert_log(expression=success, message=message)
        assert_log(expression=switched_context == context, message="Expected model/provider switching to preserve native reasoning history.")
    success, message, completion = client.prompt("Add one to your previous answer. Reply with only the number.")
    assert_type_value(obj=success, type_or_value=bool, name="success")
    assert_type_value(obj=message, type_or_value=str, name="message")
    assert_log(expression=success, message=message)
    assert_type_value(obj=completion, type_or_value=dict, name="completion")
    assert_log(expression="312" in completion['text'], message=f"Expected follow-up answer '312' but got '{completion['text']}'.")

def test_031_openai_reasoning():
    return _reasoning_completion(endpoint="OpenAI", model="gpt-6-luna", reasoning_effort="low", reasoning_summary="auto", stream=False)

def test_032_openai_reasoning_stream():
    return _reasoning_completion(endpoint="OpenAI", model="gpt-6-luna", reasoning_effort="low", reasoning_summary="auto", stream=True)

def test_033_openrouter_reasoning():
    return _reasoning_completion(endpoint="OpenRouter", model="~google/gemini-flash-latest", reasoning_effort="low", stream=False)

def test_034_openrouter_reasoning_stream():
    return _reasoning_completion(endpoint="OpenRouter", model="~google/gemini-flash-latest", reasoning_effort="low", stream=True)

def test_035_vllm_reasoning():
    success, message, model = _vllm_model()
    if not success:
        return message
    return _reasoning_completion(endpoint="vLLM", model=model, reasoning_effort="low", stream=False)

def test_036_vllm_reasoning_stream():
    success, message, model = _vllm_model()
    if not success:
        return message
    return _reasoning_completion(endpoint="vLLM", model=model, reasoning_effort="low", stream=True)

def test_037_openai_astra_reasoning():
    return _reasoning_completion(endpoint="OpenAI", model="gpt-6-astra", reasoning_effort="low")

def test_038_openai_reasoning_model_switch():
    for summary in [None, "auto"]:
        _reasoning_completion(endpoint="OpenAI", model="gpt-6-luna", reasoning_effort="low", reasoning_summary=summary, switch_settings={'model': "gpt-6-astra"})

def test_039_openai_openrouter_reasoning_switch():
    for summary in [None, "auto"]:
        _reasoning_completion(endpoint="OpenAI", model="gpt-6-luna", reasoning_effort="low", reasoning_summary=summary, switch_settings={'endpoint': "OpenRouter", 'model': "~google/gemini-flash-latest"})

def test_040_openrouter_openai_reasoning_switch():
    for summary in [None, "auto"]:
        _reasoning_completion(endpoint="OpenRouter", model="~google/gemini-flash-latest", reasoning_effort="low", reasoning_summary=summary, switch_settings={'endpoint': "OpenAI", 'model': "gpt-6-luna"})

# tool use

def _tool_use(switch_settings=None, **kwargs):
    client = Responses(kwargs)
    tools = [{
        'type': "function",
        'name': name,
        'description': f"Return the secret {name} value.",
        'parameters': {
            'type': "object",
            'properties': {},
            'additionalProperties': False
        },
        'strict': True
    } for name in ["get_number", "get_color"]]
    success, message = client.set_tools(tools)
    assert_type_value(obj=success, type_or_value=bool, name="success")
    assert_type_value(obj=message, type_or_value=str, name="message")
    assert_log(expression=success, message=message)
    for response_type in ["text", "auto", "get_number", "always"]:
        text = "Say hello without tools." if response_type == "text" else "Call get_number to retrieve the secret number. After receiving its result, report the number using decimal digits."
        success, message, completion = client.prompt(text, reset_context=True, response_type=response_type)
        assert_type_value(obj=success, type_or_value=bool, name="success")
        assert_type_value(obj=message, type_or_value=str, name="message")
        assert_log(expression=success, message=message)
        assert_type_value(obj=completion, type_or_value=dict, name="completion")
        assert_type_value(obj=completion['tools'], type_or_value=list, name="tool calls")
        assert_log(expression=bool(completion['tools']) == (response_type != "text"), message="Expected the requested tool-selection behavior.")
        if response_type == "get_number":
            assert_log(expression=all(call['name'] == "get_number" for call in completion['tools']), message="Expected only the named tool.")
            success, message, completion = client.prompt(["372"] * len(completion['tools']), response_type="auto")
            assert_type_value(obj=success, type_or_value=bool, name="success")
            assert_type_value(obj=message, type_or_value=str, name="message")
            assert_log(expression=success, message=message)
            assert_type_value(obj=completion, type_or_value=dict, name="completion")
            assert_log(expression=not completion['tools'] and "372" in completion['text'], message=f"Expected the string tool result without further calls but got '{completion['text']}' with tools '{completion['tools']}' and status '{completion['response']['status']}'.")
    success, message, completion = client.prompt("Call both get_number and get_color simultaneously. After receiving the results, report the number using decimal digits and the color by name.", reset_context=True, response_type="always", parallel_tool_calls=True)
    assert_type_value(obj=success, type_or_value=bool, name="success")
    assert_type_value(obj=message, type_or_value=str, name="message")
    assert_log(expression=success, message=message)
    assert_type_value(obj=completion, type_or_value=dict, name="completion")
    assert_log(expression={call['name'] for call in completion['tools']} == {tool['name'] for tool in tools}, message="Expected both parallel tools.")
    outputs = []
    for call in completion['tools']:
        assert_type_value(obj=call['call_id'], type_or_value=str, name="call ID")
        assert_log(expression=json.loads(call['arguments']) == {}, message="Expected empty function arguments.")
        outputs.append({
            'type': "function_call_output",
            'call_id': call['call_id'],
            'output': "372" if call['name'] == "get_number" else "blue"
        })
    success, message, awaited = client.get_awaited_tools()
    assert_type_value(obj=success, type_or_value=bool, name="success")
    assert_type_value(obj=message, type_or_value=str, name="message")
    assert_log(expression=success, message=message)
    assert_type_value(obj=awaited, type_or_value=list, name="awaited tools")
    assert_log(expression=awaited == [item['call_id'] for item in outputs], message="Expected pending parallel call IDs.")
    success, message, before = client.get_context()
    assert_type_value(obj=success, type_or_value=bool, name="success")
    assert_type_value(obj=message, type_or_value=str, name="message")
    assert_log(expression=success, message=message)
    assert_type_value(obj=before, type_or_value=list, name="context before tool outputs")
    success, message, completion = client.prompt(outputs, response_type="auto")
    assert_type_value(obj=success, type_or_value=bool, name="success")
    assert_type_value(obj=message, type_or_value=str, name="message")
    assert_log(expression=success, message=message)
    assert_type_value(obj=completion, type_or_value=dict, name="completion")
    assert_log(expression=not completion['tools'] and "372" in completion['text'] and "blue" in completion['text'].lower(), message=f"Expected both tool results without further calls but got '{completion['text']}' with tools '{completion['tools']}' and status '{completion['response']['status']}'.")
    success, message, context = client.get_context()
    assert_type_value(obj=success, type_or_value=bool, name="success")
    assert_type_value(obj=message, type_or_value=str, name="message")
    assert_log(expression=success, message=message)
    assert_type_value(obj=context, type_or_value=list, name="context")
    assert_log(expression=context == before + outputs + completion['response']['output'], message="Expected native tool-result history.")
    success, message, awaited = client.get_awaited_tools()
    assert_type_value(obj=success, type_or_value=bool, name="success")
    assert_type_value(obj=message, type_or_value=str, name="message")
    assert_log(expression=success, message=message)
    assert_type_value(obj=awaited, type_or_value=list, name="awaited tools")
    assert_log(expression=awaited == [], message="Expected all calls resolved.")
    if switch_settings is not None:
        success, message, tools_before = client.get_tools()
        assert_type_value(obj=success, type_or_value=bool, name="success")
        assert_type_value(obj=message, type_or_value=str, name="message")
        assert_log(expression=success, message=message)
        assert_type_value(obj=tools_before, type_or_value=list, name="tools before switching")
        completion = _switch_context(client, switch_settings, "Repeat the secret number and color returned by the tools earlier. Do not call tools again.")
        assert_log(expression=not completion['tools'] and "372" in completion['text'] and "blue" in completion['text'].lower(), message=f"Expected completed tool history reuse after switching but got '{completion['text']}' with calls '{completion['tools']}'.")
        success, message, tools_after = client.get_tools()
        assert_type_value(obj=success, type_or_value=bool, name="success")
        assert_type_value(obj=message, type_or_value=str, name="message")
        assert_log(expression=success, message=message)
        assert_log(expression=tools_after == tools_before, message="Expected provider switching to preserve tool definitions.")
        success, message, awaited = client.get_awaited_tools()
        assert_type_value(obj=success, type_or_value=bool, name="success")
        assert_type_value(obj=message, type_or_value=str, name="message")
        assert_log(expression=success, message=message)
        assert_log(expression=awaited == [], message="Expected no pending calls after switching with completed tool history.")

def test_041_openai_tool_use():
    return _tool_use(endpoint="OpenAI", model="gpt-6-luna", stream=False)

def test_042_openai_tool_use_stream():
    return _tool_use(endpoint="OpenAI", model="gpt-6-luna", stream=True)

def test_043_openrouter_tool_use():
    return _tool_use(endpoint="OpenRouter", model="~google/gemini-flash-latest", stream=False)

def test_044_openrouter_tool_use_stream():
    return _tool_use(endpoint="OpenRouter", model="~google/gemini-flash-latest", stream=True)

def test_045_vllm_tool_use():
    success, message, model = _vllm_model()
    if not success:
        return message
    return _tool_use(endpoint="vLLM", model=model, stream=False)

def test_046_vllm_tool_use_stream():
    success, message, model = _vllm_model()
    if not success:
        return message
    return _tool_use(endpoint="vLLM", model=model, stream=True)

def test_047_openai_openrouter_tool_switch():
    for stream in [False, True]:
        _tool_use(endpoint="OpenAI", model="gpt-6-luna", stream=stream, switch_settings={'endpoint': "OpenRouter", 'model': "~google/gemini-flash-latest"})

def test_048_openrouter_openai_tool_switch():
    for stream in [False, True]:
        _tool_use(endpoint="OpenRouter", model="~google/gemini-flash-latest", stream=stream, switch_settings={'endpoint': "OpenAI", 'model': "gpt-6-luna"})

# application tool variants

def _application_tool(tool, text, kind, output):
    client = Responses(model="gpt-6-astra")
    success, message = client.set_tools(tool)
    assert_type_value(obj=success, type_or_value=bool, name="success")
    assert_type_value(obj=message, type_or_value=str, name="message")
    assert_log(expression=success, message=message)
    success, message, result = client.prompt(text, response_type="always")
    assert_type_value(obj=success, type_or_value=bool, name="success")
    assert_type_value(obj=message, type_or_value=str, name="message")
    assert_log(expression=success, message=message)
    assert_type_value(obj=result, type_or_value=dict, name="result")
    assert_log(expression=bool(result['tools']) and all(call['type'] == kind for call in result['tools']), message=f"Expected '{kind}' calls.")
    outputs = []
    for call in result['tools']:
        item = copy.deepcopy(output)
        item['call_id'] = call['call_id']
        outputs.append(item)
    success, message, result = client.prompt(outputs, response_type="text")
    assert_type_value(obj=success, type_or_value=bool, name="success")
    assert_type_value(obj=message, type_or_value=str, name="message")
    assert_log(expression=success, message=message)
    assert_type_value(obj=result, type_or_value=dict, name="result")
    assert_log(expression=bool(result['text']) and client.get_awaited_tools()[2] == [], message="Expected tool-result continuation.")

def test_049_openai_custom_tool():
    return _application_tool(
        {'type': "custom", 'name': "get_number", 'description': "Return the secret number."},
        "Call get_number, then report its result.", "custom_tool_call",
        {'type': "custom_tool_call_output", 'output': "372"}
    )

def test_050_openai_computer_tool():
    return _application_tool(
        {'type': "computer"},
        "Take a screenshot, then describe it. Do not take any other actions.", "computer_call",
        {'type': "computer_call_output", 'output': {'type': "computer_screenshot", 'image_url': os.path.join(nimbro_api.__path__[0], "test", "assets", "test.png")}}
    )

def test_051_openai_shell_tool():
    return _application_tool(
        {'type': "shell", 'environment': {'type': "local"}},
        "Ask the test shell to run echo 372, then report its output.", "shell_call",
        {'type': "shell_call_output", 'output': [{'stdout': "372\n", 'stderr': "", 'outcome': {'type': "exit", 'exit_code': 0}}]}
    )

def test_052_openai_patch_tool():
    return _application_tool(
        {'type': "apply_patch"},
        "Request deletion of example.txt in the test workspace, then report the supplied operation status.", "apply_patch_call",
        {'type': "apply_patch_call_output", 'status': "completed", 'output': "Deleted example.txt in the test workspace."}
    )

# interruption

def _interrupt(**kwargs):
    client = Responses(validate_model=False, retry=10, **kwargs)
    results = []
    thread = threading.Thread(target=lambda: results.append(client.prompt("Write a long novel chapter about a computer scientist.")))
    thread.start()
    try:
        time.sleep(0.5)
        success, message = client.interrupt()
        assert_type_value(obj=success, type_or_value=bool, name="success")
        assert_type_value(obj=message, type_or_value=str, name="message")
        assert_log(expression=success, message=message)
        assert_log(expression="no response" not in message, message=f"Unexpected interrupt message: {message}")
    finally:
        client.interrupt()
        stamp = time.perf_counter()
        thread.join(1)
    assert_log(expression=time.perf_counter() - stamp < 0.1, message="Expected immediate thread termination after interruption.")
    assert_log(expression=not thread.is_alive() and len(results) == 1, message="Expected prompt termination after interruption.")
    success, message, completion = results[0]
    assert_type_value(obj=success, type_or_value=bool, name="success")
    assert_type_value(obj=message, type_or_value=str, name="message")
    assert_log(expression=not success, message="Expected interrupted generation to fail.")
    assert_log(expression="interrupted" in message.lower(), message=message)
    assert_type_value(obj=completion, type_or_value=None, name="completion")
    success, message, context = client.get_context()
    assert_type_value(obj=success, type_or_value=bool, name="success")
    assert_type_value(obj=message, type_or_value=str, name="message")
    assert_log(expression=success, message=message)
    assert_type_value(obj=context, type_or_value=list, name="context")
    assert_log(expression=context == [], message="Expected interruption not to commit partial history.")

def test_053_interrupt():
    return _interrupt(stream=False)

def test_054_interrupt_stream():
    return _interrupt(stream=True)

# web search

def _web_search(**kwargs):
    client = Responses(kwargs)
    flavor = client.get_settings('endpoint')
    success, message = client.set_tools({'type': "openrouter:web_search" if flavor == "OpenRouter" else "web_search"})
    assert_type_value(obj=success, type_or_value=bool, name="success")
    assert_type_value(obj=message, type_or_value=str, name="message")
    assert_log(expression=success, message=message)
    success, message, completion = client.prompt("Search for the official OpenAI Responses API documentation.", response_type="always")
    assert_type_value(obj=success, type_or_value=bool, name="success")
    assert_type_value(obj=message, type_or_value=str, name="message")
    assert_log(expression=success, message=message)
    assert_type_value(obj=completion, type_or_value=dict, name="completion")
    output = completion['response']['output']
    citations = [annotation for item in output if item.get('type') == "message" for part in item.get('content', []) for annotation in part.get('annotations', []) if annotation.get('type') == "url_citation"]
    searches = (completion['response'].get('usage') or {}).get('server_tool_use') or {}
    assert_log(expression=bool(citations) or searches.get('web_search_requests', 0) > 0 or any(item.get('type') == "web_search_call" for item in output), message="Expected native search evidence.")

def test_055_openai_web_search():
    return _web_search(endpoint="OpenAI", model="gpt-6-luna")

def test_056_openrouter_web_search():
    return _web_search(endpoint="OpenRouter", model="openai/gpt-6-luna")

# media input

def _media_input(part, switch_settings=None, **kwargs):
    client = Responses(kwargs)
    success, message, completion = client.prompt({
        'role': "user",
        'content': [
            {'type': "input_text", 'text': "Describe this input briefly."},
            part
        ]
    })
    assert_type_value(obj=success, type_or_value=bool, name="success")
    assert_type_value(obj=message, type_or_value=str, name="message")
    assert_log(expression=success, message=message)
    assert_type_value(obj=completion, type_or_value=dict, name="completion")
    assert_type_value(obj=completion['text'], type_or_value=str, name="description")
    assert_log(expression=bool(completion['text']), message="Expected a media description.")
    if switch_settings is not None:
        completion = _switch_context(client, switch_settings, "Describe the original media input again, referring to its contents rather than just the previous description.")
        assert_type_value(obj=completion['text'], type_or_value=str, name="media description after switching")
        assert_log(expression=bool(completion['text'].strip()), message="Expected a media description after switching providers.")

def _image_file_input(**kwargs):
    return _media_input({'type': "input_image", 'image_url': os.path.join(nimbro_api.__path__[0], "test", "assets", "test.png")}, **kwargs)

def _image_url_input(**kwargs):
    return _media_input({'type': "input_image", 'image_url': "https://www.ais.uni-bonn.de/nimbro/@Home/images/RC24/RoboCup_2024_NimbRo_Home_Final_4__07_21.jpg"}, download_image=False, **kwargs)

def test_057_openai_image_file_input():
    return _image_file_input(endpoint="OpenAI", model="gpt-6-luna")

def test_058_openrouter_image_file_input():
    return _image_file_input(endpoint="OpenRouter", model="~google/gemini-flash-latest")

def test_059_vllm_image_file_input():
    success, message, model = _vllm_model()
    if not success:
        return message
    return _image_file_input(endpoint="vLLM", model=model)

def test_060_openai_image_url_input():
    return _image_url_input(endpoint="OpenAI", model="gpt-6-luna")

def test_061_openrouter_image_url_input():
    return _image_url_input(endpoint="OpenRouter", model="~google/gemini-flash-latest")

def test_062_vllm_image_url_input():
    success, message, model = _vllm_model()
    if not success:
        return message
    return _image_url_input(endpoint="vLLM", model=model)

def test_063_openai_openrouter_image_switch():
    for stream in [False, True]:
        _image_file_input(endpoint="OpenAI", model="gpt-6-luna", stream=stream, switch_settings={'endpoint': "OpenRouter", 'model': "~google/gemini-flash-latest"})

def test_064_openrouter_openai_image_switch():
    for stream in [False, True]:
        _image_file_input(endpoint="OpenRouter", model="~google/gemini-flash-latest", stream=stream, switch_settings={'endpoint': "OpenAI", 'model': "gpt-6-luna"})

def test_065_openrouter_audio_file_input():
    part = {
        'type': "input_audio",
        'input_audio': {
            'data': os.path.join(nimbro_api.__path__[0], "test", "assets", "test.wav"),
            'format': "wav"
        }
    }
    return _media_input(part, endpoint="OpenRouter", model="~google/gemini-flash-latest")

def test_066_openrouter_video_file_input():
    return _media_input({'type': "input_video", 'video_url': os.path.join(nimbro_api.__path__[0], "test", "assets", "test.mp4")}, endpoint="OpenRouter", model="~google/gemini-flash-latest")

def test_067_openrouter_video_url_input():
    return _media_input({'type': "input_video", 'video_url': "https://www.youtube.com/watch?v=dQw4w9WgXcQ"}, endpoint="OpenRouter", model="~google/gemini-flash-latest", download_video=False)

def _file_local_input(**kwargs):
    return _media_input({'type': "input_file", 'file_data': os.path.join(nimbro_api.__path__[0], "test", "assets", "test.pdf")}, **kwargs)

def test_068_openai_file_local_input():
    return _file_local_input(endpoint="OpenAI", model="gpt-6-luna")

def test_069_openrouter_file_local_input():
    return _file_local_input(endpoint="OpenRouter", model="~google/gemini-flash-latest")

def test_070_openrouter_file_url_input():
    return _media_input({'type': "input_file", 'file_url': "https://bitcoinpaper.org/bitcoin.pdf"}, endpoint="OpenRouter", model="~google/gemini-flash-latest", download_file=False)

def test_071_openai_openrouter_file_switch():
    for stream in [False, True]:
        _file_local_input(endpoint="OpenAI", model="gpt-6-luna", stream=stream, switch_settings={'endpoint': "OpenRouter", 'model': "~google/gemini-flash-latest"})

def test_072_openrouter_openai_file_switch():
    for stream in [False, True]:
        _file_local_input(endpoint="OpenRouter", model="~google/gemini-flash-latest", stream=stream, switch_settings={'endpoint': "OpenAI", 'model': "gpt-6-luna"})

# concurrency

def _process_worker(index):
    client = Responses(logger_severity="off")
    success, message, completion = client.prompt(f"Reply with the number {index}.")
    assert_type_value(obj=success, type_or_value=bool, name="success")
    assert_type_value(obj=message, type_or_value=str, name="message")
    assert_log(expression=success, message=message)
    assert_type_value(obj=completion, type_or_value=dict, name="completion")
    assert_log(expression=str(index) in completion['text'], message="Expected each worker's own answer.")
    return index

def test_073_parallel_threads():
    results = []
    errors = []

    def worker(index):
        try:
            results.append(_process_worker(index))
        except Exception as error:
            errors.append(str(error))

    threads = [threading.Thread(target=worker, args=(index,)) for index in range(8)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert_log(expression=not errors, message=f"Concurrent request failures: {errors}")
    assert_log(expression=sorted(results) == list(range(8)), message="Expected all threaded requests to complete.")

def test_074_parallel_processes():
    with multiprocessing.get_context("spawn").Pool(4) as pool:
        results = pool.map(_process_worker, range(8))
    assert_log(expression=results == list(range(8)), message="Expected all process requests to complete.")

# service tiers

def _service_tier_completion(**kwargs):
    assert_log(expression='service_tier' in kwargs, message="Expected the test definition to supply 'service_tier'.")
    client = Responses(kwargs)
    success, message, completion = client.prompt("Reply with one short greeting.")
    assert_type_value(obj=success, type_or_value=bool, name="success")
    assert_type_value(obj=message, type_or_value=str, name="message")
    assert_log(expression=success, message=message)
    assert_type_value(obj=completion, type_or_value=dict, name="completion")
    assert_type_value(obj=completion['text'], type_or_value=str, name="completion text")
    assert_log(expression=bool(completion['text']), message="Expected text with the selected service tier.")

def test_075_openai_service_tier_none():
    return _service_tier_completion(endpoint="OpenAI", model="gpt-6-luna", service_tier=None, stream=False)

def test_076_openai_service_tier_none_stream():
    return _service_tier_completion(endpoint="OpenAI", model="gpt-6-luna", service_tier=None, stream=True)

def test_077_openrouter_service_tier_none():
    return _service_tier_completion(endpoint="OpenRouter", model="~google/gemini-flash-latest", service_tier=None, stream=False)

def test_078_openrouter_service_tier_none_stream():
    return _service_tier_completion(endpoint="OpenRouter", model="~google/gemini-flash-latest", service_tier=None, stream=True)

def test_079_vllm_service_tier_none():
    success, message, model = _vllm_model()
    if not success:
        return message
    return _service_tier_completion(endpoint="vLLM", model=model, service_tier=None, stream=False)

def test_080_vllm_service_tier_none_stream():
    success, message, model = _vllm_model()
    if not success:
        return message
    return _service_tier_completion(endpoint="vLLM", model=model, service_tier=None, stream=True)

def test_081_openai_service_tier_price():
    return _service_tier_completion(endpoint="OpenAI", model="gpt-6-luna", reasoning_effort="none", service_tier="price", stream=False)

def test_082_openai_service_tier_price_stream():
    return _service_tier_completion(endpoint="OpenAI", model="gpt-6-luna", reasoning_effort="none", service_tier="price", stream=True, timeout_response=20, timeout_chunk_first=20, timeout_chunk_next=20)

def test_083_openrouter_service_tier_price():
    return _service_tier_completion(endpoint="OpenRouter", model="~google/gemini-flash-latest", service_tier="price", stream=False)

def test_084_openrouter_service_tier_price_stream():
    return _service_tier_completion(endpoint="OpenRouter", model="~google/gemini-flash-latest", service_tier="price", stream=True)

def test_085_vllm_service_tier_price():
    success, message, model = _vllm_model()
    if not success:
        return message
    return _service_tier_completion(endpoint="vLLM", model=model, service_tier="price", stream=False)

def test_086_vllm_service_tier_price_stream():
    success, message, model = _vllm_model()
    if not success:
        return message
    return _service_tier_completion(endpoint="vLLM", model=model, service_tier="price", stream=True)

def test_087_openai_service_tier_latency():
    return _service_tier_completion(endpoint="OpenAI", model="gpt-6-luna", service_tier="latency", stream=False)

def test_088_openai_service_tier_latency_stream():
    return _service_tier_completion(endpoint="OpenAI", model="gpt-6-luna", service_tier="latency", stream=True)

def test_089_openrouter_service_tier_latency():
    return _service_tier_completion(endpoint="OpenRouter", model="~google/gemini-flash-latest", service_tier="latency", stream=False)

def test_090_openrouter_service_tier_latency_stream():
    return _service_tier_completion(endpoint="OpenRouter", model="~google/gemini-flash-latest", service_tier="latency", stream=True)

def test_091_vllm_service_tier_latency():
    success, message, model = _vllm_model()
    if not success:
        return message
    return _service_tier_completion(endpoint="vLLM", model=model, service_tier="latency", stream=False)

def test_092_vllm_service_tier_latency_stream():
    success, message, model = _vllm_model()
    if not success:
        return message
    return _service_tier_completion(endpoint="vLLM", model=model, service_tier="latency", stream=True)

def test_093_openai_service_tier_throughput():
    return _service_tier_completion(endpoint="OpenAI", model="gpt-6-luna", service_tier="throughput", stream=False)

def test_094_openai_service_tier_throughput_stream():
    return _service_tier_completion(endpoint="OpenAI", model="gpt-6-luna", service_tier="throughput", stream=True)

def test_095_openrouter_service_tier_throughput():
    return _service_tier_completion(endpoint="OpenRouter", model="~google/gemini-flash-latest", service_tier="throughput", stream=False)

def test_096_openrouter_service_tier_throughput_stream():
    return _service_tier_completion(endpoint="OpenRouter", model="~google/gemini-flash-latest", service_tier="throughput", stream=True)

def test_097_vllm_service_tier_throughput():
    success, message, model = _vllm_model()
    if not success:
        return message
    return _service_tier_completion(endpoint="vLLM", model=model, service_tier="throughput", stream=False)

def test_098_vllm_service_tier_throughput_stream():
    success, message, model = _vllm_model()
    if not success:
        return message
    return _service_tier_completion(endpoint="vLLM", model=model, service_tier="throughput", stream=True)

# context compaction

def test_099_openai_compact():
    client = Responses(model="gpt-6-luna")
    success, message, completion = client.prompt("Remember the number 372. Reply briefly.")
    assert_type_value(obj=success, type_or_value=bool, name="success")
    assert_type_value(obj=message, type_or_value=str, name="message")
    assert_log(expression=success, message=message)
    assert_type_value(obj=completion, type_or_value=dict, name="completion")
    success, message, before = client.get_context()
    assert_type_value(obj=success, type_or_value=bool, name="success")
    assert_type_value(obj=message, type_or_value=str, name="message")
    assert_log(expression=success, message=message)
    assert_type_value(obj=before, type_or_value=list, name="context before compaction")
    success, message, compacted = client.compact(before)
    assert_type_value(obj=success, type_or_value=bool, name="success")
    assert_type_value(obj=message, type_or_value=str, name="message")
    assert_log(expression=success, message=message)
    assert_type_value(obj=compacted, type_or_value=dict, name="compaction response")
    assert_log(expression=compacted['object'] == "response.compaction", message="Expected native compaction.")
    success, message, context = client.get_context()
    assert_type_value(obj=success, type_or_value=bool, name="success")
    assert_type_value(obj=message, type_or_value=str, name="message")
    assert_log(expression=success, message=message)
    assert_type_value(obj=context, type_or_value=list, name="context")
    assert_log(expression=context == before, message="Expected explicit compaction not to change history.")
    success, message = client.set_context(messages=compacted['output'])
    assert_type_value(obj=success, type_or_value=bool, name="success")
    assert_type_value(obj=message, type_or_value=str, name="message")
    assert_log(expression=success, message=message)
    success, message, completion = client.prompt("What number did I give you?")
    assert_type_value(obj=success, type_or_value=bool, name="success")
    assert_type_value(obj=message, type_or_value=str, name="message")
    assert_log(expression=success, message=message)
    assert_type_value(obj=completion, type_or_value=dict, name="completion")
    assert_log(expression="372" in completion['text'], message="Expected compacted history reuse.")

# native context and tool results

def test_100_utilities_native_context():
    client = Responses(retry=False)
    endpoint = client.get_settings()['endpoint']
    endpoint['api_flavor'] = "openresponses"
    success, message = client.set_settings(endpoint=endpoint, presence_penalty=0, frequency_penalty=0)
    assert_type_value(obj=success, type_or_value=bool, name="success")
    assert_type_value(obj=message, type_or_value=str, name="message")
    assert_log(expression=success, message=message)
    items = [
        {'role': "user", 'content': "Remember 372."},
        {
            'type': "reasoning",
            'encrypted_content': "opaque",
            'summary': []
        },
        {
            'type': "message",
            'role': "assistant",
            'phase': "final_answer",
            'content': [{'type': "output_text", 'text': "372"}]
        }
    ]
    before = copy.deepcopy(items)
    success, message = client.set_context(messages=items)
    assert_type_value(obj=success, type_or_value=bool, name="success")
    assert_type_value(obj=message, type_or_value=str, name="message")
    assert_log(expression=success, message=message)
    success, message, context = client.get_context()
    assert_type_value(obj=success, type_or_value=bool, name="success")
    assert_type_value(obj=message, type_or_value=str, name="message")
    assert_log(expression=success, message=message)
    assert_type_value(obj=context, type_or_value=list, name="native context")
    assert_log(expression=context[0]['type'] == "message" and context[1:] == items[1:], message="Expected Open Responses normalization and preserved opaque native items.")
    assert_log(expression=items == before, message="Expected caller-owned context not to change.")

def test_101_utilities_tool_results():
    client = Responses(retry=False)
    calls = [
        {'type': "function_call", 'call_id': "partial", 'name': "lookup", 'arguments': "{", 'status': "incomplete"},
        {'type': "function_call", 'call_id': "async", 'name': "lookup", 'arguments': "{}", 'async': True},
        {'type': "function_call", 'call_id': "program", 'name': "lookup", 'arguments': "{}", 'caller': {'type': "program", 'caller_id': "parent"}},
        {'type': "computer_call", 'call_id': "computer", 'actions': [{'type': "screenshot"}]},
        {'type': "custom_tool_call", 'call_id': "custom", 'name': "echo", 'input': "hello"},
        {'type': "shell_call", 'call_id': "shell", 'action': {'commands': ["echo hello"]}},
        {'type': "apply_patch_call", 'call_id': "patch", 'operation': {'type': "delete_file", 'path': "example.txt"}}
    ]
    success, message = client.set_context(messages=calls)
    assert_type_value(obj=success, type_or_value=bool, name="success")
    assert_type_value(obj=message, type_or_value=str, name="message")
    assert_log(expression=success, message=message)
    outputs = [
        ({'type': "custom_tool_call_output", 'call_id': "async", 'output': "wrong type"}, ["async", "program", "computer", "custom", "shell", "patch"]),
        ({'type': "function_call_output", 'name': "lookup", 'caller': {'type': "program", 'caller_id': "parent"}, 'output': "372"}, ["async", "computer", "custom", "shell", "patch"]),
        ({'type': "computer_call_output", 'call_id': "computer", 'output': {'type': "computer_screenshot", 'file_id': "file_1", 'image_url': None}}, ["async", "custom", "shell", "patch"]),
        ({'type': "function_call_output", 'call_id': "async", 'output': "372"}, ["custom", "shell", "patch"]),
        ("hello", ["shell", "patch"]),
        ({'type': "shell_call_output", 'call_id': "shell", 'output': [{'stdout': "hello", 'stderr': "", 'outcome': {'type': "exit", 'exit_code': 0}}]}, ["patch"]),
        ({'type': "apply_patch_call_output", 'call_id': "patch", 'status': "completed"}, [])
    ]
    for output, expected in outputs:
        success, message, result = client.prompt(output, response_type="none")
        assert_type_value(obj=success, type_or_value=bool, name="success")
        assert_type_value(obj=message, type_or_value=str, name="message")
        assert_log(expression=success, message=message)
        assert_type_value(obj=result, type_or_value=None, name="result")
        success, message, awaited = client.get_awaited_tools()
        assert_type_value(obj=success, type_or_value=bool, name="success")
        assert_type_value(obj=message, type_or_value=str, name="message")
        assert_log(expression=success, message=message)
        assert_log(expression=awaited == expected, message=f"Expected pending tools '{expected}' but got '{awaited}'.")
    calls[0]['arguments'] = "changed"
    success, message, context = client.get_context()
    assert_type_value(obj=success, type_or_value=bool, name="success")
    assert_type_value(obj=message, type_or_value=str, name="message")
    assert_log(expression=success, message=message)
    assert_log(expression=context[0]['arguments'] == "{", message="Expected incomplete calls to be preserved without sharing caller data.")

# optional response data

def _include_response(client, text, response_type="auto"):
    success, message, completion = client.prompt(text, response_type=response_type)
    assert_type_value(obj=success, type_or_value=bool, name="success")
    assert_type_value(obj=message, type_or_value=str, name="message")
    assert_log(expression=success, message=message)
    assert_type_value(obj=completion, type_or_value=dict, name="completion")
    success, message, context = client.get_context()
    assert_type_value(obj=success, type_or_value=bool, name="success")
    assert_type_value(obj=message, type_or_value=str, name="message")
    assert_log(expression=success, message=message)
    output = completion['response']['output']
    assert_log(expression=bool(output) and context[-len(output):] == output, message="Expected extra output data preserved in context.")
    return completion, context

def _include_reasoning(endpoint, field):
    client = Responses(endpoint=endpoint, model="gpt-6-luna" if endpoint == "OpenAI" else "openai/gpt-6-luna", reasoning_effort="low", include=[field])
    completion, _ = _include_response(client, "Find the smallest positive integer n with n modulo 17 = 5 and n modulo 19 = 7. Reply with only n.")
    assert_log(expression="311" in completion['text'], message=f"Expected initial answer '311' but got '{completion['text']}'.")
    encrypted = [item for item in completion['response']['output'] if item.get('encrypted_content')]
    assert_log(expression=bool(encrypted), message=f"Expected encrypted output with include '{field}'.")
    for item in encrypted:
        assert_type_value(obj=item['encrypted_content'], type_or_value=str, name="encrypted content returned by the provider")
        assert_log(expression=bool(item['encrypted_content'].strip()), message="Expected nonempty returned encrypted content.")
    if field == "reasoning.encrypted_content":
        assert_log(expression=any(item.get('type') == "reasoning" for item in encrypted), message="Expected encrypted reasoning state.")
    followup, _ = _include_response(client, "Add one to your answer. Output only the final integer in digits without separators.")
    assert_log(expression="312" in followup['text'], message=f"Expected follow-up answer '312' using replayed history but got '{followup['text']}'.")

def test_102_openai_include_reasoning():
    return _include_reasoning("OpenAI", "reasoning.encrypted_content")

def test_103_openrouter_include_reasoning():
    return _include_reasoning("OpenRouter", "reasoning.encrypted_content")

def test_104_openrouter_include_encrypted_output():
    return _include_reasoning("OpenRouter", "output.encrypted_content")

def _include_file_search(endpoint):
    # Supply a populated vector store belonging to the selected provider/account.
    variable = "OPENAI_TEST_VECTOR_STORE_ID" if endpoint == "OpenAI" else "OPENROUTER_TEST_VECTOR_STORE_ID"
    vector_store_id = os.getenv(variable)
    if not vector_store_id:
        return f"Set '{variable}' to a populated vector store to test included file-search results."
    client = Responses(endpoint=endpoint, model="gpt-6-luna" if endpoint == "OpenAI" else "openai/gpt-6-luna", include=["file_search_call.results"])
    success, message = client.set_tools({'type': "file_search", 'vector_store_ids': [vector_store_id]})
    assert_type_value(obj=success, type_or_value=bool, name="success")
    assert_type_value(obj=message, type_or_value=str, name="message")
    assert_log(expression=success, message=message)
    completion, _ = _include_response(client, "Search the available documents and quote one passage from a document.", "always")
    results = [result for item in completion['response']['output'] if item.get('type') == "file_search_call" for result in item.get('results') or []]
    assert_log(expression=bool(results) and any(result.get('text') for result in results), message="Expected included retrieved document passages.")

def test_105_openai_include_file_search():
    return _include_file_search("OpenAI")

def test_106_openrouter_include_file_search():
    return _include_file_search("OpenRouter")

def test_107_openai_include_code_outputs():
    client = Responses(endpoint="OpenAI", model="gpt-6-luna", include=["code_interpreter_call.outputs"])
    success, message = client.set_tools({'type': "code_interpreter", 'container': {'type': "auto"}})
    assert_type_value(obj=success, type_or_value=bool, name="success")
    assert_type_value(obj=message, type_or_value=str, name="message")
    assert_log(expression=success, message=message)
    completion, _ = _include_response(client, "Use Python to execute print(372).", "always")
    outputs = [output for item in completion['response']['output'] if item.get('type') == "code_interpreter_call" for output in item.get('outputs') or []]
    assert_log(expression=any(output.get('type') == "logs" and "372" in output.get('logs', "") for output in outputs), message=f"Expected included code-interpreter logs but got calls '{[item for item in completion['response']['output'] if item.get('type') == 'code_interpreter_call']}'.")

def _include_web_search(field):
    client = Responses(include=[field])
    success, message = client.set_tools({'type': "web_search"})
    assert_type_value(obj=success, type_or_value=bool, name="success")
    assert_type_value(obj=message, type_or_value=str, name="message")
    assert_log(expression=success, message=message)
    completion, _ = _include_response(client, "Search the web for the official OpenAI Responses API documentation.", "always")
    calls = [item for item in completion['response']['output'] if item.get('type') == "web_search_call"]
    if field == "web_search_call.action.sources":
        assert_log(expression=any(source.get('url') for call in calls for source in call.get('action', {}).get('sources') or []), message="Expected included web-search source URLs.")
    else:
        assert_log(expression=any(call.get('results') for call in calls), message="Expected included web-search results.")

def test_108_openai_include_web_results():
    return _include_web_search("web_search_call.results")

def test_109_openai_include_web_sources():
    return _include_web_search("web_search_call.action.sources")

def test_110_openai_openrouter_reasoning_format():
    for stream in [False, True]:
        _reasoning_completion(endpoint="OpenAI", model="gpt-6-luna", reasoning_effort="low", reasoning_summary="auto", stream=stream,
                              switch_settings={'endpoint': "OpenRouter", 'model': "anthropic/claude-sonnet-5"})

def test_111_openrouter_claude_readable_reasoning_switch():
    return _reasoning_completion(endpoint="OpenRouter", model="anthropic/claude-sonnet-5", reasoning_effort="low", stream=True,
                                 switch_settings={'endpoint': "OpenAI", 'model': "gpt-6-luna"})

def test_112_openrouter_send_reasoning():
    client = Responses(endpoint="OpenRouter", model="google/gemini-3.7-flash", reasoning_effort="low")
    success, message, completion = client.prompt("Remember 372 and blue. Explain in one sentence why 372 is divisible by 3.")
    assert_type_value(obj=success, type_or_value=bool, name="success")
    assert_type_value(obj=message, type_or_value=str, name="message")
    assert_log(expression=success, message=message)
    assert_type_value(obj=completion, type_or_value=dict, name="completion")
    assert_log(expression=bool(completion['reasoning']), message="Expected native reasoning to exercise omission.")
    success, message, before = client.get_context()
    assert_log(expression=success, message=message)
    success, message, completion = client.prompt("Repeat the number and color.", endpoint="OpenAI", model="gpt-6-luna", send_reasoning=False)
    assert_log(expression=success, message=message)
    assert_log(expression="372" in completion['text'] and "blue" in completion['text'].lower(), message="Expected retained facts with reasoning replay disabled.")
    success, message, context = client.get_context()
    assert_log(expression=success, message=message)
    assert_log(expression=context[:len(before)] == before, message="Expected native reasoning history to survive omission.")
    assert_log(expression=client.get_settings('send_reasoning') is True, message="Expected temporary reasoning override to be restored.")
    for send_reasoning in [True, False]:
        success, message, compacted = client.compact(input_items=before, endpoint="OpenAI", model="gpt-6-luna", send_reasoning=send_reasoning)
        assert_type_value(obj=success, type_or_value=bool, name="success")
        assert_type_value(obj=message, type_or_value=str, name="message")
        assert_log(expression=success, message=message)
        assert_type_value(obj=compacted, type_or_value=dict, name="compacted response")
        success, message, after = client.get_context()
        assert_log(expression=success, message=message)
        assert_log(expression=after == context, message="Expected compaction to preserve local history.")

def test_113_openrouter_parallel_tool_roundtrip():
    client = Responses(endpoint="OpenRouter", model="google/gemini-3.7-flash", reasoning_effort="low")
    tools = [{'type': "function", 'name': name, 'description': f"Retrieve secret {name}.",
              'parameters': {'type': "object", 'properties': {}, 'additionalProperties': False}, 'strict': True}
             for name in ["number", "color"]]
    success, message = client.set_tools(tools)
    assert_log(expression=success, message=message)
    success, message, completion = client.prompt("Call both number and color in parallel.", response_type="always")
    assert_log(expression=success, message=message)
    assert_log(expression={call['name'] for call in completion['tools']} == {"number", "color"}, message="Expected both parallel calls.")
    outputs = [{'type': "function_call_output", 'call_id': call['call_id'], 'output': "372" if call['name'] == "number" else "blue"}
               for call in completion['tools']]
    success, message, before = client.get_context()
    assert_log(expression=success, message=message)
    success, message, completion = client.prompt(outputs, endpoint="OpenAI", model="gpt-6-astra", response_type="text")
    assert_log(expression=success, message=message)
    assert_log(expression="372" in completion['text'] and "blue" in completion['text'].lower(), message="Expected both tool results after switching.")
    success, message, context = client.get_context()
    assert_log(expression=success, message=message)
    assert_log(expression=context == before + outputs + completion['response']['output'], message="Expected native mixed history after tool handoff.")
    success, message, completion = client.prompt("Repeat both secret values without calling tools. Reply with only the number and color.", response_type="text")
    assert_log(expression=success, message=message)
    assert_log(expression="372" in completion['text'] and "blue" in completion['text'].lower(), message="Expected retained tool results after returning to the original provider.")

def test_114_openrouter_deepseek_readable_reasoning_switch():
    return _reasoning_completion(endpoint="OpenRouter", model="deepseek/deepseek-v4-pro-0813", reasoning_effort="low", stream=True,
                                 switch_settings={'endpoint': "OpenAI", 'model': "gpt-6-luna"})

def test_115_openrouter_qwen_readable_reasoning_switch():
    return _reasoning_completion(endpoint="OpenRouter", model="qwen/qwen3.8-flash", reasoning_effort="low", stream=True,
                                 switch_settings={'endpoint': "OpenAI", 'model': "gpt-6-luna"})

def test_116_openrouter_gemini_readable_reasoning_switch():
    return _reasoning_completion(endpoint="OpenRouter", model="google/gemini-3.7-flash", reasoning_effort="low", stream=True,
                                 switch_settings={'endpoint': "OpenAI", 'model': "gpt-6-luna"})
