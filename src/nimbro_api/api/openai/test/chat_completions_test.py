import os
import json
import math
import time
import threading
import multiprocessing

import nimbro_api
from nimbro_api.utility.misc import UnrecoverableError, assert_type_value, assert_keys, assert_log
from ..client.chat_completions import ChatCompletions

def test_01_utilities():
    client = ChatCompletions()

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

def test_02_endpoint():
    client = ChatCompletions()

    success, message, models = client.get_models()
    assert_type_value(obj=success, type_or_value=bool, name="success")
    assert_type_value(obj=message, type_or_value=str, name="message")
    assert_log(expression=success, message=message)
    assert_type_value(obj=models, type_or_value=list, name="result of get_models()")

# text completions

def _text_completions(**kwargs):
    client = ChatCompletions(kwargs)

    # text completion
    text = "Hi!"
    success, message, completion = client.prompt(text=text, response_type="text")
    assert_type_value(obj=success, type_or_value=bool, name="success")
    assert_type_value(obj=message, type_or_value=str, name="message")
    assert_log(expression=success, message=message)
    assert_type_value(obj=completion, type_or_value=dict, name="completion")
    assert_keys(obj=completion, keys=['usage', 'text', 'logs'], mode="match", name="completion")
    assert_type_value(obj=completion['usage'], type_or_value=dict, name="usage in completion")
    assert_type_value(obj=completion['text'], type_or_value=str, name="text in completion")
    assert_log(expression=len(completion['text']) > 0, message="Expected text in completion to be non-empty string.")
    assert_type_value(obj=completion['logs'], type_or_value=list, name="logs in completion")
    assert_log(expression=len(completion['logs']) > 0, message="Expected to find at least one log in completion.")
    for i, log in enumerate(completion['logs']):
        assert_type_value(obj=log, type_or_value=str, name=f"log '{i}' in completion logs")

    # inspect context
    success, message, context = client.get_context()
    assert_type_value(obj=success, type_or_value=bool, name="success")
    assert_type_value(obj=message, type_or_value=str, name="message")
    assert_log(expression=success, message=message)
    assert_type_value(obj=context, type_or_value=list, name="context")
    assert_log(expression=len(context) == 2, message=f"Expected context to contain '2' messages but got '{len(context)}'.")
    for i, message in enumerate(context):
        assert_type_value(obj=message, type_or_value=dict, name=f"message '{i}' in context")
    target = {'role': 'user', 'content': [{'type': 'text', 'text': text}]}
    assert_log(expression=context[0] == target, message=f"Expected message '0' in context to be {target} but got {context[0]}.")
    target = {'role': "assistant", 'content': completion['text']}
    assert_log(expression=context[1] == target, message=f"Expected message '1' in context to be {target} but got {context[1]}.")

    # auto completion
    success, message, completion = client.prompt(text=text, reset_context=True, response_type="auto")
    assert_type_value(obj=success, type_or_value=bool, name="success")
    assert_type_value(obj=message, type_or_value=str, name="message")
    assert_log(expression=success, message=message)
    assert_type_value(obj=completion, type_or_value=dict, name="completion")
    assert_keys(obj=completion, keys=['usage', 'text', 'logs'], mode="match", name="completion")
    assert_type_value(obj=completion['usage'], type_or_value=dict, name="usage in completion")
    assert_type_value(obj=completion['text'], type_or_value=str, name="text in completion")
    assert_log(expression=len(completion['text']) > 0, message="Expected text in completion to be non-empty string.")
    assert_type_value(obj=completion['logs'], type_or_value=list, name="logs in completion")
    assert_log(expression=len(completion['logs']) > 0, message="Expected to find at least one log in completion.")
    for i, log in enumerate(completion['logs']):
        assert_type_value(obj=log, type_or_value=str, name=f"log '{i}' in completion logs")

    # inspect context
    success, message, context = client.get_context()
    assert_type_value(obj=success, type_or_value=bool, name="success")
    assert_type_value(obj=message, type_or_value=str, name="message")
    assert_log(expression=success, message=message)
    assert_type_value(obj=context, type_or_value=list, name="context")
    assert_log(expression=len(context) == 2, message=f"Expected context to contain '2' messages but got '{len(context)}'.")
    for i, message in enumerate(context):
        assert_type_value(obj=message, type_or_value=dict, name=f"message '{i}' in context")
    target = {'role': 'user', 'content': [{'type': 'text', 'text': text}]}
    assert_log(expression=context[0] == target, message=f"Expected message '0' in context to be {target} but got {context[0]}.")
    target = {'role': "assistant", 'content': completion['text']}
    assert_log(expression=context[1] == target, message=f"Expected message '1' in context to be {target} but got {context[1]}.")

    # JSON mode completion
    new_text = "Tell me a joke in JSON format!"
    success, message, completion = client.prompt(text=new_text, response_type="json")
    assert_type_value(obj=success, type_or_value=bool, name="success")
    assert_type_value(obj=message, type_or_value=str, name="message")
    assert_log(expression=success, message=message)
    assert_type_value(obj=completion, type_or_value=dict, name="completion")
    assert_keys(obj=completion, keys=['usage', 'text', 'logs'], mode="match", name="completion")
    assert_type_value(obj=completion['usage'], type_or_value=dict, name="usage in completion")
    assert_type_value(obj=completion['text'], type_or_value=dict, name="text in completion")
    assert_type_value(obj=completion['logs'], type_or_value=list, name="logs in completion")
    assert_log(expression=len(completion['logs']) > 0, message="Expected to find at least one log in completion.")
    for i, log in enumerate(completion['logs']):
        assert_type_value(obj=log, type_or_value=str, name=f"log '{i}' in completion logs")

    # inspect context
    success, message, context = client.get_context()
    assert_type_value(obj=success, type_or_value=bool, name="success")
    assert_type_value(obj=message, type_or_value=str, name="message")
    assert_log(expression=success, message=message)
    assert_type_value(obj=context, type_or_value=list, name="context")
    assert_log(expression=len(context) == 4, message=f"Expected context to contain '4' messages but got '{len(context)}'.")
    for i, message in enumerate(context):
        assert_type_value(obj=message, type_or_value=dict, name=f"message '{i}' in context")
    assert_log(expression=context[1] == target, message=f"Expected message '1' in context to be {target} but got {context[1]}.")
    target = {'role': 'user', 'content': [{'type': 'text', 'text': text}]}
    assert_log(expression=context[0] == target, message=f"Expected message '0' in context to be {target} but got {context[0]}.")
    target = {'role': 'user', 'content': [{'type': 'text', 'text': new_text}]}
    assert_log(expression=context[2] == target, message=f"Expected message '2' in context to be {target} but got {context[2]}.")
    assert_keys(obj=context[3], keys=['role', 'content'], mode="match", name="message '3' in context")
    try:
        context[3]['content'] = json.loads(context[3]['content'])
    except Exception as e:
        raise UnrecoverableError(f"Expected message '3' in context to be JSON-compliant but '{context[3]}' is not: {repr(e)}") from e
    target = {'role': "assistant", 'content': completion['text']}
    assert_log(expression=context[3] == target, message=f"Expected message '3' in context to be {target} but got {context[3]}.")

    # reset context
    success, message = client.set_context()
    assert_type_value(obj=success, type_or_value=bool, name="success")
    assert_type_value(obj=message, type_or_value=str, name="message")
    assert_log(expression=success, message=message)

    # inspect context
    success, message, context = client.get_context()
    assert_type_value(obj=success, type_or_value=bool, name="success")
    assert_type_value(obj=message, type_or_value=str, name="message")
    assert_log(expression=success, message=message)
    assert_type_value(obj=context, type_or_value=list, name="context")
    assert_log(expression=len(context) == 0, message=f"Expected context to contain '0' messages but got '{len(context)}'.")

    # add one message
    target = {'role': 'user', 'content': [{'type': 'text', 'text': "Hi, what's up?"}]}
    success, message = client.set_context(messages=[target])
    assert_type_value(obj=success, type_or_value=bool, name="success")
    assert_type_value(obj=message, type_or_value=str, name="message")
    assert_log(expression=success, message=message)

    # inspect context
    success, message, context = client.get_context()
    assert_type_value(obj=success, type_or_value=bool, name="success")
    assert_type_value(obj=message, type_or_value=str, name="message")
    assert_log(expression=success, message=message)
    assert_type_value(obj=context, type_or_value=list, name="context")
    assert_log(expression=len(context) == 1, message=f"Expected context to contain '1' message but got '{len(context)}'.")
    assert_log(expression=context[0] == target, message=f"Expected message '1' in context to be {target} but got '{context[0]}'.")

    # prompt
    success, message, completion = client.prompt()
    assert_type_value(obj=success, type_or_value=bool, name="success")
    assert_type_value(obj=message, type_or_value=str, name="message")
    assert_log(expression=success, message=message)
    assert_type_value(obj=completion, type_or_value=dict, name="completion")
    assert_keys(obj=completion, keys=['usage', 'text', 'logs'], mode="match", name="completion")
    assert_type_value(obj=completion['usage'], type_or_value=dict, name="usage in completion")
    assert_type_value(obj=completion['text'], type_or_value=str, name="text in completion")
    assert_type_value(obj=completion['logs'], type_or_value=list, name="logs in completion")
    assert_log(expression=len(completion['logs']) > 0, message="Expected to find at least one log in completion.")
    for i, log in enumerate(completion['logs']):
        assert_type_value(obj=log, type_or_value=str, name=f"log '{i}' in completion logs")

    # inspect context
    success, message, context = client.get_context()
    assert_type_value(obj=success, type_or_value=bool, name="success")
    assert_type_value(obj=message, type_or_value=str, name="message")
    assert_log(expression=success, message=message)
    assert_type_value(obj=context, type_or_value=list, name="context")
    assert_log(expression=len(context) == 2, message=f"Expected context to contain '2' messages but got '{len(context)}'.")
    assert_log(expression=context[0] == target, message=f"Expected message '1' in context to be {target} but got '{context[0]}'.")
    target = {'role': "assistant", 'content': completion['text']}
    assert_log(expression=context[1] == target, message=f"Expected message '1' in context to be {target} but got {context[1]}.")

def test_03_openrouter_text_completions():
    return _text_completions(stream=False, endpoint="OpenRouter", model="google/gemini-3-flash-preview")

def test_04_openrouter_text_completions_stream():
    return _text_completions(stream=True, endpoint="OpenRouter", model="google/gemini-3-flash-preview")

def test_05_openai_text_completions():
    return _text_completions(stream=False, endpoint="OpenAI", model="gpt-5-chat-latest")

def test_06_openai_text_completions_stream():
    return _text_completions(stream=True, endpoint="OpenAI", model="gpt-5-chat-latest")

def test_07_mistral_text_completions():
    return _text_completions(stream=False, endpoint="Mistral", model="mistral-large-2512")

def test_08_mistral_text_completions_stream():
    return _text_completions(stream=True, endpoint="Mistral", model="mistral-large-2512")

def test_09_vllm_text_completions():
    return _text_completions(stream=False, endpoint="AIS", model="ais/qwen3.5-27b", timeout_read=60, timeout_completion=60)

def test_10_vllm_text_completions_stream():
    return _text_completions(stream=True, endpoint="AIS", model="ais/qwen3.5-27b", timeout_read=60, timeout_completion=60)

# reasoning

def _reasoning_completion(**kwargs):
    client = ChatCompletions(kwargs)

    text = "Respond with 'test'. Your answer must not contain anything else."
    success, message, completion = client.prompt(text=text, reasoning_effort="low")

    assert_type_value(obj=success, type_or_value=bool, name="success")
    assert_type_value(obj=message, type_or_value=str, name="message")
    assert_log(expression=success, message=message)
    assert_type_value(obj=completion, type_or_value=dict, name="completion")
    assert_keys(obj=completion, keys=['usage', 'text', 'reasoning', 'logs'], mode="match", name="completion")
    assert_type_value(obj=completion['usage'], type_or_value=dict, name="usage in completion")
    assert_type_value(obj=completion['text'], type_or_value=str, name="text in completion")
    assert_log(expression=len(completion['text']) > 0, message="Expected text in completion to be non-empty string.")
    assert_type_value(obj=completion['reasoning'], type_or_value=str, name="reasoning in completion")
    assert_log(expression=len(completion['reasoning']) > 0, message="Expected reasoning in completion to be non-empty string.")
    assert_type_value(obj=completion['logs'], type_or_value=list, name="logs in completion")
    assert_log(expression=len(completion['logs']) > 0, message="Expected to find at least one log in completion.")
    for i, log in enumerate(completion['logs']):
        assert_type_value(obj=log, type_or_value=str, name=f"log '{i}' in completion logs")

def test_11_openrouter_reasoning():
    return _reasoning_completion(stream=False, endpoint="OpenRouter", model="qwen/qwen3.5-397b-a17b")

def test_12_openrouter_reasoning_stream():
    return _reasoning_completion(stream=True, endpoint="OpenRouter", model="qwen/qwen3.5-397b-a17b")

def test_13_openai_reasoning():
    return "NOT SUPPORTED."

def test_14_openai_reasoning_stream():
    return "NOT SUPPORTED."

def test_15_mistral_reasoning():
    return _reasoning_completion(stream=False, endpoint="Mistral", model="magistral-small-latest")

def test_16_mistral_reasoning_stream():
    return _reasoning_completion(stream=True, endpoint="Mistral", model="magistral-small-latest")

def test_17_vllm_reasoning():
    return _reasoning_completion(stream=False, endpoint="AIS", model="ais/qwen3.5-27b", timeout_read=60, timeout_completion=60)

def test_18_vllm_reasoning_stream():
    return _reasoning_completion(stream=True, endpoint="AIS", model="ais/qwen3.5-27b", timeout_read=60, timeout_completion=60)

# interrupt

def _interrupt(**kwargs):
    client = ChatCompletions(validate_model=False, reasoning_effort="high", retry=10, **kwargs)

    text = "Write a 10 sentence novel chapter about a computer scientist."
    response = None

    def fun(client):
        nonlocal response
        response = client.prompt(text=text)

    t = threading.Thread(target=fun, args=(client,))
    t.start()
    time.sleep(0.5)
    success, interrupt_message = client.interrupt()
    assert_type_value(obj=success, type_or_value=bool, name="success")
    assert_type_value(obj=interrupt_message, type_or_value=str, name="message")
    assert_log(expression=success, message=interrupt_message)
    assert_log(expression="no completion" not in interrupt_message, message=f"Unexpected interrupt message: {interrupt_message}")

    tic = time.perf_counter()
    t.join()
    toc = time.perf_counter()
    join_duration = toc - tic
    assert_log(expression=join_duration < 0.1, message=f"Expected immediate join of thread after successful interrupt but it took '{join_duration:.3f}s'.")

    assert_type_value(obj=response, type_or_value=tuple, name="response")
    assert_log(expression=len(response) == 3, message=f"Expected response of to be tuple of length '3' but got '{len(response)}'.")
    success, message, completion = response
    assert_type_value(obj=success, type_or_value=bool, name="success")
    assert_type_value(obj=message, type_or_value=str, name="message")
    assert_log(expression=not success, message=message)
    assert_type_value(obj=completion, type_or_value=[dict, None], name="completion")

    return interrupt_message

def test_19_interrupt():
    # return _interrupt(stream=False)
    return "NOT SUPPORTED."

def test_20_interrupt_stream():
    return _interrupt(stream=True)

# web search

def _web_search(**kwargs):
    client = ChatCompletions(kwargs)

    text = "Summarize the major news story of yesterday in one short sentence."
    success, message, completion = client.prompt(text=text)

    assert_type_value(obj=success, type_or_value=bool, name="success")
    assert_type_value(obj=message, type_or_value=str, name="message")
    assert_log(expression=success, message=message)
    assert_type_value(obj=completion, type_or_value=dict, name="completion")
    assert_keys(obj=completion, keys=['usage', 'text', 'logs'], mode="match", name="completion")
    assert_type_value(obj=completion['usage'], type_or_value=dict, name="usage in completion")
    assert_type_value(obj=completion['text'], type_or_value=str, name="text in completion")
    assert_log(expression=len(completion['text']) > 0, message="Expected text in completion to be non-empty string.")
    assert_type_value(obj=completion['logs'], type_or_value=list, name="logs in completion")
    assert_log(expression=len(completion['logs']) > 0, message="Expected to find at least one log in completion.")
    for i, log in enumerate(completion['logs']):
        assert_type_value(obj=log, type_or_value=str, name=f"log '{i}' in completion logs")

    return completion['text']

def test_21_openrouter_web_search():
    return _web_search(endpoint="OpenRouter", model="google/gemini-2.5-flash:online")

def test_22_openai_web_search():
    # return _web_search(endpoint="OpenAI", model="")
    return "NOT SUPPORTED."

def test_23_mistral_web_search():
    # return _web_search(endpoint="Mistral", model="")
    return "NOT SUPPORTED."

def test_24_vllm_web_search():
    return "NOT SUPPORTED."

# image input

def _image_file_input(**kwargs):
    client = ChatCompletions(kwargs)

    text = {"role": "user", "content": [
        {'type': "text", 'text': "Please describe this image in one short sentence."},
        {'type': "image_url", 'image_url': {'url': os.path.join(nimbro_api.__path__[0], "test", "assets", "test.png"), 'detail': "high"}}
    ]}
    success, message, completion = client.prompt(text=text)

    assert_type_value(obj=success, type_or_value=bool, name="success")
    assert_type_value(obj=message, type_or_value=str, name="message")
    assert_log(expression=success, message=message)
    assert_type_value(obj=completion, type_or_value=dict, name="completion")
    assert_keys(obj=completion, keys=['usage', 'text', 'logs'], mode="match", name="completion")
    assert_type_value(obj=completion['usage'], type_or_value=dict, name="usage in completion")
    assert_type_value(obj=completion['text'], type_or_value=str, name="text in completion")
    assert_log(expression=len(completion['text']) > 0, message="Expected text in completion to be non-empty string.")
    assert_type_value(obj=completion['logs'], type_or_value=list, name="logs in completion")
    assert_log(expression=len(completion['logs']) > 0, message="Expected to find at least one log in completion.")
    for i, log in enumerate(completion['logs']):
        assert_type_value(obj=log, type_or_value=str, name=f"log '{i}' in completion logs")

    return completion['text']

def test_25_openrouter_image_file_input():
    return _image_file_input(endpoint="OpenRouter", model="google/gemini-2.5-flash")

def test_26_openai_image_file_input():
    return _image_file_input(endpoint="OpenAI", model="gpt-5-chat-latest")

def test_27_mistral_image_file_input():
    return _image_file_input(endpoint="Mistral", model="mistral-large-2512")

def test_28_vllm_image_file_input():
    return _image_file_input(endpoint="AIS", model="ais/qwen3.5-27b", timeout_read=60, timeout_completion=60)

def _image_url_input(**kwargs):
    kwargs['download_image'] = False
    client = ChatCompletions(kwargs)

    text = {"role": "user", "content": [
        {'type': "text", 'text': "Please describe this image in one short sentence."},
        {'type': "image_url", 'image_url': {'url': "https://www.ais.uni-bonn.de/nimbro/@Home/images/RC24/RoboCup_2024_NimbRo_Home_Final_4__07_21.jpg", 'detail': "high"}}
    ]}
    success, message, completion = client.prompt(text=text)

    assert_type_value(obj=success, type_or_value=bool, name="success")
    assert_type_value(obj=message, type_or_value=str, name="message")
    assert_log(expression=success, message=message)
    assert_type_value(obj=completion, type_or_value=dict, name="completion")
    assert_keys(obj=completion, keys=['usage', 'text', 'logs'], mode="match", name="completion")
    assert_type_value(obj=completion['usage'], type_or_value=dict, name="usage in completion")
    assert_type_value(obj=completion['text'], type_or_value=str, name="text in completion")
    assert_log(expression=len(completion['text']) > 0, message="Expected text in completion to be non-empty string.")
    assert_type_value(obj=completion['logs'], type_or_value=list, name="logs in completion")
    assert_log(expression=len(completion['logs']) > 0, message="Expected to find at least one log in completion.")
    for i, log in enumerate(completion['logs']):
        assert_type_value(obj=log, type_or_value=str, name=f"log '{i}' in completion logs")

    return completion['text']

def test_29_openrouter_image_url_input():
    return _image_url_input(endpoint="OpenRouter", model="google/gemini-2.5-flash")

def test_30_openai_image_url_input():
    return _image_url_input(endpoint="OpenAI", model="gpt-5-chat-latest")

def test_31_mistral_image_url_input():
    return _image_url_input(endpoint="Mistral", model="mistral-large-2512")

def test_32_vllm_image_url_input():
    return _image_url_input(endpoint="AIS", model="ais/qwen3.5-27b", timeout_read=60, timeout_completion=60)

# audio input

def _audio_file_input(**kwargs):
    client = ChatCompletions(kwargs)

    text = {"role": "user", "content": [
        {'type': "text", 'text': "Please describe this audio in one short sentence."},
        {'type': "input_audio", 'input_audio': {'data': os.path.join(nimbro_api.__path__[0], "test", "assets", "test.wav"), 'format': "wav"}}
    ]}
    success, message, completion = client.prompt(text=text)

    assert_type_value(obj=success, type_or_value=bool, name="success")
    assert_type_value(obj=message, type_or_value=str, name="message")
    assert_log(expression=success, message=message)
    assert_type_value(obj=completion, type_or_value=dict, name="completion")
    assert_keys(obj=completion, keys=['usage', 'text', 'logs'], mode="match", name="completion")
    assert_type_value(obj=completion['usage'], type_or_value=dict, name="usage in completion")
    assert_type_value(obj=completion['text'], type_or_value=str, name="text in completion")
    assert_log(expression=len(completion['text']) > 0, message="Expected text in completion to be non-empty string.")
    assert_type_value(obj=completion['logs'], type_or_value=list, name="logs in completion")
    assert_log(expression=len(completion['logs']) > 0, message="Expected to find at least one log in completion.")
    for i, log in enumerate(completion['logs']):
        assert_type_value(obj=log, type_or_value=str, name=f"log '{i}' in completion logs")

    return completion['text']

def test_33_openrouter_audio_file_input():
    return _audio_file_input(endpoint="OpenRouter", model="google/gemini-2.5-flash")

def test_34_openai_audio_file_input():
    return _audio_file_input(endpoint="OpenAI", model="gpt-audio-2025-08-28")

def test_35_mistral_audio_file_input():
    return _audio_file_input(endpoint="Mistral", model="voxtral-small-latest")

def test_36_vllm_audio_file_input():
    return "NOT SUPPORTED."

def _audio_url_input(**kwargs):
    kwargs['download_audio'] = False
    client = ChatCompletions(kwargs)

    text = {"role": "user", "content": [
        {'type': "text", 'text': "Please describe this audio in one short sentence."},
        {'type': "input_audio", 'input_audio': {'data': "https://sample-files.com/downloads/audio/wav/voice-sample.wav", 'format': "wav"}}
    ]}
    success, message, completion = client.prompt(text=text)

    assert_type_value(obj=success, type_or_value=bool, name="success")
    assert_type_value(obj=message, type_or_value=str, name="message")
    assert_log(expression=success, message=message)
    assert_type_value(obj=completion, type_or_value=dict, name="completion")
    assert_keys(obj=completion, keys=['usage', 'text', 'logs'], mode="match", name="completion")
    assert_type_value(obj=completion['usage'], type_or_value=dict, name="usage in completion")
    assert_type_value(obj=completion['text'], type_or_value=str, name="text in completion")
    assert_log(expression=len(completion['text']) > 0, message="Expected text in completion to be non-empty string.")
    assert_type_value(obj=completion['logs'], type_or_value=list, name="logs in completion")
    assert_log(expression=len(completion['logs']) > 0, message="Expected to find at least one log in completion.")
    for i, log in enumerate(completion['logs']):
        assert_type_value(obj=log, type_or_value=str, name=f"log '{i}' in completion logs")

    return completion['text']

def test_37_openrouter_audio_url_input():
    # return _audio_url_input(endpoint="OpenRouter", model="")
    return "NOT SUPPORTED."

def test_38_openai_audio_url_input():
    # return _audio_url_input(endpoint="OpenAI", model="gpt-audio-2025-08-28")
    return "NOT SUPPORTED."

def test_39_mistral_audio_url_input():
    return _audio_url_input(endpoint="Mistral", model="voxtral-small-latest")

def test_40_vllm_audio_url_input():
    return "NOT SUPPORTED."

# video input

def _video_file_input(**kwargs):
    client = ChatCompletions(kwargs)

    text = {"role": "user", "content": [
        {'type': "text", 'text': "Please describe this video in one short sentence."},
        {'type': "video_url", 'video_url': {'url': os.path.join(nimbro_api.__path__[0], "test", "assets", "test.mp4")}}
    ]}
    success, message, completion = client.prompt(text=text)

    assert_type_value(obj=success, type_or_value=bool, name="success")
    assert_type_value(obj=message, type_or_value=str, name="message")
    assert_log(expression=success, message=message)
    assert_type_value(obj=completion, type_or_value=dict, name="completion")
    assert_keys(obj=completion, keys=['usage', 'text', 'logs'], mode="match", name="completion")
    assert_type_value(obj=completion['usage'], type_or_value=dict, name="usage in completion")
    assert_type_value(obj=completion['text'], type_or_value=str, name="text in completion")
    assert_log(expression=len(completion['text']) > 0, message="Expected text in completion to be non-empty string.")
    assert_type_value(obj=completion['logs'], type_or_value=list, name="logs in completion")
    assert_log(expression=len(completion['logs']) > 0, message="Expected to find at least one log in completion.")
    for i, log in enumerate(completion['logs']):
        assert_type_value(obj=log, type_or_value=str, name=f"log '{i}' in completion logs")

    return completion['text']

def test_41_openrouter_video_file_input():
    return _video_file_input(endpoint="OpenRouter", model="google/gemini-2.5-flash")

def test_42_openai_video_file_input():
    # return _video_file_input(endpoint="OpenAI", model="")
    return "NOT SUPPORTED."

def test_43_mistral_video_file_input():
    # return _video_file_input(endpoint="Mistral", model="")
    return "NOT SUPPORTED."

def test_44_vllm_video_file_input():
    endpoint = {
        'name': "spark",
        'api_flavor': "vllm",
        'api_url': "http://asus-gx10-0.ais.uni-bonn.de:8000/v1/chat/completions",
        # 'api_url': "https://api-code.ais.uni-bonn.de/video/v1/chat/completions",
        'key_type': "plain",
        'key_value': "xnxcScUkYIsXZ7"
    }
    return _video_file_input(endpoint=endpoint, model="ais/qwen3.5-27b", timeout_read=60, timeout_completion=60)

def _video_url_input(**kwargs):
    kwargs['download_video'] = False
    client = ChatCompletions(kwargs)

    text = {"role": "user", "content": [
        {'type': "text", 'text': "Please describe this video in one short sentence."},
        {'type': "video_url", 'video_url': {'url': "https://www.youtube.com/watch?v=dQw4w9WgXcQ"}}
    ]}
    success, message, completion = client.prompt(text=text)

    assert_type_value(obj=success, type_or_value=bool, name="success")
    assert_type_value(obj=message, type_or_value=str, name="message")
    assert_log(expression=success, message=message)
    assert_type_value(obj=completion, type_or_value=dict, name="completion")
    assert_keys(obj=completion, keys=['usage', 'text', 'logs'], mode="match", name="completion")
    assert_type_value(obj=completion['usage'], type_or_value=dict, name="usage in completion")
    assert_type_value(obj=completion['text'], type_or_value=str, name="text in completion")
    assert_log(expression=len(completion['text']) > 0, message="Expected text in completion to be non-empty string.")
    assert_type_value(obj=completion['logs'], type_or_value=list, name="logs in completion")
    assert_log(expression=len(completion['logs']) > 0, message="Expected to find at least one log in completion.")
    for i, log in enumerate(completion['logs']):
        assert_type_value(obj=log, type_or_value=str, name=f"log '{i}' in completion logs")

    return completion['text']

def test_45_openrouter_video_url_input():
    return _video_url_input(endpoint="OpenRouter", model="google/gemini-2.5-flash")

def test_46_openai_video_url_input():
    # return _video_url_input(endpoint="OpenAI", model="")
    return "NOT SUPPORTED."

def test_47_mistral_video_url_input():
    # return _video_url_input(endpoint="Mistral", model="")
    return "NOT SUPPORTED."

def test_48_vllm_video_url_input():
    return "NOT SUPPORTED."

# file input

def _file_local_input(**kwargs):
    client = ChatCompletions(kwargs)

    text = {"role": "user", "content": [
        {'type': "text", 'text': "Please describe this document in one short sentence."},
        {'type': "file", 'file': {'filename': "paper.pdf", 'file_data': os.path.join(nimbro_api.__path__[0], "test", "assets", "test.pdf")}}
    ]}
    success, message, completion = client.prompt(text=text)

    assert_type_value(obj=success, type_or_value=bool, name="success")
    assert_type_value(obj=message, type_or_value=str, name="message")
    assert_log(expression=success, message=message)
    assert_type_value(obj=completion, type_or_value=dict, name="completion")
    assert_keys(obj=completion, keys=['usage', 'text', 'logs'], mode="match", name="completion")
    assert_type_value(obj=completion['usage'], type_or_value=dict, name="usage in completion")
    assert_type_value(obj=completion['text'], type_or_value=str, name="text in completion")
    assert_log(expression=len(completion['text']) > 0, message="Expected text in completion to be non-empty string.")
    assert_type_value(obj=completion['logs'], type_or_value=list, name="logs in completion")
    assert_log(expression=len(completion['logs']) > 0, message="Expected to find at least one log in completion.")
    for i, log in enumerate(completion['logs']):
        assert_type_value(obj=log, type_or_value=str, name=f"log '{i}' in completion logs")

    return completion['text']

def test_49_openrouter_file_local_input():
    return _file_local_input(endpoint="OpenRouter", model="google/gemini-2.5-flash")

def test_50_openai_file_local_input():
    return _file_local_input(endpoint="OpenAI", model="gpt-5-chat-latest")

def test_51_mistral_file_local_input():
    # return _file_local_input(endpoint="Mistral", model="")
    return "NOT SUPPORTED."

def test_52_vllm_file_local_input():
    return "NOT SUPPORTED."

def _file_url_input(**kwargs):
    kwargs['download_file'] = False
    client = ChatCompletions(kwargs)

    text = {"role": "user", "content": [
        {'type': "text", 'text': "Please describe this document in one short sentence."},
        {'type': "file", 'file': {'filename': "paper.pdf", 'file_data': "https://bitcoinpaper.org/bitcoin.pdf"}}
    ]}
    success, message, completion = client.prompt(text=text)

    assert_type_value(obj=success, type_or_value=bool, name="success")
    assert_type_value(obj=message, type_or_value=str, name="message")
    assert_log(expression=success, message=message)
    assert_type_value(obj=completion, type_or_value=dict, name="completion")
    assert_keys(obj=completion, keys=['usage', 'text', 'logs'], mode="match", name="completion")
    assert_type_value(obj=completion['usage'], type_or_value=dict, name="usage in completion")
    assert_type_value(obj=completion['text'], type_or_value=str, name="text in completion")
    assert_log(expression=len(completion['text']) > 0, message="Expected text in completion to be non-empty string.")
    assert_type_value(obj=completion['logs'], type_or_value=list, name="logs in completion")
    assert_log(expression=len(completion['logs']) > 0, message="Expected to find at least one log in completion.")
    for i, log in enumerate(completion['logs']):
        assert_type_value(obj=log, type_or_value=str, name=f"log '{i}' in completion logs")

    return completion['text']

def test_53_openrouter_file_url_input():
    return _file_url_input(endpoint="OpenRouter", model="google/gemini-2.5-flash")

def test_54_openai_file_url_input():
    # return _file_url_input(endpoint="OpenAI", model="")
    return "NOT SUPPORTED."

def test_55_mistral_file_url_input():
    # return _file_url_input(endpoint="Mistral", model="")
    return "NOT SUPPORTED."

def test_56_vllm_file_url_input():
    return "NOT SUPPORTED."

# parallel

def test_57_parallel_threads(n=100):
    class PropagatingThread(threading.Thread):
        def run(self):
            self.exc = None
            try:
                if hasattr(self, '_Thread__target'):
                    # Thread uses name mangling prior to Python 3.
                    self.ret = self._Thread__target(*self._Thread__args, **self._Thread__kwargs)
                else:
                    self.ret = self._target(*self._args, **self._kwargs)
            except BaseException as e:
                self.exc = e

        def join(self, timeout=None):
            super(PropagatingThread, self).join(timeout)
            if self.exc:
                raise self.exc
            return self.ret

    def _thread_worker(client):
        success, message, completion = client.prompt("Tell me a joke!", response_type="text")
        assert_type_value(obj=success, type_or_value=bool, name="success")
        assert_type_value(obj=message, type_or_value=str, name="message")
        assert_log(expression=success, message=message)
        assert_type_value(obj=completion, type_or_value=dict, name="completion")
        assert_keys(obj=completion, keys=['usage', 'text', 'logs'], mode="match", name="completion")
        assert_type_value(obj=completion['usage'], type_or_value=dict, name="usage in completion")
        assert_keys(obj=completion['usage'], keys=['duration'], mode="required", name="usage in completion")
        assert_type_value(obj=completion['usage']['duration'], type_or_value=float, name="duration of usage in completion")
        assert_log(expression=completion['usage']['duration'] > 0, message=f"Expected duration of usage in completion to be grater zero but got '{completion['usage']['duration']}'.")
        stamps.append(completion['usage']['duration'])

    stamp = time.perf_counter()

    threads, clients, stamps = [], [], []
    for _ in range(n):
        client = ChatCompletions()
        clients.append(client)
        thread = PropagatingThread(target=_thread_worker, args=(client,))
        threads.append(thread)

    time_created = time.perf_counter() - stamp

    for t in threads:
        t.start()

    time_started = time.perf_counter() - stamp - time_created

    exceptions = []
    for t in threads:
        try:
            t.join()
        except Exception as e:
            if isinstance(e, UnrecoverableError):
                exceptions.append(str(e))
            else:
                exceptions.append(repr(e))
    if len(exceptions) > 0:
        raise UnrecoverableError(f"Failed to generate '{len(exceptions)}' of '{n}' completions: {exceptions}")

    time_joined = time.perf_counter() - stamp - time_created - time_started

    mean = sum(stamps) / len(stamps)
    median = sorted(stamps)[len(stamps) // 2] if len(stamps) % 2 == 1 else (sorted(stamps)[len(stamps) // 2 - 1] + sorted(stamps)[len(stamps) // 2]) / 2
    std_dev = math.sqrt(sum((x - mean) ** 2 for x in stamps) / len(stamps))

    return f"Generated '{n}' chat completions (Create:'{time_created:.3f}s', Start:'{time_started:.3f}s', Join:'{time_joined:.3f}s', Mean:'{mean:.3f}s', Median:'{median:.3f}s', Std.:'{std_dev:.3f}s', Min.:'{min(stamps):.3f}s', Max.:'{max(stamps):.3f}s')."

def _process_wrapper(target, queue, *args, **kwargs):
    # defined here instead of inside test_11_parallel_processes for windows compatibility
    try:
        target(*args, **kwargs)
    except Exception as e:
        queue.put(e)

def _process_worker(shared_stamps):
    # defined here instead of inside test_11_parallel_processes for windows compatibility
    success, message = nimbro_api.set_settings(settings={'logger_mute': True})
    assert_type_value(obj=success, type_or_value=bool, name="success")
    assert_type_value(obj=message, type_or_value=str, name="message")
    assert_log(expression=success, message=message)

    client = ChatCompletions()
    success, message, completion = client.prompt("Tell me a joke!", response_type="text")
    assert_type_value(obj=success, type_or_value=bool, name="success")
    assert_type_value(obj=message, type_or_value=str, name="message")
    assert_log(expression=success, message=message)
    assert_type_value(obj=completion, type_or_value=dict, name="completion")
    assert_keys(obj=completion, keys=['usage', 'text', 'logs'], mode="match", name="completion")
    assert_type_value(obj=completion['usage'], type_or_value=dict, name="usage in completion")
    assert_keys(obj=completion['usage'], keys=['duration'], mode="required", name="usage in completion")
    assert_type_value(obj=completion['usage']['duration'], type_or_value=float, name="duration of usage in completion")
    assert_log(expression=completion['usage']['duration'] > 0, message=f"Expected duration of usage in completion to be grater zero but got '{completion['usage']['duration']}'.")
    shared_stamps.append(completion['usage']['duration'])

def test_58_parallel_processes(n=100):
    stamp = time.perf_counter()

    with multiprocessing.Manager() as manager:
        stamps = manager.list()

        processes, queues = [], []
        for _ in range(n):
            q = multiprocessing.Queue()
            queues.append(q)
            p = multiprocessing.Process(
                target=_process_wrapper,
                args=(_process_worker, q, stamps)
            )
            processes.append(p)

        time_created = time.perf_counter() - stamp

        for p in processes:
            p.start()

        time_started = time.perf_counter() - stamp - time_created

        exceptions = []
        for idx, p in enumerate(processes):
            p.join()
            if not queues[idx].empty():
                e = queues[idx].get()
                if isinstance(e, UnrecoverableError):
                    exceptions.append(str(e))
                else:
                    exceptions.append(repr(e))
        if len(exceptions) > 0:
            raise UnrecoverableError(f"Failed to generate '{len(exceptions)}' of '{n}' completions: {exceptions}")

        time_joined = time.perf_counter() - stamp - time_created - time_started

        final_stamps = list(stamps)
        mean = sum(final_stamps) / len(final_stamps)
        median = sorted(final_stamps)[len(final_stamps) // 2] if len(final_stamps) % 2 == 1 else (sorted(final_stamps)[len(final_stamps) // 2 - 1] + sorted(final_stamps)[len(final_stamps) // 2]) / 2
        std_dev = math.sqrt(sum((x - mean) ** 2 for x in final_stamps) / len(final_stamps))

        return f"Generated '{n}' chat completions (Create:'{time_created:.3f}s', Start:'{time_started:.3f}s', Join:'{time_joined:.3f}s', Mean:'{mean:.3f}s', Median:'{median:.3f}s', Std.:'{std_dev:.3f}s', Min.:'{min(final_stamps):.3f}s', Max.:'{max(final_stamps):.3f}s')."

# tools

def test_59_tool_config():
    client = ChatCompletions()

    success, message, tools = client.get_tools()
    assert_type_value(obj=success, type_or_value=bool, name="success")
    assert_type_value(obj=message, type_or_value=str, name="message")
    assert_log(expression=success, message=message)
    assert_type_value(obj=tools, type_or_value=list, name="tools")
    assert_log(expression=len(tools) == 0, message=f"Expected zero tools to be defined but got '{len(tools)}': {tools}")

    success, message = client.set_tools(tools=None)
    assert_type_value(obj=success, type_or_value=bool, name="success")
    assert_type_value(obj=message, type_or_value=str, name="message")
    assert_log(expression=success, message=message)

    success, message, tools = client.get_tools()
    assert_type_value(obj=success, type_or_value=bool, name="success")
    assert_type_value(obj=message, type_or_value=str, name="message")
    assert_log(expression=success, message=message)
    assert_type_value(obj=tools, type_or_value=list, name="tools")
    assert_log(expression=len(tools) == 0, message=f"Expected zero tools to be defined but got '{len(tools)}': {tools}")

    tool_definitions = [
        {
            'type': "function",
            'function': {
                'name': "get_current_weather",
                'description': "Get the current weather at the current location",
                'parameters': {
                    'type': "object",
                    'properties': {}
                }
            }
        },
        {
            'type': "function",
            'function': {
                'name': "get_current_time",
                'description': "Get the current time at the current location",
                'parameters': {
                    'type': "object",
                    'properties': {},
                    'additionalProperties': False
                },
                'strict': True
            }
        },
        {
            'type': "function",
            'function': {
                'name': "speak",
                'description': "Speak to a person, e.g. the user. Never use plain text repsonses to address anyone.",
                'parameters': {
                    'type': 'object',
                    'properties': {
                        'person': {
                            'type': "string",
                            'description': "Specifies the person to speak to. Pass 'everyone' to address everyone in the robot's vicinity, rather than a specific person"
                        },
                        'text': {
                            'type': "string",
                            'description': "Specifies the text to be said. Be friendly, concise, and helpful"
                        },
                        'requires_answer': {
                            'type': "boolean",
                            'description': "Signals that the spoken text requires an answer and makes the robot wait for it. The answer will then be returned with the response to this function call"
                        }
                    },
                    'required': ['person', 'text', 'requires_answer']
                }
            }
        },
        {
            'type': 'function',
            'function': {
                'name': 'translate_text',
                'description': 'Translate text from one language to another.',
                'parameters': {
                    'type': 'object',
                    'properties': {
                        'source_language': {
                            'type': 'string',
                            'description': 'The language to translate from.'
                        },
                        'target_language': {
                            'type': 'string',
                            'description': 'The language to translate into.'
                        },
                        'text': {
                            'type': 'string',
                            'description': 'The text to translate.'
                        },
                        'advanced_options': {
                            'type': 'object',
                            'description': 'Optional advanced translation settings.',
                            'properties': {
                                'formal': {
                                    'type': 'boolean',
                                    'description': 'Whether to use a formal tone.'
                                },
                                'context': {
                                    'type': 'string',
                                    'description': 'Additional context to improve translation accuracy.'
                                }
                            },
                            'required': []
                        }
                    },
                    'required': ['source_language', 'target_language', 'text']
                }
            }
        }
    ]

    success, message = client.set_tools(tools=tool_definitions)
    assert_type_value(obj=success, type_or_value=bool, name="success")
    assert_type_value(obj=message, type_or_value=str, name="message")
    assert_log(expression=success, message=message)

    success, message, tools = client.get_tools()
    assert_type_value(obj=success, type_or_value=bool, name="success")
    assert_type_value(obj=message, type_or_value=str, name="message")
    assert_log(expression=success, message=message)
    assert_type_value(obj=tools, type_or_value=list, name="tools")
    assert_log(expression=len(tools) == len(tool_definitions), message=f"Expected '{len(tool_definitions)}' tools to be defined but got '{len(tools)}': {tools}")
    for i, tool in enumerate(tools):
        assert_type_value(obj=tool, type_or_value=dict, name=f"tool '{i}'")
        assert_log(expression=tool == tool_definitions[i], message=f"Expected tool '{i}' to match definition but got: {tool}")

    tool_definitions[1]['function']['parameters']['additionalProperties'] = True

    success, message = client.set_tools(tools=tool_definitions)
    assert_type_value(obj=success, type_or_value=bool, name="success")
    assert_type_value(obj=message, type_or_value=str, name="message")
    assert_log(expression=not success, message=message)

    del tool_definitions[1]['function']['parameters']['additionalProperties']

    success, message = client.set_tools(tools=tool_definitions)
    assert_type_value(obj=success, type_or_value=bool, name="success")
    assert_type_value(obj=message, type_or_value=str, name="message")
    assert_log(expression=not success, message=message)

    del tool_definitions[1]['function']['strict']

    success, message = client.set_tools(tools=tool_definitions)
    assert_type_value(obj=success, type_or_value=bool, name="success")
    assert_type_value(obj=message, type_or_value=str, name="message")
    assert_log(expression=success, message=message)

    success, message, tools = client.get_tools()
    assert_type_value(obj=success, type_or_value=bool, name="success")
    assert_type_value(obj=message, type_or_value=str, name="message")
    assert_log(expression=success, message=message)
    assert_type_value(obj=tools, type_or_value=list, name="tools")
    assert_log(expression=len(tools) == len(tool_definitions), message=f"Expected '{len(tool_definitions)}' tools to be defined but got '{len(tools)}': {tools}")
    for i, tool in enumerate(tools):
        assert_type_value(obj=tool, type_or_value=dict, name=f"tool '{i}'")
        assert_log(expression=tool == tool_definitions[i], message=f"Expected tool '{i}' to match definition but got: {tool}")

    success, message = client.set_tools(tools=[])
    success, message, tools = client.get_tools()
    assert_type_value(obj=success, type_or_value=bool, name="success")
    assert_type_value(obj=message, type_or_value=str, name="message")
    assert_log(expression=success, message=message)
    assert_type_value(obj=tools, type_or_value=list, name="tools")
    assert_log(expression=len(tools) == 0, message=f"Expected zero tools to be defined but got '{len(tools)}': {tools}")

    success, message = client.set_tools(tools=tool_definitions[0])
    assert_type_value(obj=success, type_or_value=bool, name="success")
    assert_type_value(obj=message, type_or_value=str, name="message")
    assert_log(expression=success, message=message)

    success, message, tools = client.get_tools()
    assert_type_value(obj=success, type_or_value=bool, name="success")
    assert_type_value(obj=message, type_or_value=str, name="message")
    assert_log(expression=success, message=message)
    assert_type_value(obj=tools, type_or_value=list, name="tools")
    assert_log(expression=len(tools) == 1, message=f"Expected '1' tool to be defined but got '{len(tools)}': {tools}")
    assert_type_value(obj=tool, type_or_value=dict, name="tool")
    assert_log(expression=tool == tool_definitions[i], message=f"Expected tool to match definition but got: {tool}")

def test_60_context_config():
    client = ChatCompletions()

    success, message, context = client.get_context()
    assert_type_value(obj=success, type_or_value=bool, name="success")
    assert_type_value(obj=message, type_or_value=str, name="message")
    assert_log(expression=success, message=message)
    assert_type_value(obj=context, type_or_value=list, name="context")
    assert_log(expression=len(context) == 0, message=f"Expected zero messages in context but got '{len(context)}': {context}")

    success, message = client.set_context(mode="remove", messages=None, index=0, reverse_indexing=True)
    assert_type_value(obj=success, type_or_value=bool, name="success")
    assert_type_value(obj=message, type_or_value=str, name="message")
    assert_log(expression=not success, message=message)

    success, message, context = client.get_context()
    assert_type_value(obj=success, type_or_value=bool, name="success")
    assert_type_value(obj=message, type_or_value=str, name="message")
    assert_log(expression=success, message=message)
    assert_type_value(obj=context, type_or_value=list, name="context")
    assert_log(expression=len(context) == 0, message=f"Expected zero messages in context but got '{len(context)}': {context}")

    target = [
        {
            'role': "system",
            'content': "You are a helpful assistant."
        }
    ]

    success, message = client.set_context(mode="replace", messages=target, index=0, reverse_indexing=True)
    assert_type_value(obj=success, type_or_value=bool, name="success")
    assert_type_value(obj=message, type_or_value=str, name="message")
    assert_log(expression=not success, message=message)

    success, message, context = client.get_context()
    assert_type_value(obj=success, type_or_value=bool, name="success")
    assert_type_value(obj=message, type_or_value=str, name="message")
    assert_log(expression=success, message=message)
    assert_type_value(obj=context, type_or_value=list, name="context")
    assert_log(expression=len(context) == 0, message=f"Expected zero messages in context but got '{len(context)}': {context}")

    success, message = client.set_context(mode="insert", messages=target, index=0, reverse_indexing=True)
    assert_type_value(obj=success, type_or_value=bool, name="success")
    assert_type_value(obj=message, type_or_value=str, name="message")
    assert_log(expression=success, message=message)

    success, message, context = client.get_context()
    assert_type_value(obj=success, type_or_value=bool, name="success")
    assert_type_value(obj=message, type_or_value=str, name="message")
    assert_log(expression=success, message=message)
    assert_log(expression=context == target, message=f"Expected context to match set messages but got': {context}")

    target.append({
        'role': "user",
        'content': "Hello there!"
    })

    success, message = client.set_context(mode="insert", messages=target[-1], index=0, reverse_indexing=True)
    assert_type_value(obj=success, type_or_value=bool, name="success")
    assert_type_value(obj=message, type_or_value=str, name="message")
    assert_log(expression=success, message=message)

    success, message, context = client.get_context()
    assert_type_value(obj=success, type_or_value=bool, name="success")
    assert_type_value(obj=message, type_or_value=str, name="message")
    assert_log(expression=success, message=message)
    assert_log(expression=context == target, message=f"Expected context to match set messages but got': {context}")

    target[-1]['content'] = "Hello dear language model!"
    target.append({
        'role': "assistant",
        'content': "How can I help you?"
    })

    success, message = client.set_context(mode="replace", messages=target[1:], index=1, reverse_indexing=False)
    assert_type_value(obj=success, type_or_value=bool, name="success")
    assert_type_value(obj=message, type_or_value=str, name="message")
    assert_log(expression=success, message=message)

    success, message, context = client.get_context()
    assert_type_value(obj=success, type_or_value=bool, name="success")
    assert_type_value(obj=message, type_or_value=str, name="message")
    assert_log(expression=success, message=message)
    assert_log(expression=context == target, message=f"Expected context to match set messages but got': {context}")

    target = [{
        'role': "system",
        'content': "You are a helpful assistant."
    }] + target

    success, message = client.set_context(mode="insert", messages=target[0], index=0, reverse_indexing=False)
    assert_type_value(obj=success, type_or_value=bool, name="success")
    assert_type_value(obj=message, type_or_value=str, name="message")
    assert_log(expression=success, message=message)

    success, message, context = client.get_context()
    assert_type_value(obj=success, type_or_value=bool, name="success")
    assert_type_value(obj=message, type_or_value=str, name="message")
    assert_log(expression=success, message=message)
    assert_log(expression=context == target, message=f"Expected context to match set messages but got': {context}")

    target = target[1:]

    success, message = client.set_context(mode="remove", messages=None, index=0, reverse_indexing=False)
    assert_type_value(obj=success, type_or_value=bool, name="success")
    assert_type_value(obj=message, type_or_value=str, name="message")
    assert_log(expression=success, message=message)

    success, message, context = client.get_context()
    assert_type_value(obj=success, type_or_value=bool, name="success")
    assert_type_value(obj=message, type_or_value=str, name="message")
    assert_log(expression=success, message=message)
    assert_log(expression=context == target, message=f"Expected context to match set messages but got': {context}")

    target.append({
        'role': "user",
        'content': "I'm fine thanks!"
    })

    success, message = client.set_context(mode="reset", messages=target, index=0, reverse_indexing=True)
    assert_type_value(obj=success, type_or_value=bool, name="success")
    assert_type_value(obj=message, type_or_value=str, name="message")
    assert_log(expression=success, message=message)

    success, message, context = client.get_context()
    assert_type_value(obj=success, type_or_value=bool, name="success")
    assert_type_value(obj=message, type_or_value=str, name="message")
    assert_log(expression=success, message=message)
    assert_log(expression=context == target, message=f"Expected context to match set messages but got': {context}")

    success, message = client.set_context(mode="reset", messages=None, index=0, reverse_indexing=True)
    assert_type_value(obj=success, type_or_value=bool, name="success")
    assert_type_value(obj=message, type_or_value=str, name="message")
    assert_log(expression=success, message=message)

    success, message, context = client.get_context()
    assert_type_value(obj=success, type_or_value=bool, name="success")
    assert_type_value(obj=message, type_or_value=str, name="message")
    assert_log(expression=success, message=message)
    assert_type_value(obj=context, type_or_value=list, name="context")
    assert_log(expression=len(context) == 0, message=f"Expected zero messages in context but got '{len(context)}': {context}")

def _tool_use(**kwargs):
    kwargs['max_tool_calls'] = 1
    client = ChatCompletions(kwargs)

    tool_definitions = [
        {
            'type': "function",
            'function': {
                'name': "get_current_time",
                'description': "Get the current time at the users location",
                'parameters': {
                    'type': "object",
                    'properties': {},
                    'additionalProperties': False
                },
                'strict': True
            }
        }
    ]

    names_to_args = {tool['function']['name']: set(tool['function']['parameters']['properties'].keys()) for tool in tool_definitions}

    success, message = client.set_tools(tools=tool_definitions)
    assert_type_value(obj=success, type_or_value=bool, name="success")
    assert_type_value(obj=message, type_or_value=str, name="message")
    assert_log(expression=success, message=message)

    success, message = client.set_context(mode="insert", messages={'role': "system", 'content': "You are a helpful assistant."})
    assert_type_value(obj=success, type_or_value=bool, name="success")
    assert_type_value(obj=message, type_or_value=str, name="message")
    assert_log(expression=success, message=message)

    success, message, completion = client.prompt(text="Hello!", response_type="text")
    assert_type_value(obj=success, type_or_value=bool, name="success")
    assert_type_value(obj=message, type_or_value=str, name="message")
    assert_log(expression=success, message=message)
    assert_type_value(obj=completion, type_or_value=dict, name="completion")
    assert_keys(obj=completion, keys=['usage', 'text', 'logs'], mode="match", name="completion")
    assert_type_value(obj=completion['text'], type_or_value=str, name="text in completion")
    assert_log(expression=len(completion['text']) > 0, message="Expected text in completion to be non-empty string.")

    success, message, completion = client.prompt(text="Tell me what time it is!", response_type="auto")
    assert_type_value(obj=success, type_or_value=bool, name="success")
    assert_type_value(obj=message, type_or_value=str, name="message")
    assert_log(expression=success, message=message)
    assert_type_value(obj=completion, type_or_value=dict, name="completion")
    assert_keys(obj=completion, keys=['usage', 'tools', 'logs', 'text'], mode="whitelist", name="completion")
    assert_keys(obj=completion, keys=['usage', 'tools', 'logs'], mode="required", name="completion")
    assert_type_value(obj=completion['tools'], type_or_value=list, name="tools in completion")
    assert_log(expression=len(completion['tools']) == 1, message=f"Expected '1' tool call in completion but got '{len(completion['tools'])}'.")
    assert_type_value(obj=completion['tools'][0], type_or_value=dict, name="tool call")
    assert_keys(obj=completion['tools'][0], keys=['id', 'name', 'arguments'], mode="match", name="tool call")
    assert_type_value(obj=completion['tools'][0]['id'], type_or_value=str, name="value of key 'id' in tool call")
    assert_log(expression=len(completion['tools'][0]['id']) > 0, message="Expected tool call ID to be non-empty.")
    assert_type_value(obj=completion['tools'][0]['name'], type_or_value=list(names_to_args.keys()), name="value of key 'name' in tool call")
    assert_log(expression=set(completion['tools'][0]['arguments'].keys()) == names_to_args[completion['tools'][0]['name']], message=f"Expected tool call '{completion['tools'][0]['name']}' to contain arguments {names_to_args[completion['tools'][0]['name']]} but got {set(completion['tools'][0]['arguments'].keys())}.")

    success, message, completion = client.prompt(text="It is exactly four p.m.", response_type="text")
    assert_type_value(obj=success, type_or_value=bool, name="success")
    assert_type_value(obj=message, type_or_value=str, name="message")
    assert_log(expression=success, message=message)
    assert_type_value(obj=completion, type_or_value=dict, name="completion")

    tool_definitions.append({
        'type': "function",
        'function': {
            'name': "get_current_weather",
            'description': "Get the current weather at the users location",
            'parameters': {
                'type': "object",
                'properties': {},
                'additionalProperties': False
            },
            'strict': True
        }
    })

    names_to_args = {tool['function']['name']: set(tool['function']['parameters']['properties'].keys()) for tool in tool_definitions}

    success, message = client.set_tools(tools=tool_definitions)
    assert_type_value(obj=success, type_or_value=bool, name="success")
    assert_type_value(obj=message, type_or_value=str, name="message")
    assert_log(expression=success, message=message)

    success, message, completion = client.prompt(text="Cool, thanks! Are you sure though?", response_type=tool_definitions[0]['function']['name'])
    assert_type_value(obj=success, type_or_value=bool, name="success")
    assert_type_value(obj=message, type_or_value=str, name="message")
    assert_log(expression=success, message=message)
    assert_type_value(obj=completion, type_or_value=dict, name="completion")
    assert_keys(obj=completion, keys=['usage', 'tools', 'logs'], mode="match", name="completion")
    assert_type_value(obj=completion['tools'], type_or_value=list, name="tools in completion")
    assert_log(expression=len(completion['tools']) == 1, message=f"Expected '1' tool call in completion but got '{len(completion['tools'])}'.")
    assert_type_value(obj=completion['tools'][0], type_or_value=dict, name="tool call")
    assert_keys(obj=completion['tools'][0], keys=['id', 'name', 'arguments'], mode="match", name="tool call")
    assert_type_value(obj=completion['tools'][0]['id'], type_or_value=str, name="value of key 'id' in tool call")
    assert_log(expression=len(completion['tools'][0]['id']) > 0, message="Expected tool call ID to be non-empty.")
    assert_type_value(obj=completion['tools'][0]['name'], type_or_value=list(names_to_args.keys())[0], name="value of key 'name' in tool call")
    assert_log(expression=set(completion['tools'][0]['arguments'].keys()) == names_to_args[completion['tools'][0]['name']], message=f"Expected tool call '{completion['tools'][0]['name']}' to contain arguments {names_to_args[completion['tools'][0]['name']]} but got {set(completion['tools'][0]['arguments'].keys())}.")

    success, message, completion = client.prompt(text="Tell me the time and how warm it is!", response_type="always", reset_context=True)
    assert_type_value(obj=success, type_or_value=bool, name="success")
    assert_type_value(obj=message, type_or_value=str, name="message")
    assert_log(expression=success, message=message)
    assert_type_value(obj=completion, type_or_value=dict, name="completion")
    assert_keys(obj=completion, keys=['usage', 'tools', 'logs'], mode="match", name="completion")
    assert_type_value(obj=completion['tools'], type_or_value=list, name="tools in completion")
    assert_log(expression=len(completion['tools']) == 1, message=f"Expected '1' tool call in completion but got '{len(completion['tools'])}'.")
    assert_type_value(obj=completion['tools'][0], type_or_value=dict, name="tool call")
    assert_keys(obj=completion['tools'][0], keys=['id', 'name', 'arguments'], mode="match", name="tool call")
    assert_type_value(obj=completion['tools'][0]['id'], type_or_value=str, name="value of key 'id' in tool call")
    assert_log(expression=len(completion['tools'][0]['id']) > 0, message="Expected tool call ID to be non-empty.")
    assert_type_value(obj=completion['tools'][0]['name'], type_or_value=list(names_to_args.keys()), name="value of key 'name' in tool call")
    assert_log(expression=set(completion['tools'][0]['arguments'].keys()) == names_to_args[completion['tools'][0]['name']], message=f"Expected tool call '{completion['tools'][0]['name']}' to contain arguments {names_to_args[completion['tools'][0]['name']]} but got {set(completion['tools'][0]['arguments'].keys())}.")

    success, message, context = client.get_context()
    assert_type_value(obj=success, type_or_value=bool, name="success")
    assert_type_value(obj=message, type_or_value=str, name="message")
    assert_log(expression=success, message=message)
    assert_type_value(obj=context, type_or_value=list, name="context")
    assert_log(expression=len(context) == 2, message=f"Expected context to contain '2' messages but got '{len(context)}'.")

    success, message = client.set_settings(max_tool_calls=2)
    assert_type_value(obj=success, type_or_value=bool, name="success")
    assert_type_value(obj=message, type_or_value=str, name="message")
    assert_log(expression=success, message=message)

    success, message, completion = client.prompt(text="Tell me the time and how warm it is! Use both tools simultaneously!", response_type="always", reset_context=True)
    assert_type_value(obj=success, type_or_value=bool, name="success")
    assert_type_value(obj=message, type_or_value=str, name="message")
    assert_log(expression=success, message=message)
    assert_type_value(obj=completion, type_or_value=dict, name="completion")
    assert_keys(obj=completion, keys=['usage', 'tools', 'logs'], mode="match", name="completion")
    assert_type_value(obj=completion['tools'], type_or_value=list, name="tools in completion")
    assert_log(expression=len(completion['tools']) == 2, message=f"Expected '2' tool calls in completion but got '{len(completion['tools'])}'.")
    for tool in completion['tools']:
        assert_type_value(obj=tool, type_or_value=dict, name="tool call")
        assert_keys(obj=tool, keys=['id', 'name', 'arguments'], mode="match", name="tool call")
        assert_type_value(obj=tool['id'], type_or_value=str, name="value of key 'id' in tool call")
        assert_log(expression=len(tool['id']) > 0, message="Expected tool call ID to be non-empty.")
        assert_type_value(obj=tool['name'], type_or_value=list(names_to_args.keys()), name="value of key 'name' in tool call")
        assert_log(expression=set(tool['arguments'].keys()) == names_to_args[tool['name']], message=f"Expected tool call '{tool['name']}' to contain arguments {names_to_args[tool['name']]} but got {set(tool['arguments'].keys())}.")

    responses = []
    for tool in completion['tools']:
        if tool['name'] == tool_definitions[0]['function']['name']:
            responses.append({'role': "tool", 'tool_call_id': tool['id'], 'content': "It is exactly four p.m."})
        else:
            responses.append({'role': "tool", 'tool_call_id': tool['id'], 'content': "It is 3°C"})

    success, message, completion = client.prompt(text=responses, response_type="text")
    assert_type_value(obj=success, type_or_value=bool, name="success")
    assert_type_value(obj=message, type_or_value=str, name="message")
    assert_log(expression=success, message=message)
    assert_type_value(obj=completion, type_or_value=dict, name="completion")
    assert_keys(obj=completion, keys=['usage', 'text', 'logs'], mode="match", name="completion")
    assert_type_value(obj=completion['text'], type_or_value=str, name="text in completion")
    assert_log(expression=len(completion['text']) > 0, message="Expected text in completion to be non-empty string.")

def test_61_openrouter_tool_use():
    return _tool_use(stream=False, endpoint="OpenRouter", model="google/gemini-3-flash-preview")

def test_62_openrouter_tool_use_stream():
    return _tool_use(stream=True, endpoint="OpenRouter", model="google/gemini-3-flash-preview")

def test_63_openai_tool_use():
    return _tool_use(stream=False, endpoint="OpenAI", model="gpt-5.2-2025-12-11")

def test_64_openai_tool_use_stream():
    return _tool_use(stream=True, endpoint="OpenAI", model="gpt-5.2-2025-12-11")

def test_65_mistral_tool_use():
    return _tool_use(stream=False, endpoint="Mistral", model="mistral-large-2512")

def test_66_mistral_tool_use_stream():
    return _tool_use(stream=True, endpoint="Mistral", model="mistral-large-2512")

def test_67_vllm_tool_use():
    return _tool_use(endpoint="AIS", model="ais/qwen3.5-27b", timeout_read=60, timeout_completion=60)

def test_68_vllm_tool_use_stream():
    return _tool_use(endpoint="AIS", model="ais/qwen3.5-27b", timeout_read=60, timeout_completion=60)
