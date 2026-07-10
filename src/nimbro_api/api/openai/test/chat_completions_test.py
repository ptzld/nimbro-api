import os
import json
import math
import time
import threading
import multiprocessing

import nimbro_api
from nimbro_api.utility.misc import UnrecoverableError, assert_type_value, assert_keys, assert_log
from ..client.chat_completions import ChatCompletions

def test_001_utilities():
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

def test_002_endpoint():
    client = ChatCompletions()

    success, message, models = client.get_models()
    assert_type_value(obj=success, type_or_value=bool, name="success")
    assert_type_value(obj=message, type_or_value=str, name="message")
    assert_log(expression=success, message=message)
    assert_type_value(obj=models, type_or_value=list, name="result of get_models()")

# text completions

def _text_completions(**kwargs):
    kwargs['parser'] = []
    client = ChatCompletions(kwargs)

    # text completion
    text = "Hi!"
    success, message, completion = client.prompt(text=text, response_type="text")
    assert_type_value(obj=success, type_or_value=bool, name="success")
    assert_type_value(obj=message, type_or_value=str, name="message")
    assert_log(expression=success, message=message)
    assert_type_value(obj=completion, type_or_value=dict, name="completion")
    assert_keys(obj=completion, keys=['usage', 'text', 'logs', 'reasoning'], mode="whitelist", name="completion")
    assert_keys(obj=completion, keys=['usage', 'text', 'logs'], mode="required", name="completion")
    assert_type_value(obj=completion['usage'], type_or_value=dict, name="usage in completion")
    assert_type_value(obj=completion['text'], type_or_value=str, name="text in completion")
    assert_log(expression=len(completion['text']) > 0, message="Expected text in completion to be a non-empty string.")
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
    assert_keys(obj=completion, keys=['usage', 'text', 'logs', 'reasoning'], mode="whitelist", name="completion")
    assert_keys(obj=completion, keys=['usage', 'text', 'logs'], mode="required", name="completion")
    assert_type_value(obj=completion['usage'], type_or_value=dict, name="usage in completion")
    assert_type_value(obj=completion['text'], type_or_value=str, name="text in completion")
    assert_log(expression=len(completion['text']) > 0, message="Expected text in completion to be a non-empty string.")
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
    assert_keys(obj=completion, keys=['usage', 'text', 'logs', 'reasoning'], mode="whitelist", name="completion")
    assert_keys(obj=completion, keys=['usage', 'text', 'logs'], mode="required", name="completion")
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

def test_003_openrouter_text_completions():
    return _text_completions(stream=False, endpoint="OpenRouter", model="~google/gemini-flash-latest", reasoning_effort="minimal")

def test_004_openrouter_text_completions_stream():
    return _text_completions(stream=True, endpoint="OpenRouter", model="~google/gemini-flash-latest", reasoning_effort="minimal")

def test_005_openai_text_completions():
    return _text_completions(stream=False, endpoint="OpenAI", model="gpt-5.6-luna", reasoning_effort="none")

def test_006_openai_text_completions_stream():
    return _text_completions(stream=True, endpoint="OpenAI", model="gpt-5.6-luna", reasoning_effort="none")

def test_007_mistral_text_completions():
    return _text_completions(stream=False, endpoint="Mistral", model="mistral-small-latest", reasoning_effort="none")

def test_008_mistral_text_completions_stream():
    return _text_completions(stream=True, endpoint="Mistral", model="mistral-small-latest", reasoning_effort="none")

def test_009_vllm_text_completions():
    success, message, models = ChatCompletions(endpoint="AIS").get_models()
    assert_log(expression=success, message=message)
    assert_type_value(obj=models, type_or_value=list, name="models")
    if len(models) == 0:
        return "AIS endpoint is not hosting any models."
    model = models[-1]
    return _text_completions(stream=False, endpoint="AIS", model=model, reasoning_effort="none", timeout_read=60, timeout_completion=60)

def test_010_vllm_text_completions_stream():
    success, message, models = ChatCompletions(endpoint="AIS").get_models()
    assert_log(expression=success, message=message)
    assert_type_value(obj=models, type_or_value=list, name="models")
    if len(models) == 0:
        return "AIS endpoint is not hosting any models."
    model = models[-1]
    return _text_completions(stream=True, endpoint="AIS", model=model, reasoning_effort="none", timeout_read=60, timeout_completion=60)

# text completions choices

def _text_completions_choices(**kwargs):
    assert_log(expression='choices' in kwargs, message="Expected setting 'choices' to be provided by test definition.")
    assert_log(expression=kwargs['choices'] > 1, message=f"Expected setting 'choices' provided by test definition to be greater '1' but got '{kwargs['choices']}'.")
    kwargs['parser'] = []
    client = ChatCompletions(kwargs)

    # text completion
    text = "Hi!"
    success, message, completion = client.prompt(text=text, response_type="text")
    assert_type_value(obj=success, type_or_value=bool, name="success")
    assert_type_value(obj=message, type_or_value=str, name="message")
    assert_log(expression=success, message=message)
    assert_type_value(obj=completion, type_or_value=dict, name="completion")
    assert_keys(obj=completion, keys=['usage', 'choices', 'logs'], mode="match", name="completion")
    assert_type_value(obj=completion['usage'], type_or_value=dict, name="usage in completion")
    assert_type_value(obj=completion['choices'], type_or_value=list, name="choices in completion")
    assert_log(expression=len(completion['choices']) == kwargs['choices'], message=f"Expected number of choices in completion to match setting 'choices' provided by test definition but got '{len(completion['choices'])}' instead of '{kwargs['choices']}'.")
    for i, choice in enumerate(completion['choices']):
        assert_type_value(obj=choice, type_or_value=dict, name=f"choice '{i + 1}' of '{len(completion['choices'])}' in completion")
        assert_keys(obj=choice, keys=['text', 'reasoning', 'logs'], mode="whitelist", name=f"choice '{i + 1}' of '{len(completion['choices'])}' in completion")
        assert_keys(obj=choice, keys=['text'], mode="required", name=f"choice '{i + 1}' of '{len(completion['choices'])}' in completion")
        if 'text' in choice:
            assert_type_value(obj=choice['text'], type_or_value=str, name=f"text in choice '{i + 1}' of '{len(completion['choices'])}' in completion")
            assert_log(expression=len(choice['text']) > 0, message=f"Expected text in choice '{i + 1}' of '{len(completion['choices'])}' in completion to be a non-empty string.")
        if 'reasoning' in choice:
            assert_type_value(obj=choice['reasoning'], type_or_value=str, name=f"reasoning in choice '{i + 1}' of '{len(completion['choices'])}' in completion")
            assert_log(expression=len(choice['reasoning']) > 0, message=f"Expected reasoning in choice '{i + 1}' of '{len(completion['choices'])}' in completion to be a non-empty string.")
        if 'logs' in choice:
            assert_type_value(obj=choice['logs'], type_or_value=list, name=f"logs in choice '{i + 1}' of '{len(completion['choices'])}' in completion")
            assert_log(expression=len(choice['logs']) > 0, message=f"Expected to find at least one log in in choice '{i + 1}' of '{len(completion['choices'])}' completion.")
            for i, log in enumerate(choice['logs']):
                assert_type_value(obj=log, type_or_value=str, name=f"log '{i}' in choice '{i + 1}' of '{len(completion['choices'])}' in completion logs")
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
    assert_log(expression=len(context) == 1, message=f"Expected context to contain '1' messages but got '{len(context)}'.")

    # auto completion
    success, message, completion = client.prompt(text=text, reset_context=True, response_type="auto")
    assert_type_value(obj=success, type_or_value=bool, name="success")
    assert_type_value(obj=message, type_or_value=str, name="message")
    assert_log(expression=success, message=message)
    assert_type_value(obj=completion, type_or_value=dict, name="completion")
    assert_keys(obj=completion, keys=['usage', 'choices', 'logs'], mode="match", name="completion")
    assert_type_value(obj=completion['usage'], type_or_value=dict, name="usage in completion")
    assert_type_value(obj=completion['choices'], type_or_value=list, name="choices in completion")
    assert_log(expression=len(completion['choices']) == kwargs['choices'], message=f"Expected number of choices in completion to match setting 'choices' provided by test definition but got '{len(completion['choices'])}' instead of '{kwargs['choices']}'.")
    for i, choice in enumerate(completion['choices']):
        assert_type_value(obj=choice, type_or_value=dict, name=f"choice '{i + 1}' of '{len(completion['choices'])}' in completion")
        assert_keys(obj=choice, keys=['text', 'reasoning', 'logs'], mode="whitelist", name=f"choice '{i + 1}' of '{len(completion['choices'])}' in completion")
        assert_keys(obj=choice, keys=['text'], mode="required", name=f"choice '{i + 1}' of '{len(completion['choices'])}' in completion")
        if 'text' in choice:
            assert_type_value(obj=choice['text'], type_or_value=str, name=f"text in choice '{i + 1}' of '{len(completion['choices'])}' in completion")
            assert_log(expression=len(choice['text']) > 0, message=f"Expected text in choice '{i + 1}' of '{len(completion['choices'])}' in completion to be a non-empty string.")
        if 'reasoning' in choice:
            assert_type_value(obj=choice['reasoning'], type_or_value=str, name=f"reasoning in choice '{i + 1}' of '{len(completion['choices'])}' in completion")
            assert_log(expression=len(choice['reasoning']) > 0, message=f"Expected reasoning in choice '{i + 1}' of '{len(completion['choices'])}' in completion to be a non-empty string.")
        if 'logs' in choice:
            assert_type_value(obj=choice['logs'], type_or_value=list, name=f"logs in choice '{i + 1}' of '{len(completion['choices'])}' in completion")
            assert_log(expression=len(choice['logs']) > 0, message=f"Expected to find at least one log in in choice '{i + 1}' of '{len(completion['choices'])}' completion.")
            for i, log in enumerate(choice['logs']):
                assert_type_value(obj=log, type_or_value=str, name=f"log '{i}' in choice '{i + 1}' of '{len(completion['choices'])}' in completion logs")
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
    assert_log(expression=len(context) == 1, message=f"Expected context to contain '1' messages but got '{len(context)}'.")

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
    assert_keys(obj=completion, keys=['usage', 'choices', 'logs'], mode="match", name="completion")
    assert_type_value(obj=completion['usage'], type_or_value=dict, name="usage in completion")
    assert_type_value(obj=completion['choices'], type_or_value=list, name="choices in completion")
    assert_log(expression=len(completion['choices']) == kwargs['choices'], message=f"Expected number of choices in completion to match setting 'choices' provided by test definition but got '{len(completion['choices'])}' instead of '{kwargs['choices']}'.")
    for i, choice in enumerate(completion['choices']):
        assert_type_value(obj=choice, type_or_value=dict, name=f"choice '{i + 1}' of '{len(completion['choices'])}' in completion")
        assert_keys(obj=choice, keys=['text', 'reasoning', 'logs'], mode="whitelist", name=f"choice '{i + 1}' of '{len(completion['choices'])}' in completion")
        assert_keys(obj=choice, keys=['text'], mode="required", name=f"choice '{i + 1}' of '{len(completion['choices'])}' in completion")
        if 'text' in choice:
            assert_type_value(obj=choice['text'], type_or_value=str, name=f"text in choice '{i + 1}' of '{len(completion['choices'])}' in completion")
            assert_log(expression=len(choice['text']) > 0, message=f"Expected text in choice '{i + 1}' of '{len(completion['choices'])}' in completion to be a non-empty string.")
        if 'reasoning' in choice:
            assert_type_value(obj=choice['reasoning'], type_or_value=str, name=f"reasoning in choice '{i + 1}' of '{len(completion['choices'])}' in completion")
            assert_log(expression=len(choice['reasoning']) > 0, message=f"Expected reasoning in choice '{i + 1}' of '{len(completion['choices'])}' in completion to be a non-empty string.")
        if 'logs' in choice:
            assert_type_value(obj=choice['logs'], type_or_value=list, name=f"logs in choice '{i + 1}' of '{len(completion['choices'])}' in completion")
            assert_log(expression=len(choice['logs']) > 0, message=f"Expected to find at least one log in in choice '{i + 1}' of '{len(completion['choices'])}' completion.")
            for i, log in enumerate(choice['logs']):
                assert_type_value(obj=log, type_or_value=str, name=f"log '{i}' in choice '{i + 1}' of '{len(completion['choices'])}' in completion logs")
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
    assert_log(expression=len(context) == 1, message=f"Expected context to contain '1' messages but got '{len(context)}'.")
    assert_log(expression=context[0] == target, message=f"Expected message '1' in context to be {target} but got '{context[0]}'.")

def test_011_openrouter_text_completions_choices():
    return "NOT SUPPORTED."

def test_012_openrouter_text_completions_choices_stream():
    return "NOT SUPPORTED."

def test_013_openai_text_completions_choices():
    return _text_completions_choices(stream=False, endpoint="OpenAI", model="gpt-5.6-luna", choices=2, reasoning_effort="none")

def test_014_openai_text_completions_choices_stream():
    return _text_completions_choices(stream=True, endpoint="OpenAI", model="gpt-5.6-luna", choices=2, reasoning_effort="none")

def test_015_mistral_text_completions_choices():
    return _text_completions_choices(stream=False, endpoint="Mistral", model="mistral-small-latest", choices=2, reasoning_effort="none")

def test_016_mistral_text_completions_choices_stream():
    return _text_completions_choices(stream=True, endpoint="Mistral", model="mistral-small-latest", choices=2, reasoning_effort="none")

def test_017_vllm_text_completions_choices():
    success, message, models = ChatCompletions(endpoint="AIS").get_models()
    assert_log(expression=success, message=message)
    assert_type_value(obj=models, type_or_value=list, name="models")
    if len(models) == 0:
        return "AIS endpoint is not hosting any models."
    model = models[-1]
    return _text_completions_choices(stream=False, endpoint="AIS", model=model, choices=2, reasoning_effort="none", timeout_read=60, timeout_completion=60)

def test_018_vllm_text_completions_choices_stream():
    success, message, models = ChatCompletions(endpoint="AIS").get_models()
    assert_log(expression=success, message=message)
    assert_type_value(obj=models, type_or_value=list, name="models")
    if len(models) == 0:
        return "AIS endpoint is not hosting any models."
    model = models[-1]
    return _text_completions_choices(stream=True, endpoint="AIS", model=model, choices=2, reasoning_effort="none", timeout_read=60, timeout_completion=60)

# JSON mode

def _json_completion(**kwargs):
    kwargs['parser'] = []
    client = ChatCompletions(kwargs)

    # JSON mode completion
    text = "Tell me a joke in JSON format!"
    success, message, completion = client.prompt(text=text, response_type="json")
    assert_type_value(obj=success, type_or_value=bool, name="success")
    assert_type_value(obj=message, type_or_value=str, name="message")
    assert_log(expression=success, message=message)
    assert_type_value(obj=completion, type_or_value=dict, name="completion")
    assert_keys(obj=completion, keys=['usage', 'text', 'logs', 'reasoning'], mode="whitelist", name="completion")
    assert_keys(obj=completion, keys=['usage', 'text', 'logs'], mode="required", name="completion")
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
    assert_log(expression=len(context) == 2, message=f"Expected context to contain '2' messages but got '{len(context)}'.")
    for i, message in enumerate(context):
        assert_type_value(obj=message, type_or_value=dict, name=f"message '{i}' in context")
    target = {'role': 'user', 'content': [{'type': 'text', 'text': text}]}
    assert_log(expression=context[0] == target, message=f"Expected message '0' in context to be {target} but got {context[0]}.")
    assert_keys(obj=context[1], keys=['role', 'content'], mode="match", name="message '3' in context")
    try:
        context[1]['content'] = json.loads(context[1]['content'])
    except Exception as e:
        raise UnrecoverableError(f"Expected message '3' in context to be JSON-compliant but '{context[1]}' is not: {repr(e)}") from e
    target = {'role': "assistant", 'content': completion['text']}
    assert_log(expression=context[1] == target, message=f"Expected message '3' in context to be {target} but got {context[1]}.")

def test_019_openrouter_json():
    return _text_completions(stream=False, endpoint="OpenRouter", model="~google/gemini-flash-latest", reasoning_effort="minimal")

def test_020_openrouter_json_stream():
    return _text_completions(stream=True, endpoint="OpenRouter", model="~google/gemini-flash-latest", reasoning_effort="minimal")

def test_021_openai_json():
    return _text_completions(stream=False, endpoint="OpenAI", model="gpt-5.6-luna", reasoning_effort="none")

def test_022_openai_json_stream():
    return _text_completions(stream=True, endpoint="OpenAI", model="gpt-5.6-luna", reasoning_effort="none")

def test_023_mistral_json():
    return _text_completions(stream=False, endpoint="Mistral", model="mistral-small-latest", reasoning_effort="none")

def test_024_mistral_json_stream():
    return _text_completions(stream=True, endpoint="Mistral", model="mistral-small-latest", reasoning_effort="none")

def test_025_vllm_json():
    success, message, models = ChatCompletions(endpoint="AIS").get_models()
    assert_log(expression=success, message=message)
    assert_type_value(obj=models, type_or_value=list, name="models")
    if len(models) == 0:
        return "AIS endpoint is not hosting any models."
    model = models[-1]
    return _text_completions(stream=False, endpoint="AIS", model=model, timeout_read=60, timeout_completion=60)

def test_026_vllm_json_stream():
    success, message, models = ChatCompletions(endpoint="AIS").get_models()
    assert_log(expression=success, message=message)
    assert_type_value(obj=models, type_or_value=list, name="models")
    if len(models) == 0:
        return "AIS endpoint is not hosting any models."
    model = models[-1]
    return _text_completions(stream=True, endpoint="AIS", model=model, timeout_read=60, timeout_completion=60)

# reasoning

def _reasoning_completion(**kwargs):
    assert_log(expression='reasoning_effort' in kwargs, message="Expected setting 'reasoning_effort' to be provided by test definition.")
    assert_log(expression=kwargs['reasoning_effort'] not in ['', 'none'], message=f"Expected setting 'reasoning_effort' provided by test definition to be anything but '' or 'none' but got '{kwargs['reasoning_effort']}'.")
    client = ChatCompletions(kwargs)

    # text = (
    #     "Five distinct tasks A–E must be assigned to five consecutive slots, "
    #     "one task per slot.\n\n"
    #     "- A occurs before D.\n"
    #     "- B occurs immediately after C.\n"
    #     "- E occurs before A.\n"
    #     "- D is not last.\n\n"
    #     "How many valid schedules exist?\n\n"
    #     "Reason through the constraints internally, check your result, and output "
    #     "only the final integer. Do not reveal calculations or intermediate steps."
    # )
    text = "Tell me a joke about computer scientists. Briefly reason which one you should use and then only respond with the joke."
    success, message, completion = client.prompt(text=text)

    assert_type_value(obj=success, type_or_value=bool, name="success")
    assert_type_value(obj=message, type_or_value=str, name="message")
    assert_log(expression=success, message=message)
    assert_type_value(obj=completion, type_or_value=dict, name="completion")
    assert_keys(obj=completion, keys=['usage', 'text', 'reasoning', 'logs'], mode="match", name="completion")
    assert_type_value(obj=completion['usage'], type_or_value=dict, name="usage in completion")
    assert_type_value(obj=completion['text'], type_or_value=str, name="text in completion")
    assert_log(expression=len(completion['text']) > 0, message="Expected text in completion to be a non-empty string.")
    assert_type_value(obj=completion['reasoning'], type_or_value=str, name="reasoning in completion")
    assert_log(expression=len(completion['reasoning']) > 0, message="Expected reasoning in completion to be a non-empty string.")
    assert_type_value(obj=completion['logs'], type_or_value=list, name="logs in completion")
    assert_log(expression=len(completion['logs']) > 0, message="Expected to find at least one log in completion.")
    for i, log in enumerate(completion['logs']):
        assert_type_value(obj=log, type_or_value=str, name=f"log '{i}' in completion logs")

def test_027_openrouter_reasoning():
    return _reasoning_completion(stream=False, endpoint="OpenRouter", model="~google/gemini-flash-latest", reasoning_effort="high")

def test_028_openrouter_reasoning_stream():
    return _reasoning_completion(stream=True, endpoint="OpenRouter", model="~google/gemini-flash-latest", reasoning_effort="high")

def test_029_openai_reasoning():
    return "NOT SUPPORTED."

def test_030_openai_reasoning_stream():
    return "NOT SUPPORTED."

def test_031_mistral_reasoning():
    return _reasoning_completion(stream=False, endpoint="Mistral", model="mistral-small-latest", reasoning_effort="high")

def test_032_mistral_reasoning_stream():
    return _reasoning_completion(stream=True, endpoint="Mistral", model="mistral-small-latest", reasoning_effort="high")

def test_033_vllm_reasoning():
    success, message, models = ChatCompletions(endpoint="AIS").get_models()
    assert_log(expression=success, message=message)
    assert_type_value(obj=models, type_or_value=list, name="models")
    if len(models) == 0:
        return "AIS endpoint is not hosting any models."
    model = models[-1]
    return _reasoning_completion(stream=False, endpoint="AIS", model=model, reasoning_effort="high", timeout_read=60, timeout_completion=60)

def test_034_vllm_reasoning_stream():
    success, message, models = ChatCompletions(endpoint="AIS").get_models()
    assert_log(expression=success, message=message)
    assert_type_value(obj=models, type_or_value=list, name="models")
    if len(models) == 0:
        return "AIS endpoint is not hosting any models."
    model = models[-1]
    return _reasoning_completion(stream=True, endpoint="AIS", model=model, reasoning_effort="high", timeout_read=60, timeout_completion=60)

# reasoning choices

def _reasoning_choices_completion(**kwargs):
    assert_log(expression='reasoning_effort' in kwargs, message="Expected setting 'reasoning_effort' to be provided by test definition.")
    assert_log(expression=kwargs['reasoning_effort'] not in ['', 'none'], message=f"Expected setting 'reasoning_effort' provided by test definition to be anything but '' or 'none' but got '{kwargs['reasoning_effort']}'.")
    assert_log(expression='choices' in kwargs, message="Expected setting 'choices' to be provided by test definition.")
    assert_log(expression=kwargs['choices'] > 1, message=f"Expected setting 'choices' provided by test definition to be greater '1' but got '{kwargs['choices']}'.")
    client = ChatCompletions(kwargs)

    # text = (
    #     "Five distinct tasks A–E must be assigned to five consecutive slots, "
    #     "one task per slot.\n\n"
    #     "- A occurs before D.\n"
    #     "- B occurs immediately after C.\n"
    #     "- E occurs before A.\n"
    #     "- D is not last.\n\n"
    #     "How many valid schedules exist?\n\n"
    #     "Reason through the constraints internally, check your result, and output "
    #     "only the final integer. Do not reveal calculations or intermediate steps."
    # )
    text = "Tell me a joke about computer scientists. Briefly reason which one you should use and then only respond with the joke."
    success, message, completion = client.prompt(text=text)

    assert_type_value(obj=success, type_or_value=bool, name="success")
    assert_type_value(obj=message, type_or_value=str, name="message")
    assert_log(expression=success, message=message)
    assert_type_value(obj=completion, type_or_value=dict, name="completion")
    assert_keys(obj=completion, keys=['usage', 'choices', 'logs'], mode="match", name="completion")
    assert_type_value(obj=completion['usage'], type_or_value=dict, name="usage in completion")
    assert_type_value(obj=completion['choices'], type_or_value=list, name="choices in completion")
    assert_log(expression=len(completion['choices']) == kwargs['choices'], message=f"Expected number of choices in completion to match setting 'choices' provided by test definition but got '{len(completion['choices'])}' instead of '{kwargs['choices']}'.")
    for i, choice in enumerate(completion['choices']):
        assert_type_value(obj=choice, type_or_value=dict, name=f"choice '{i + 1}' of '{len(completion['choices'])}' in completion")
        assert_keys(obj=choice, keys=['text', 'reasoning', 'logs'], mode="whitelist", name=f"choice '{i + 1}' of '{len(completion['choices'])}' in completion")
        assert_keys(obj=choice, keys=['text', 'reasoning'], mode="required", name=f"choice '{i + 1}' of '{len(completion['choices'])}' in completion")
        if 'text' in choice:
            assert_type_value(obj=choice['text'], type_or_value=str, name=f"text in choice '{i + 1}' of '{len(completion['choices'])}' in completion")
            assert_log(expression=len(choice['text']) > 0, message=f"Expected text in choice '{i + 1}' of '{len(completion['choices'])}' in completion to be a non-empty string.")
        if 'reasoning' in choice:
            assert_type_value(obj=choice['reasoning'], type_or_value=str, name=f"reasoning in choice '{i + 1}' of '{len(completion['choices'])}' in completion")
            assert_log(expression=len(choice['reasoning']) > 0, message=f"Expected reasoning in choice '{i + 1}' of '{len(completion['choices'])}' in completion to be a non-empty string.")
        if 'logs' in choice:
            assert_type_value(obj=choice['logs'], type_or_value=list, name=f"logs in choice '{i + 1}' of '{len(completion['choices'])}' in completion")
            assert_log(expression=len(choice['logs']) > 0, message=f"Expected to find at least one log in in choice '{i + 1}' of '{len(completion['choices'])}' completion.")
            for i, log in enumerate(choice['logs']):
                assert_type_value(obj=log, type_or_value=str, name=f"log '{i}' in choice '{i + 1}' of '{len(completion['choices'])}' in completion logs")
    assert_type_value(obj=completion['logs'], type_or_value=list, name="logs in completion")
    assert_log(expression=len(completion['logs']) > 0, message="Expected to find at least one log in completion.")
    for i, log in enumerate(completion['logs']):
        assert_type_value(obj=log, type_or_value=str, name=f"log '{i}' in completion logs")

def test_035_openrouter_reasoning_choices():
    return "NOT SUPPORTED."

def test_036_openrouter_reasoning_choices_stream():
    return "NOT SUPPORTED."

def test_037_openai_reasoning_choices():
    return "NOT SUPPORTED."

def test_038_openai_reasoning_choices_stream():
    return "NOT SUPPORTED."

def test_039_mistral_reasoning_choices():
    return _reasoning_choices_completion(stream=False, endpoint="Mistral", model="mistral-small-latest", choices=2, reasoning_effort="high")

def test_040_mistral_reasoning_choices_stream():
    return _reasoning_choices_completion(stream=True, endpoint="Mistral", model="mistral-small-latest", choices=2, reasoning_effort="high")

def test_041_vllm_reasoning_choices():
    success, message, models = ChatCompletions(endpoint="AIS").get_models()
    assert_log(expression=success, message=message)
    assert_type_value(obj=models, type_or_value=list, name="models")
    if len(models) == 0:
        return "AIS endpoint is not hosting any models."
    model = models[-1]
    return _reasoning_choices_completion(stream=False, endpoint="AIS", model=model, choices=2, reasoning_effort="high", timeout_read=60, timeout_completion=60)

def test_042_vllm_reasoning_choices_stream():
    success, message, models = ChatCompletions(endpoint="AIS").get_models()
    assert_log(expression=success, message=message)
    assert_type_value(obj=models, type_or_value=list, name="models")
    if len(models) == 0:
        return "AIS endpoint is not hosting any models."
    model = models[-1]
    return _reasoning_choices_completion(stream=True, endpoint="AIS", model=model, choices=2, reasoning_effort="high", timeout_read=60, timeout_completion=60)

# interrupt

def _interrupt(**kwargs):
    client = ChatCompletions(validate_model=False, retry=10, **kwargs)

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

def test_043_interrupt():
    return "NOT SUPPORTED."

def test_044_interrupt_stream():
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
    assert_keys(obj=completion, keys=['usage', 'text', 'logs', 'reasoning'], mode="whitelist", name="completion")
    assert_keys(obj=completion, keys=['usage', 'text', 'logs'], mode="required", name="completion")
    assert_type_value(obj=completion['usage'], type_or_value=dict, name="usage in completion")
    assert_type_value(obj=completion['text'], type_or_value=str, name="text in completion")
    assert_log(expression=len(completion['text']) > 0, message="Expected text in completion to be a non-empty string.")
    assert_type_value(obj=completion['logs'], type_or_value=list, name="logs in completion")
    assert_log(expression=len(completion['logs']) > 0, message="Expected to find at least one log in completion.")
    for i, log in enumerate(completion['logs']):
        assert_type_value(obj=log, type_or_value=str, name=f"log '{i}' in completion logs")

    return completion['text']

def test_045_openrouter_web_search():
    return _web_search(endpoint="OpenRouter", model="~google/gemini-flash-latest:online", reasoning_effort="minimal")

def test_046_openai_web_search():
    return "NOT SUPPORTED."

def test_047_mistral_web_search():
    return "NOT SUPPORTED."

def test_048_vllm_web_search():
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
    assert_keys(obj=completion, keys=['usage', 'text', 'logs', 'reasoning'], mode="whitelist", name="completion")
    assert_keys(obj=completion, keys=['usage', 'text', 'logs'], mode="required", name="completion")
    assert_type_value(obj=completion['usage'], type_or_value=dict, name="usage in completion")
    assert_type_value(obj=completion['text'], type_or_value=str, name="text in completion")
    assert_log(expression=len(completion['text']) > 0, message="Expected text in completion to be a non-empty string.")
    assert_type_value(obj=completion['logs'], type_or_value=list, name="logs in completion")
    assert_log(expression=len(completion['logs']) > 0, message="Expected to find at least one log in completion.")
    for i, log in enumerate(completion['logs']):
        assert_type_value(obj=log, type_or_value=str, name=f"log '{i}' in completion logs")

    return completion['text']

def test_049_openrouter_image_file_input():
    return _image_file_input(endpoint="OpenRouter", model="~google/gemini-flash-latest", reasoning_effort="minimal")

def test_050_openai_image_file_input():
    return _image_file_input(endpoint="OpenAI", model="gpt-5.6-luna", reasoning_effort="none")

def test_051_mistral_image_file_input():
    return _image_file_input(endpoint="Mistral", model="mistral-small-latest", reasoning_effort="none")

def test_052_vllm_image_file_input():
    success, message, models = ChatCompletions(endpoint="AIS").get_models()
    assert_log(expression=success, message=message)
    assert_type_value(obj=models, type_or_value=list, name="models")
    if len(models) == 0:
        return "AIS endpoint is not hosting any models."
    model = models[-1]
    return _image_file_input(endpoint="AIS", model=model, reasoning_effort="none", timeout_read=60, timeout_completion=60)

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
    assert_keys(obj=completion, keys=['usage', 'text', 'logs', 'reasoning'], mode="whitelist", name="completion")
    assert_keys(obj=completion, keys=['usage', 'text', 'logs'], mode="required", name="completion")
    assert_type_value(obj=completion['usage'], type_or_value=dict, name="usage in completion")
    assert_type_value(obj=completion['text'], type_or_value=str, name="text in completion")
    assert_log(expression=len(completion['text']) > 0, message="Expected text in completion to be a non-empty string.")
    assert_type_value(obj=completion['logs'], type_or_value=list, name="logs in completion")
    assert_log(expression=len(completion['logs']) > 0, message="Expected to find at least one log in completion.")
    for i, log in enumerate(completion['logs']):
        assert_type_value(obj=log, type_or_value=str, name=f"log '{i}' in completion logs")

    return completion['text']

def test_053_openrouter_image_url_input():
    return _image_url_input(endpoint="OpenRouter", model="~google/gemini-flash-latest", reasoning_effort="minimal")

def test_054_openai_image_url_input():
    return _image_url_input(endpoint="OpenAI", model="gpt-5.6-luna", reasoning_effort="none")

def test_055_mistral_image_url_input():
    return _image_url_input(endpoint="Mistral", model="mistral-small-latest", reasoning_effort="none")

def test_056_vllm_image_url_input():
    success, message, models = ChatCompletions(endpoint="AIS").get_models()
    assert_log(expression=success, message=message)
    assert_type_value(obj=models, type_or_value=list, name="models")
    if len(models) == 0:
        return "AIS endpoint is not hosting any models."
    model = models[-1]
    return _image_url_input(endpoint="AIS", model=model, reasoning_effort="none", timeout_read=60, timeout_completion=60)

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
    assert_keys(obj=completion, keys=['usage', 'text', 'logs', 'reasoning'], mode="whitelist", name="completion")
    assert_keys(obj=completion, keys=['usage', 'text', 'logs'], mode="required", name="completion")
    assert_type_value(obj=completion['usage'], type_or_value=dict, name="usage in completion")
    assert_type_value(obj=completion['text'], type_or_value=str, name="text in completion")
    assert_log(expression=len(completion['text']) > 0, message="Expected text in completion to be a non-empty string.")
    assert_type_value(obj=completion['logs'], type_or_value=list, name="logs in completion")
    assert_log(expression=len(completion['logs']) > 0, message="Expected to find at least one log in completion.")
    for i, log in enumerate(completion['logs']):
        assert_type_value(obj=log, type_or_value=str, name=f"log '{i}' in completion logs")

    return completion['text']

def test_057_openrouter_audio_file_input():
    return _audio_file_input(endpoint="OpenRouter", model="~google/gemini-flash-latest", reasoning_effort="minimal")

def test_058_openai_audio_file_input():
    return _audio_file_input(endpoint="OpenAI", model="gpt-audio-2025-08-28", reasoning_effort="")

def test_059_mistral_audio_file_input():
    return _audio_file_input(endpoint="Mistral", model="voxtral-small-latest", reasoning_effort="")

def test_060_vllm_audio_file_input():
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
    assert_keys(obj=completion, keys=['usage', 'text', 'logs', 'reasoning'], mode="whitelist", name="completion")
    assert_keys(obj=completion, keys=['usage', 'text', 'logs'], mode="required", name="completion")
    assert_type_value(obj=completion['usage'], type_or_value=dict, name="usage in completion")
    assert_type_value(obj=completion['text'], type_or_value=str, name="text in completion")
    assert_log(expression=len(completion['text']) > 0, message="Expected text in completion to be a non-empty string.")
    assert_type_value(obj=completion['logs'], type_or_value=list, name="logs in completion")
    assert_log(expression=len(completion['logs']) > 0, message="Expected to find at least one log in completion.")
    for i, log in enumerate(completion['logs']):
        assert_type_value(obj=log, type_or_value=str, name=f"log '{i}' in completion logs")

    return completion['text']

def test_061_openrouter_audio_url_input():
    return "NOT SUPPORTED."

def test_062_openai_audio_url_input():
    return "NOT SUPPORTED."

def test_063_mistral_audio_url_input():
    return _audio_url_input(endpoint="Mistral", model="voxtral-small-latest", reasoning_effort="")

def test_064_vllm_audio_url_input():
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
    assert_keys(obj=completion, keys=['usage', 'text', 'logs', 'reasoning'], mode="whitelist", name="completion")
    assert_keys(obj=completion, keys=['usage', 'text', 'logs'], mode="required", name="completion")
    assert_type_value(obj=completion['usage'], type_or_value=dict, name="usage in completion")
    assert_type_value(obj=completion['text'], type_or_value=str, name="text in completion")
    assert_log(expression=len(completion['text']) > 0, message="Expected text in completion to be a non-empty string.")
    assert_type_value(obj=completion['logs'], type_or_value=list, name="logs in completion")
    assert_log(expression=len(completion['logs']) > 0, message="Expected to find at least one log in completion.")
    for i, log in enumerate(completion['logs']):
        assert_type_value(obj=log, type_or_value=str, name=f"log '{i}' in completion logs")

    return completion['text']

def test_065_openrouter_video_file_input():
    return _video_file_input(endpoint="OpenRouter", model="~google/gemini-flash-latest", reasoning_effort="minimal")

def test_066_openai_video_file_input():
    return "NOT SUPPORTED."

def test_067_mistral_video_file_input():
    return "NOT SUPPORTED."

def test_068_vllm_video_file_input():
    # success, message, models = ChatCompletions(endpoint="AIS").get_models()
    # assert_log(expression=success, message=message)
    # assert_type_value(obj=models, type_or_value=list, name="models")
    # if len(models) == 0: return "AIS endpoint is not hosting any models."
    # model = models[-1]
    # return _video_file_input(endpoint="AIS", model=model, reasoning_effort="none", timeout_read=60, timeout_completion=60)
    return "NOT SUPPORTED."

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
    assert_keys(obj=completion, keys=['usage', 'text', 'logs', 'reasoning'], mode="whitelist", name="completion")
    assert_keys(obj=completion, keys=['usage', 'text', 'logs'], mode="required", name="completion")
    assert_type_value(obj=completion['usage'], type_or_value=dict, name="usage in completion")
    assert_type_value(obj=completion['text'], type_or_value=str, name="text in completion")
    assert_log(expression=len(completion['text']) > 0, message="Expected text in completion to be a non-empty string.")
    assert_type_value(obj=completion['logs'], type_or_value=list, name="logs in completion")
    assert_log(expression=len(completion['logs']) > 0, message="Expected to find at least one log in completion.")
    for i, log in enumerate(completion['logs']):
        assert_type_value(obj=log, type_or_value=str, name=f"log '{i}' in completion logs")

    return completion['text']

def test_069_openrouter_video_url_input():
    return _video_url_input(endpoint="OpenRouter", model="~google/gemini-flash-latest", reasoning_effort="minimal")

def test_070_openai_video_url_input():
    return "NOT SUPPORTED."

def test_071_mistral_video_url_input():
    return "NOT SUPPORTED."

def test_072_vllm_video_url_input():
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
    assert_keys(obj=completion, keys=['usage', 'text', 'logs', 'reasoning'], mode="whitelist", name="completion")
    assert_keys(obj=completion, keys=['usage', 'text', 'logs'], mode="required", name="completion")
    assert_type_value(obj=completion['usage'], type_or_value=dict, name="usage in completion")
    assert_type_value(obj=completion['text'], type_or_value=str, name="text in completion")
    assert_log(expression=len(completion['text']) > 0, message="Expected text in completion to be a non-empty string.")
    assert_type_value(obj=completion['logs'], type_or_value=list, name="logs in completion")
    assert_log(expression=len(completion['logs']) > 0, message="Expected to find at least one log in completion.")
    for i, log in enumerate(completion['logs']):
        assert_type_value(obj=log, type_or_value=str, name=f"log '{i}' in completion logs")

    return completion['text']

def test_073_openrouter_file_local_input():
    return _file_local_input(endpoint="OpenRouter", model="~google/gemini-flash-latest", reasoning_effort="minimal")

def test_074_openai_file_local_input():
    return _file_local_input(endpoint="OpenAI", model="gpt-5.6-luna", reasoning_effort="none")

def test_075_mistral_file_local_input():
    return "NOT SUPPORTED."

def test_076_vllm_file_local_input():
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
    assert_keys(obj=completion, keys=['usage', 'text', 'logs', 'reasoning'], mode="whitelist", name="completion")
    assert_keys(obj=completion, keys=['usage', 'text', 'logs'], mode="required", name="completion")
    assert_type_value(obj=completion['usage'], type_or_value=dict, name="usage in completion")
    assert_type_value(obj=completion['text'], type_or_value=str, name="text in completion")
    assert_log(expression=len(completion['text']) > 0, message="Expected text in completion to be a non-empty string.")
    assert_type_value(obj=completion['logs'], type_or_value=list, name="logs in completion")
    assert_log(expression=len(completion['logs']) > 0, message="Expected to find at least one log in completion.")
    for i, log in enumerate(completion['logs']):
        assert_type_value(obj=log, type_or_value=str, name=f"log '{i}' in completion logs")

    return completion['text']

def test_077_openrouter_file_url_input():
    return _file_url_input(endpoint="OpenRouter", model="~google/gemini-flash-latest", reasoning_effort="minimal")

def test_078_openai_file_url_input():
    return "NOT SUPPORTED."

def test_079_mistral_file_url_input():
    return "NOT SUPPORTED."

def test_080_vllm_file_url_input():
    return "NOT SUPPORTED."

# parallel

def test_081_parallel_threads(n=64):
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
        assert_keys(obj=completion, keys=['usage', 'text', 'logs', 'reasoning'], mode="whitelist", name="completion")
        assert_keys(obj=completion, keys=['usage', 'text', 'logs'], mode="required", name="completion")
        assert_type_value(obj=completion['usage'], type_or_value=dict, name="usage in completion")
        assert_keys(obj=completion['usage'], keys=['duration'], mode="required", name="usage in completion")
        assert_type_value(obj=completion['usage']['duration'], type_or_value=float, name="duration of usage in completion")
        assert_log(expression=completion['usage']['duration'] > 0, message=f"Expected duration of usage in completion to be grater zero but got '{completion['usage']['duration']}'.")
        stamps.append(completion['usage']['duration'])

    stamp = time.perf_counter()

    threads, clients, stamps = [], [], []
    for _ in range(n):
        client = ChatCompletions(endpoint="Mistral", model="mistral-small-latest", reasoning_effort="none")
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

    client = ChatCompletions(endpoint="Mistral", model="mistral-small-latest", reasoning_effort="none")
    success, message, completion = client.prompt("Tell me a joke!", response_type="text")
    assert_type_value(obj=success, type_or_value=bool, name="success")
    assert_type_value(obj=message, type_or_value=str, name="message")
    assert_log(expression=success, message=message)
    assert_type_value(obj=completion, type_or_value=dict, name="completion")
    assert_keys(obj=completion, keys=['usage', 'text', 'logs', 'reasoning'], mode="whitelist", name="completion")
    assert_keys(obj=completion, keys=['usage', 'text', 'logs'], mode="required", name="completion")
    assert_type_value(obj=completion['usage'], type_or_value=dict, name="usage in completion")
    assert_keys(obj=completion['usage'], keys=['duration'], mode="required", name="usage in completion")
    assert_type_value(obj=completion['usage']['duration'], type_or_value=float, name="duration of usage in completion")
    assert_log(expression=completion['usage']['duration'] > 0, message=f"Expected duration of usage in completion to be grater zero but got '{completion['usage']['duration']}'.")
    shared_stamps.append(completion['usage']['duration'])

def test_082_parallel_processes(n=32):
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

def test_083_tool_config():
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

def test_084_context_config():
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

## tool use

def _tool_use(**kwargs):
    kwargs['parser'] = []
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
    assert_keys(obj=completion, keys=['usage', 'text', 'logs', 'reasoning'], mode="whitelist", name="completion")
    assert_keys(obj=completion, keys=['usage', 'text', 'logs'], mode="required", name="completion")
    assert_type_value(obj=completion['text'], type_or_value=str, name="text in completion")
    assert_log(expression=len(completion['text']) > 0, message="Expected text in completion to be a non-empty string.")

    success, message, completion = client.prompt(text="Tell me what time it is!", response_type="auto")
    assert_type_value(obj=success, type_or_value=bool, name="success")
    assert_type_value(obj=message, type_or_value=str, name="message")
    assert_log(expression=success, message=message)
    assert_type_value(obj=completion, type_or_value=dict, name="completion")
    assert_keys(obj=completion, keys=['usage', 'tools', 'logs', 'text', 'reasoning'], mode="whitelist", name="completion")
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
    assert_keys(obj=completion, keys=['usage', 'tools', 'logs', 'text', 'reasoning'], mode="whitelist", name="completion")
    assert_keys(obj=completion, keys=['usage', 'tools', 'logs'], mode="required", name="completion")
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
    assert_keys(obj=completion, keys=['usage', 'tools', 'logs', 'text', 'reasoning'], mode="whitelist", name="completion")
    assert_keys(obj=completion, keys=['usage', 'tools', 'logs'], mode="required", name="completion")
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
    assert_keys(obj=completion, keys=['usage', 'tools', 'logs', 'text', 'reasoning'], mode="whitelist", name="completion")
    assert_keys(obj=completion, keys=['usage', 'tools', 'logs'], mode="required", name="completion")
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
    assert_keys(obj=completion, keys=['usage', 'text', 'logs', 'reasoning'], mode="whitelist", name="completion")
    assert_keys(obj=completion, keys=['usage', 'text', 'logs'], mode="required", name="completion")
    assert_type_value(obj=completion['text'], type_or_value=str, name="text in completion")
    assert_log(expression=len(completion['text']) > 0, message="Expected text in completion to be a non-empty string.")

def test_085_openrouter_tool_use():
    return _tool_use(stream=False, endpoint="OpenRouter", model="~google/gemini-flash-latest", reasoning_effort="minimal")

def test_086_openrouter_tool_use_stream():
    return _tool_use(stream=True, endpoint="OpenRouter", model="~google/gemini-flash-latest", reasoning_effort="minimal")

def test_087_openai_tool_use():
    return _tool_use(stream=False, endpoint="OpenAI", model="gpt-5.2-2025-12-11", reasoning_effort="none")

def test_088_openai_tool_use_stream():
    return _tool_use(stream=True, endpoint="OpenAI", model="gpt-5.2-2025-12-11", reasoning_effort="none")

def test_089_mistral_tool_use():
    return _tool_use(stream=False, endpoint="Mistral", model="mistral-small-latest", reasoning_effort="none")

def test_090_mistral_tool_use_stream():
    return _tool_use(stream=True, endpoint="Mistral", model="mistral-small-latest", reasoning_effort="none")

def test_091_vllm_tool_use():
    success, message, models = ChatCompletions(endpoint="AIS").get_models()
    assert_log(expression=success, message=message)
    assert_type_value(obj=models, type_or_value=list, name="models")
    if len(models) == 0:
        return "AIS endpoint is not hosting any models."
    model = models[-1]
    return _tool_use(endpoint="AIS", model=model, reasoning_effort="none", timeout_read=60, timeout_completion=60)

def test_092_vllm_tool_use_stream():
    success, message, models = ChatCompletions(endpoint="AIS").get_models()
    assert_log(expression=success, message=message)
    assert_type_value(obj=models, type_or_value=list, name="models")
    if len(models) == 0:
        return "AIS endpoint is not hosting any models."
    model = models[-1]
    return _tool_use(endpoint="AIS", model=model, reasoning_effort="none", timeout_read=60, timeout_completion=60)

## tool use choices

def _tool_use_choices(**kwargs):
    assert_log(expression='choices' in kwargs, message="Expected setting 'choices' to be provided by test definition.")
    assert_log(expression=kwargs['choices'] > 1, message=f"Expected setting 'choices' provided by test definition to be greater '1' but got '{kwargs['choices']}'.")
    kwargs['parser'] = []
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

    success, message, completion = client.prompt(text="Tell me what time it is!", response_type="always")
    assert_type_value(obj=success, type_or_value=bool, name="success")
    assert_type_value(obj=message, type_or_value=str, name="message")
    assert_log(expression=success, message=message)
    assert_type_value(obj=completion, type_or_value=dict, name="completion")
    assert_keys(obj=completion, keys=['usage', 'choices', 'logs'], mode="match", name="completion")
    assert_type_value(obj=completion['usage'], type_or_value=dict, name="usage in completion")
    assert_type_value(obj=completion['choices'], type_or_value=list, name="choices in completion")
    assert_log(expression=len(completion['choices']) == kwargs['choices'], message=f"Expected number of choices in completion to match setting 'choices' provided by test definition but got '{len(completion['choices'])}' instead of '{kwargs['choices']}'.")
    for i, choice in enumerate(completion['choices']):
        assert_type_value(obj=choice, type_or_value=dict, name=f"choice '{i + 1}' of '{len(completion['choices'])}' in completion")
        assert_keys(obj=choice, keys=['tools', 'text', 'reasoning', 'logs'], mode="whitelist", name=f"choice '{i + 1}' of '{len(completion['choices'])}' in completion")
        assert_keys(obj=choice, keys=['tools'], mode="required", name=f"choice '{i + 1}' of '{len(completion['choices'])}' in completion")
        if 'text' in choice:
            assert_type_value(obj=choice['text'], type_or_value=str, name=f"text in choice '{i + 1}' of '{len(completion['choices'])}' in completion")
            assert_log(expression=len(choice['text']) > 0, message=f"Expected text in choice '{i + 1}' of '{len(completion['choices'])}' in completion to be a non-empty string.")
        if 'reasoning' in choice:
            assert_type_value(obj=choice['reasoning'], type_or_value=str, name=f"reasoning in choice '{i + 1}' of '{len(completion['choices'])}' in completion")
            assert_log(expression=len(choice['reasoning']) > 0, message=f"Expected reasoning in choice '{i + 1}' of '{len(completion['choices'])}' in completion to be a non-empty string.")
        assert_type_value(obj=choice['tools'], type_or_value=list, name=f"tools in choice '{i + 1}' of '{len(completion['choices'])}' in completion")
        assert_log(expression=len(choice['tools']) == 1, message=f"Expected tools in choice '{i + 1}' of '{len(completion['choices'])}' in completion to to contain '1' tool call but got '{len(choice['tools'])}'.")
        for j, tool_call in enumerate(choice['tools']):
            assert_type_value(obj=tool_call, type_or_value=dict, name=f"tool call '{i + j}' of '{len(choice['tools'])}'")
            assert_keys(obj=tool_call, keys=['id', 'name', 'arguments'], mode="match", name=f"tool call '{i + j}' of '{len(choice['tools'])}' in choice '{i + 1}' of '{len(completion['choices'])}' in completion")
            assert_type_value(obj=tool_call['id'], type_or_value=str, name=f"value of key 'id' in tool call '{i + j}' of '{len(choice['tools'])}' in choice '{i + 1}' of '{len(completion['choices'])}' in completion")
            assert_log(expression=len(tool_call['id']) > 0, message=f"Expected tool call ID in tool call '{i + j}' of '{len(choice['tools'])}' in choice '{i + 1}' of '{len(completion['choices'])}' in completion to be non-empty.")
            assert_type_value(obj=tool_call['name'], type_or_value=list(names_to_args.keys()), name=f"value of key 'name' in tool call '{i + j}' of '{len(choice['tools'])}' in choice '{i + 1}' of '{len(completion['choices'])}' in completion")
            assert_log(expression=set(tool_call['arguments'].keys()) == names_to_args[tool_call['name']], message=f"Expected tool call '{i + j}' of '{len(choice['tools'])}' with name '{tool_call['name']}' in choice '{i + 1}' of '{len(completion['choices'])}' in completion to contain arguments {names_to_args[tool_call['name']]} but got {set(choice['tools'][0]['arguments'].keys())}.")
        assert_type_value(obj=choice['tools'][0]['name'], type_or_value=list(names_to_args.keys())[0], name=f"value of key 'name' in tool call in choice '{i + 1}' of '{len(completion['choices'])}' in completion")
        if 'logs' in choice:
            assert_type_value(obj=choice['logs'], type_or_value=list, name=f"logs in choice '{i + 1}' of '{len(completion['choices'])}' in completion")
            assert_log(expression=len(choice['logs']) > 0, message=f"Expected to find at least one log in in choice '{i + 1}' of '{len(completion['choices'])}' completion.")
            for i, log in enumerate(choice['logs']):
                assert_type_value(obj=log, type_or_value=str, name=f"log '{i}' in choice '{i + 1}' of '{len(completion['choices'])}' in completion logs")
    assert_type_value(obj=completion['logs'], type_or_value=list, name="logs in completion")
    assert_log(expression=len(completion['logs']) > 0, message="Expected to find at least one log in completion.")
    for i, log in enumerate(completion['logs']):
        assert_type_value(obj=log, type_or_value=str, name=f"log '{i}' in completion logs")

    success, message, context = client.get_context()
    assert_type_value(obj=success, type_or_value=bool, name="success")
    assert_type_value(obj=message, type_or_value=str, name="message")
    assert_log(expression=success, message=message)
    assert_type_value(obj=context, type_or_value=list, name="context")
    assert_log(expression=len(context) == 1, message=f"Expected context to contain '1' messages but got '{len(context)}'.")

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

    success, message, completion = client.prompt(text="Tell me how warm it is!", response_type=tool_definitions[-1]['function']['name'], reset_context=True)
    assert_type_value(obj=success, type_or_value=bool, name="success")
    assert_type_value(obj=message, type_or_value=str, name="message")
    assert_log(expression=success, message=message)
    assert_keys(obj=completion, keys=['usage', 'choices', 'logs'], mode="match", name="completion")
    assert_type_value(obj=completion['usage'], type_or_value=dict, name="usage in completion")
    assert_type_value(obj=completion['choices'], type_or_value=list, name="choices in completion")
    assert_log(expression=len(completion['choices']) == kwargs['choices'], message=f"Expected number of choices in completion to match setting 'choices' provided by test definition but got '{len(completion['choices'])}' instead of '{kwargs['choices']}'.")
    for i, choice in enumerate(completion['choices']):
        assert_type_value(obj=choice, type_or_value=dict, name=f"choice '{i + 1}' of '{len(completion['choices'])}' in completion")
        assert_keys(obj=choice, keys=['tools', 'text', 'reasoning', 'logs'], mode="whitelist", name=f"choice '{i + 1}' of '{len(completion['choices'])}' in completion")
        assert_keys(obj=choice, keys=['tools'], mode="required", name=f"choice '{i + 1}' of '{len(completion['choices'])}' in completion")
        if 'text' in choice:
            assert_type_value(obj=choice['text'], type_or_value=str, name=f"text in choice '{i + 1}' of '{len(completion['choices'])}' in completion")
            assert_log(expression=len(choice['text']) > 0, message=f"Expected text in choice '{i + 1}' of '{len(completion['choices'])}' in completion to be a non-empty string.")
        if 'reasoning' in choice:
            assert_type_value(obj=choice['reasoning'], type_or_value=str, name=f"reasoning in choice '{i + 1}' of '{len(completion['choices'])}' in completion")
            assert_log(expression=len(choice['reasoning']) > 0, message=f"Expected reasoning in choice '{i + 1}' of '{len(completion['choices'])}' in completion to be a non-empty string.")
        assert_type_value(obj=choice['tools'], type_or_value=list, name=f"tools in choice '{i + 1}' of '{len(completion['choices'])}' in completion")
        assert_log(expression=len(choice['tools']) == 1, message=f"Expected tools in choice '{i + 1}' of '{len(completion['choices'])}' in completion to to contain '1' tool call but got '{len(choice['tools'])}'.")
        for j, tool_call in enumerate(choice['tools']):
            assert_type_value(obj=tool_call, type_or_value=dict, name=f"tool call '{i + j}' of '{len(choice['tools'])}'")
            assert_keys(obj=tool_call, keys=['id', 'name', 'arguments'], mode="match", name=f"tool call '{i + j}' of '{len(choice['tools'])}' in choice '{i + 1}' of '{len(completion['choices'])}' in completion")
            assert_type_value(obj=tool_call['id'], type_or_value=str, name=f"value of key 'id' in tool call '{i + j}' of '{len(choice['tools'])}' in choice '{i + 1}' of '{len(completion['choices'])}' in completion")
            assert_log(expression=len(tool_call['id']) > 0, message=f"Expected tool call ID in tool call '{i + j}' of '{len(choice['tools'])}' in choice '{i + 1}' of '{len(completion['choices'])}' in completion to be non-empty.")
            assert_type_value(obj=tool_call['name'], type_or_value=list(names_to_args.keys()), name=f"value of key 'name' in tool call '{i + j}' of '{len(choice['tools'])}' in choice '{i + 1}' of '{len(completion['choices'])}' in completion")
            assert_log(expression=set(tool_call['arguments'].keys()) == names_to_args[tool_call['name']], message=f"Expected tool call '{i + j}' of '{len(choice['tools'])}' with name '{tool_call['name']}' in choice '{i + 1}' of '{len(completion['choices'])}' in completion to contain arguments {names_to_args[tool_call['name']]} but got {set(choice['tools'][0]['arguments'].keys())}.")
        assert_type_value(obj=choice['tools'][0]['name'], type_or_value=list(names_to_args.keys())[1], name=f"value of key 'name' in tool call in choice '{i + 1}' of '{len(completion['choices'])}' in completion")
        if 'logs' in choice:
            assert_type_value(obj=choice['logs'], type_or_value=list, name=f"logs in choice '{i + 1}' of '{len(completion['choices'])}' in completion")
            assert_log(expression=len(choice['logs']) > 0, message=f"Expected to find at least one log in in choice '{i + 1}' of '{len(completion['choices'])}' completion.")
            for i, log in enumerate(choice['logs']):
                assert_type_value(obj=log, type_or_value=str, name=f"log '{i}' in choice '{i + 1}' of '{len(completion['choices'])}' in completion logs")
    assert_type_value(obj=completion['logs'], type_or_value=list, name="logs in completion")
    assert_log(expression=len(completion['logs']) > 0, message="Expected to find at least one log in completion.")
    for i, log in enumerate(completion['logs']):
        assert_type_value(obj=log, type_or_value=str, name=f"log '{i}' in completion logs")

    success, message, context = client.get_context()
    assert_type_value(obj=success, type_or_value=bool, name="success")
    assert_type_value(obj=message, type_or_value=str, name="message")
    assert_log(expression=success, message=message)
    assert_type_value(obj=context, type_or_value=list, name="context")
    assert_log(expression=len(context) == 1, message=f"Expected context to contain '1' messages but got '{len(context)}'.")

    success, message = client.set_settings(max_tool_calls=2)
    assert_type_value(obj=success, type_or_value=bool, name="success")
    assert_type_value(obj=message, type_or_value=str, name="message")
    assert_log(expression=success, message=message)

    success, message, completion = client.prompt(text="Tell me the time and how warm it is! Use both tools simultaneously!", response_type="always", reset_context=True)
    assert_type_value(obj=success, type_or_value=bool, name="success")
    assert_type_value(obj=message, type_or_value=str, name="message")
    assert_log(expression=success, message=message)
    assert_keys(obj=completion, keys=['usage', 'choices', 'logs'], mode="match", name="completion")
    assert_type_value(obj=completion['usage'], type_or_value=dict, name="usage in completion")
    assert_type_value(obj=completion['choices'], type_or_value=list, name="choices in completion")
    assert_log(expression=len(completion['choices']) == kwargs['choices'], message=f"Expected number of choices in completion to match setting 'choices' provided by test definition but got '{len(completion['choices'])}' instead of '{kwargs['choices']}'.")
    for i, choice in enumerate(completion['choices']):
        assert_type_value(obj=choice, type_or_value=dict, name=f"choice '{i + 1}' of '{len(completion['choices'])}' in completion")
        assert_keys(obj=choice, keys=['tools', 'text', 'reasoning', 'logs'], mode="whitelist", name=f"choice '{i + 1}' of '{len(completion['choices'])}' in completion")
        assert_keys(obj=choice, keys=['tools'], mode="required", name=f"choice '{i + 1}' of '{len(completion['choices'])}' in completion")
        if 'text' in choice:
            assert_type_value(obj=choice['text'], type_or_value=str, name=f"text in choice '{i + 1}' of '{len(completion['choices'])}' in completion")
            assert_log(expression=len(choice['text']) > 0, message=f"Expected text in choice '{i + 1}' of '{len(completion['choices'])}' in completion to be a non-empty string.")
        if 'reasoning' in choice:
            assert_type_value(obj=choice['reasoning'], type_or_value=str, name=f"reasoning in choice '{i + 1}' of '{len(completion['choices'])}' in completion")
            assert_log(expression=len(choice['reasoning']) > 0, message=f"Expected reasoning in choice '{i + 1}' of '{len(completion['choices'])}' in completion to be a non-empty string.")
        assert_type_value(obj=choice['tools'], type_or_value=list, name=f"tools in choice '{i + 1}' of '{len(completion['choices'])}' in completion")
        assert_log(expression=len(choice['tools']) == 2, message=f"Expected tools in choice '{i + 1}' of '{len(completion['choices'])}' in completion to to contain '2' tool call but got '{len(choice['tools'])}'.")
        names = []
        for j, tool_call in enumerate(choice['tools']):
            assert_type_value(obj=tool_call, type_or_value=dict, name=f"tool call '{i + j}' of '{len(choice['tools'])}'")
            assert_keys(obj=tool_call, keys=['id', 'name', 'arguments'], mode="match", name=f"tool call '{i + j}' of '{len(choice['tools'])}' in choice '{i + 1}' of '{len(completion['choices'])}' in completion")
            assert_type_value(obj=tool_call['id'], type_or_value=str, name=f"value of key 'id' in tool call '{i + j}' of '{len(choice['tools'])}' in choice '{i + 1}' of '{len(completion['choices'])}' in completion")
            assert_log(expression=len(tool_call['id']) > 0, message=f"Expected tool call ID in tool call '{i + j}' of '{len(choice['tools'])}' in choice '{i + 1}' of '{len(completion['choices'])}' in completion to be non-empty.")
            assert_type_value(obj=tool_call['name'], type_or_value=list(names_to_args.keys()), name=f"value of key 'name' in tool call '{i + j}' of '{len(choice['tools'])}' in choice '{i + 1}' of '{len(completion['choices'])}' in completion")
            assert_log(expression=set(tool_call['arguments'].keys()) == names_to_args[tool_call['name']], message=f"Expected tool call '{i + j}' of '{len(choice['tools'])}' with name '{tool_call['name']}' in choice '{i + 1}' of '{len(completion['choices'])}' in completion to contain arguments {names_to_args[tool_call['name']]} but got {set(choice['tools'][0]['arguments'].keys())}.")
            names.append(tool_call['name'])
        assert_log(expression=set(names) == set(names_to_args.keys()), message=f"Expected tool calls in choice '{i + 1}' of '{len(completion['choices'])}' in completion to cover the names {set(names_to_args.keys())} but got {names}.")
        if 'logs' in choice:
            assert_type_value(obj=choice['logs'], type_or_value=list, name=f"logs in choice '{i + 1}' of '{len(completion['choices'])}' in completion")
            assert_log(expression=len(choice['logs']) > 0, message=f"Expected to find at least one log in in choice '{i + 1}' of '{len(completion['choices'])}' completion.")
            for i, log in enumerate(choice['logs']):
                assert_type_value(obj=log, type_or_value=str, name=f"log '{i}' in choice '{i + 1}' of '{len(completion['choices'])}' in completion logs")
    assert_type_value(obj=completion['logs'], type_or_value=list, name="logs in completion")
    assert_log(expression=len(completion['logs']) > 0, message="Expected to find at least one log in completion.")
    for i, log in enumerate(completion['logs']):
        assert_type_value(obj=log, type_or_value=str, name=f"log '{i}' in completion logs")

    success, message, context = client.get_context()
    assert_type_value(obj=success, type_or_value=bool, name="success")
    assert_type_value(obj=message, type_or_value=str, name="message")
    assert_log(expression=success, message=message)
    assert_type_value(obj=context, type_or_value=list, name="context")
    assert_log(expression=len(context) == 1, message=f"Expected context to contain '1' messages but got '{len(context)}'.")

def test_093_openrouter_tool_use_choices():
    return "NOT SUPPORTED."

def test_094_openrouter_tool_use_choices_stream():
    return "NOT SUPPORTED."

def test_095_openai_tool_use_choices():
    return _tool_use_choices(stream=False, endpoint="OpenAI", model="gpt-5.2-2025-12-11", choices=2, reasoning_effort="none")

def test_096_openai_tool_use_choices_stream():
    return "NOT SUPPORTED."

def test_097_mistral_tool_use_choices():
    return _tool_use_choices(stream=False, endpoint="Mistral", model="mistral-small-latest", choices=2, reasoning_effort="none")

def test_098_mistral_tool_use_choices_stream():
    return _tool_use_choices(stream=True, endpoint="Mistral", model="mistral-small-latest", choices=2, reasoning_effort="none")

def test_099_vllm_tool_use_choices():
    success, message, models = ChatCompletions(endpoint="AIS").get_models()
    assert_log(expression=success, message=message)
    assert_type_value(obj=models, type_or_value=list, name="models")
    if len(models) == 0:
        return "AIS endpoint is not hosting any models."
    model = models[-1]
    return _tool_use_choices(endpoint="AIS", model=model, choices=2, reasoning_effort="none", timeout_read=60, timeout_completion=60)

def test_100_vllm_tool_use_choices_stream():
    success, message, models = ChatCompletions(endpoint="AIS").get_models()
    assert_log(expression=success, message=message)
    assert_type_value(obj=models, type_or_value=list, name="models")
    if len(models) == 0:
        return "AIS endpoint is not hosting any models."
    model = models[-1]
    return _tool_use_choices(endpoint="AIS", model=model, choices=2, reasoning_effort="none", timeout_read=60, timeout_completion=60)
