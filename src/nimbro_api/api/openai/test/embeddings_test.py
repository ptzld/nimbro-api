import os
import uuid

import nimbro_api
from nimbro_api.utility.misc import assert_type_value, assert_log
from ..client.embeddings import Embeddings

def test_1_utilities():
    client = Embeddings()

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
    client = Embeddings(endpoint="OpenAI") # Models API from OpenRouter does not include embedding models

    success, message, models = client.get_models()
    assert_type_value(obj=success, type_or_value=bool, name="success")
    assert_type_value(obj=message, type_or_value=str, name="message")
    assert_log(expression=success, message=message)
    assert_type_value(obj=models, type_or_value=list, name="result of get_models()")

def test_3_inference():
    client = Embeddings(settings={'cache_read': False})

    # generate
    texts = ["mouse", "cat", "helicopter"] + [uuid.uuid4().hex for _ in range(3)]
    success, message, embeddings_inf = client.get_embedding(text=texts)
    assert_type_value(obj=success, type_or_value=bool, name="success")
    assert_type_value(obj=message, type_or_value=str, name="message")
    assert_log(expression=success, message=message)
    assert_type_value(obj=embeddings_inf, type_or_value=list, name="result of get_embedding()")
    assert_log(expression=len(embeddings_inf) == len(texts), message=f"Expected number of generated embeddings '{len(embeddings_inf)}' to match the number of input texts '{len(texts)}'.")

    # read cache

    success, message = nimbro_api.execute_deferred_jobs()
    assert_type_value(obj=success, type_or_value=bool, name="success")
    assert_type_value(obj=message, type_or_value=str, name="message")
    assert_log(expression=success, message=message)

    success, message = client.set_settings(settings={'cache_read': True})
    assert_type_value(obj=success, type_or_value=bool, name="success")
    assert_type_value(obj=message, type_or_value=str, name="message")
    assert_log(expression=success, message=message)

    success, message, embeddings_cache = client.get_embedding(text=texts)
    assert_type_value(obj=success, type_or_value=bool, name="success")
    assert_type_value(obj=message, type_or_value=str, name="message")
    assert_log(expression=success, message=message)
    assert_type_value(obj=embeddings_cache, type_or_value=list, name="result of get_embedding()")
    assert_log(expression=len(embeddings_cache) == len(texts), message=f"Expected number of generated embeddings '{len(embeddings_cache)}' to match the number of input texts '{len(texts)}'.")

    assert_log(expression=len(embeddings_inf) == len(embeddings_cache), message=f"Expected inferred and cached results to contain '{len(texts)}' elements but got '{len(embeddings_inf)}' and '{len(embeddings_cache)}'.")
    for i in range(len(texts)):
        assert_log(expression=len(embeddings_inf[i]) == len(embeddings_cache[i]), message=f"Expected inferred and cached embedding '{i + 1}' of '{len(texts)}' to match in lengths but got '{len(embeddings_inf[i])}' and '{len(embeddings_cache[i])}'.")
        if embeddings_inf[i] != embeddings_cache[i]:
            for j in range(len(embeddings_inf[i])):
                assert_log(expression=embeddings_inf[i][j] == embeddings_cache[i][j], message=f"Expected inferred and cached embedding '{i + 1}' of '{len(texts)}' to match at element '{j + 1}' of '{len(embeddings_inf[i])}' got '{embeddings_inf[i][j]}' and '{embeddings_cache[i][j]}'.")

    # both
    texts = texts[:3] + [uuid.uuid4().hex for _ in range(3)]
    success, message, embeddings_mix = client.get_embedding(text=texts)
    assert_type_value(obj=success, type_or_value=bool, name="success")
    assert_type_value(obj=message, type_or_value=str, name="message")
    assert_log(expression=success, message=message)
    assert_type_value(obj=embeddings_mix, type_or_value=list, name="result of get_embedding()")
    assert_log(expression=len(embeddings_mix) == len(texts), message=f"Expected number of generated embeddings '{len(embeddings_mix)}' to match the number of input texts '{len(texts)}'.")

    # single text

    success, message, embedding_cache = client.get_embedding(text=[texts[0]])
    assert_type_value(obj=success, type_or_value=bool, name="success")
    assert_type_value(obj=message, type_or_value=str, name="message")
    assert_log(expression=success, message=message)
    assert_type_value(obj=embedding_cache, type_or_value=list, name="result of get_embedding()")
    assert_log(expression=len(embedding_cache) == 1, message=f"Expected number of generated embeddings '{len(embedding_cache)}' to match the number of input texts '{len(texts)}'.")

    success, message, embedding_cache = client.get_embedding(text=texts[0])
    assert_type_value(obj=success, type_or_value=bool, name="success")
    assert_type_value(obj=message, type_or_value=str, name="message")
    assert_log(expression=success, message=message)
    assert_type_value(obj=embedding_cache, type_or_value=list, name="result of get_embedding()")
    assert_log(expression=len(embedding_cache) > 0, message=f"Expected generated embedding to contain at least one value but got '{len(embedding_cache)}'.")
    assert_type_value(obj=embedding_cache[0], type_or_value=[int, float], name="element in result of get_embedding()")

    success, message = client.set_settings(settings={'cache_read': False})
    assert_type_value(obj=success, type_or_value=bool, name="success")
    assert_type_value(obj=message, type_or_value=str, name="message")
    assert_log(expression=success, message=message)

    success, message, embedding_cache = client.get_embedding(text=[texts[0]])
    assert_type_value(obj=success, type_or_value=bool, name="success")
    assert_type_value(obj=message, type_or_value=str, name="message")
    assert_log(expression=success, message=message)
    assert_type_value(obj=embedding_cache, type_or_value=list, name="result of get_embedding()")
    assert_log(expression=len(embedding_cache) == 1, message=f"Expected number of generated embeddings '{len(embedding_cache)}' to match the number of input texts '{len(texts)}'.")

    success, message, embedding_cache = client.get_embedding(text=texts[0])
    assert_type_value(obj=success, type_or_value=bool, name="success")
    assert_type_value(obj=message, type_or_value=str, name="message")
    assert_log(expression=success, message=message)
    assert_type_value(obj=embedding_cache, type_or_value=list, name="result of get_embedding()")
    assert_log(expression=len(embedding_cache) > 0, message=f"Expected generated embedding to contain at least one value but got '{len(embedding_cache)}'.")
    assert_type_value(obj=embedding_cache[0], type_or_value=[int, float], name="element in result of get_embedding()")

    success, message = nimbro_api.execute_deferred_jobs()
    assert_type_value(obj=success, type_or_value=bool, name="success")
    assert_type_value(obj=message, type_or_value=str, name="message")
    assert_log(expression=success, message=message)
