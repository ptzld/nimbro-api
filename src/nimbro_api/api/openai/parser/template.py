# This file serves as a template and tutorial for writing an import-based completion parser for the ChatCompletions Client.
# Once a ChatCompletions Client generates a valid response 3-tuple (success[bool], message[str], completion[dict]),
# initiated via its `prompt()` function, it can be modified by a completion parser as defined by the 'parser'
# setting of the ChatCompletions Client. Note, that this setting is a list that chains parsers sequentially as long as `success` remains `True`.
# When setting `success` to `False`, the response remains eligible for retry according to the 'retry' setting.
# To prevent retry, raise an `UnreciverableError` instead (`from nimbro_api.utility.misc import UnrecoverableError`).
# The ChatCompletions Client using this parser will wait indefinitely for `parse()` to terminate.

def parse(self, success, message, completion):
    # Reference for response structure:
    assert isinstance(success, bool)
    assert success is True
    assert isinstance(message, str)
    assert completion is None or isinstance(completion, dict)
    if isinstance(completion, dict):
        assert 'logs' in completion
        assert isinstance(completion['logs'], list)
        if 'reasoning' in completion:
            assert isinstance(completion['reasoning'], str)
        if 'text' in completion:
            assert isinstance(completion['text'], (str, dict))
        if 'tool_calls' in completion:
            assert isinstance(completion['tool_calls'], list)
            assert len(completion['tool_calls']) > 0
            for tool_call in completion['tool_calls']:
                assert isinstance(tool_call, dict)
                assert set(tool_call.keys()) == {"id", "name", "arguments"}
                assert isinstance(tool_call['id'], str)
                assert len(tool_call['id']) > 0
                assert isinstance(tool_call['name'], str)
                assert len(tool_call['name']) > 0
                assert isinstance(tool_call['arguments'], dict)
        if 'usage' in completion:
            assert isinstance(completion['usage'], dict)

    # Modify the response without violating any of the following restrictions:
    # - Value of `success` must be of type `bool`.
    # - Value of `message` must be of type `str`.
    # - Value of `completion` must be of type `dict` or `None`.
    # - The response must be serializable.
    # - This script must terminate with code 0.
    # - Any violation against these rules results in a failed chat completion without allowing for further retries.

    # Use any attribute of the ChatCompletions ClientBase, e.g. its Logger.
    self._logger.info("This completion parser does nothing except logging this message from inside the ChatCompletions Client.")

    # Return the (modified) response.
    return success, message, completion
