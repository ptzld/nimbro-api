# Language models tend to generate strings that sometimes contain leading/trailing whitespace.
# This completions parser strips such whitespace from all strings in the response,
# including text, reasoning, JSON objects, tool calls, and logs.

def recursive_strip(obj):
    if isinstance(obj, str):
        return obj.strip()
    if isinstance(obj, list):
        return [recursive_strip(item) for item in obj]
    if isinstance(obj, dict):
        return {key: recursive_strip(value) for key, value in obj.items()}
    return obj

def parse(self, success, message, completion):
    message = message.strip()
    if isinstance(completion, dict):
        completion = recursive_strip(completion)
        completion['logs'].append("Stripped all strings in completion.")
        self._logger.debug(completion['logs'][-1])
    return success, message, completion
