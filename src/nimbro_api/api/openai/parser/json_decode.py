# This completion parser provides a pseudo-JSON mode that does not affect model inference,
# yet still guarantees that a successful response is valid JSON, by decoding JSON from a given text-completion.
# It is robust against artifacts outside of the JSON-body, like Markdown tags or summaries.

from nimbro_api.utility.misc import format_obj
from nimbro_api.utility.string import extract_json

def parse(self, success, message, completion):
    if 'text' in completion:
        if isinstance(completion['text'], str):
            json_obj = extract_json(completion['text'], first_over_longest=False)
            if json_obj:
                completion['text'] = json_obj
                completion['logs'].append(f"Successfully decoded text-completion {format_obj(json_obj)} as JSON.")
                self._logger.info(f"Successfully decoded text-completion as JSON: {format_obj(json_obj)}.")
            else:
                success = False
                message = "Failed to decode text-completion as JSON."
                completion['logs'].append(message)
        else:
            success = False
            message = f"Cannot decode JSON from text-completion of type '{type(completion['text']).__name__}' instead of 'str'."
            completion['logs'].append(message)
    else:
        success = False
        message = "Cannot decode JSON without a text-completion."
        completion['logs'].append(message)

    return success, message, completion
