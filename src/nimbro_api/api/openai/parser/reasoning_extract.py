# Some reasoning models exhibit reasoning within the text-completion, rather than in a separate reasoning channel.
# Typically, the reasoning content is encapsulated by special tags, such as: "<think> I should focus on... </think>"
# When parsing model outputs directly, this can be problematic, e.g., when requesting JSON or any specific output format.
# This completion parser extracts existing reasoning content from a text-completion and moves it to the regular reasoning
# channel (`completion['reasoning']`). Thereby, making such models behave like regular reasoning models.

import re
from nimbro_api.utility.misc import format_obj

tags = [
    ("<think>", "</think>"),
    ("<thinking>", "</thinking>"),
    ("<reason>", "</reason>"),
    ("<reasoning>", "</reasoning>"),
    ("<explain>", "</explain>"),
    ("<thought>", "</thought>"),
    ("<cot>", "</cot>"),
    ("<|begin_of_thought|>", "<|end_of_thought|>"),
]

def extract_reasoning(text):
    tag_patterns = [f"{re.escape(start)}(.*?){re.escape(end)}" for start, end in tags]
    combined_pattern = "|".join(tag_patterns)

    reasoning_parts = []

    matches = list(re.finditer(combined_pattern, text, re.DOTALL))
    for m in matches:
        for group in m.groups():
            if group is not None:
                reasoning_parts.append(group.strip())
                break

    if len(reasoning_parts) == 0:
        return None, text

    outside_list = list(text)
    for m in reversed(matches):
        start, end = m.start(), m.end()
        outside_list[start:end] = [" "]
    outside_content = "".join(outside_list).strip()

    reasoning_content = " ".join(reasoning_parts).strip()
    return reasoning_content, outside_content

def parse(self, success, message, completion):
    if success and isinstance(completion, dict):
        if 'text' in completion:
            if isinstance(completion['text'], str):
                reasoning_content, outside = extract_reasoning(completion['text'])
                if reasoning_content is None:
                    self._logger.debug("There is no reasoning content inside the text-completion.")
                else:
                    if len(reasoning_content) > 0:
                        if 'reasoning' in completion:
                            completion['reasoning'] = f"{completion['reasoning'].rstrip()} {reasoning_content}"
                            completion['logs'].append(f"Extracted reasoning content '{format_obj(reasoning_content)}' from text-completion and appended it to existing reasoning content.")
                        else:
                            completion['reasoning'] = reasoning_content
                            completion['logs'].append(f"Extracted reasoning content '{format_obj(reasoning_content)}' from text-completion.")
                    else:
                        completion['logs'].append("Removed reasoning tags without any reasoning content from text-completion.")

                    self._logger.info(completion['logs'][-1])

                    if len(outside) > 0:
                        completion['text'] = outside
                    else:
                        del completion['text']
                        completion['logs'].append("Removed text-completion after being emptied by reasoning extraction.")
            else:
                self._logger.debug(f"Cannot extract reasoning content from text-completion of type '{type(completion['text']).__name__}' instead of 'str'.")
        else:
            self._logger.debug("Cannot extract reasoning content without a text-completion.")

    return success, message, completion
