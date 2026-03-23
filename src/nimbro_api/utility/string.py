import re
import json

from .misc import assert_type_value

url_pattern = re.compile(r'^(https?):\/\/\S+$')
# url_pattern = r'^(https?):\/\/[^\s\/$.?#].[^\s]*$'
# url_pattern = r'^(http|https):\/\/([\w.-]+)(\.[\w.-]+)+([\/\w\.-]*)*\/?$'
base64_pattern = re.compile(r'^[A-Za-z0-9+/]*$')

def is_url(string):
    """
    Check if a string is a valid URL.

    Args:
        string (str): The input string to process.

    Raises:
        UnrecoverableError: If input arguments are invalid.

    Returns:
        bool: True, if 'string' is a valid URL.
    """
    # parse arguments
    assert_type_value(obj=string, type_or_value=str, name="argument 'string'")

    # identify URL
    valid = bool(url_pattern.fullmatch(string))

    return valid

def is_base64(string):
    """
    Check if a string is valid Base64.

    Args:
        string (str): The input string to process.

    Raises:
        UnrecoverableError: If input arguments are invalid.

    Returns:
        bool: True, if 'string' is valid Base64.

    Notes:
        - Only standard Base64 characters (A–Z, a–z, 0–9, +, /) and optional padding '=' (up to 2 characters) are considered valid.
        - URL-safe Base64 is not allowed.
    """
    # parse arguments
    assert_type_value(obj=string, type_or_value=str, name="argument 'string'")

    # identify Base64
    length = len(string)
    if length > 0:
        mod = length % 4
        if string.strip() != string:
            return False
        if mod == 1:
            return False
        if string.endswith(('=', '==', '===')):
            if string.endswith('==='):
                return False
            stripped = string.rstrip('=')
            stripped_len = len(stripped)
            padding_count = length - stripped_len
            if padding_count > 2:
                return False
            stripped_mod = stripped_len % 4
            if padding_count == 2 and stripped_mod != 2:
                return False
            if padding_count == 1 and stripped_mod != 3:
                return False
        else:
            if mod != 0:
                return False
            stripped = string
        if '=' in stripped or not base64_pattern.fullmatch(stripped):
            return False

    return True

def extract_json(string, *, first_over_longest=False):
    """
    Extract the first or longest valid JSON object from a string.

    Args:
        string (str):
            The input string to process.
        first_over_longest (bool, optional):
            If True, the first encountered JSON object is returned instead of the longest. Defaults to False.

    Raises:
        UnrecoverableError: If input arguments are invalid.

    Returns:
        any | None: The extracted JSON object, or None if no valid JSON object is found.
    """
    # parse arguments
    assert_type_value(obj=string, type_or_value=str, name="argument 'string'")
    assert_type_value(obj=first_over_longest, type_or_value=bool, name="argument 'first_over_longest'")

    # find JSON
    opening_indices = []
    opening_indices.extend([m.start() for m in re.finditer(r'[{\[]', string)])
    if len(opening_indices) > 0:
        closing_indices = []
        closing_indices.extend([m.start() for m in re.finditer(r'[}\]]', string)])
        if len(closing_indices) > 0:
            options = []
            for i in opening_indices:
                for j in closing_indices:
                    if j < i:
                        continue
                    else:
                        options.append(string[i:j + 1])
            if not first_over_longest:
                options.sort(key=len, reverse=True)
            for option in options:
                try:
                    return json.loads(option)
                except json.JSONDecodeError:
                    pass
