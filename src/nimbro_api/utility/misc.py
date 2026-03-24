import json
import copy
import shutil

# Assertions

class UnrecoverableError(Exception):
    """
    Custom exception raised when an operation cannot proceed due to an invalid state or arguments.

    Notes:
        - This exception is primarily intended to be raised inside of 'nimbro_utils.utility.api.base_wrapper()',  which wraps most user-exposed functions.
        - When raised within the wrapper, it signals that the wrapped function must not be considered
          for retry and should instead return a failure response containing the raised message immediately.
        - While this package is generally designed to avoid raising exceptions to the user, this is
          the specific exception that may be raised in rare cases, such as during object initialization.
    """

def assert_type_value(obj, type_or_value, *, match_types=True, match_inherited_types=True, match_types_as_values=False, name="object", prefix="", logger=None):
    """
    Validates that an object is of a specified type or equals a specified value.

    Args:
        obj (object):
            The object to validate.
        type_or_value (type | value | list[type | value] | dict_keys[type | value]):
            A type, value, or list of types/values that 'obj' is checked against.
        match_types (bool, optional):
            If `True`, object types are matched against provided types. Defaults to `True`.
        match_inherited_types (bool, optional):
            If `True`, uses `isinstance()` for type matching; if `False`, requires exact type equality. Ignored if 'match_types' is `False`. Defaults to `True`.
        match_types_as_values (bool, optional):
            If `True`, types in 'type_or_value' are also accepted as values, even when type matching is disabled. Defaults to `True`.
        name (str, optional):
            Name of the object, used in error messages. Defaults to "object".
        prefix (str, optional):
            Prefix string prepended to the log message. Defaults to "".
        logger (nimbro_api.utility.logger.Logger | None, optional):
            If provided, logs default assertion message before raising assertion. Defaults to `None`.

    Raises:
        UnrecoverableError: If input arguments are invalid or 'obj' does not match any of the specified types or values.

    Notes:
        - `None` in 'type_or_value' is always treated as `type(None)` and bypasses 'match_types'.
        - Boolean values are never matched as integers (i.e., `True`/`False` will not satisfy an `int` type).
        - Types in 'type_or_value' are counted as values only when 'match_types_as_values' is `True`, regardless of 'match_types'.
        - Long string representations of 'obj' are truncated in messages for readability.
        - If 'logger' is provided and validation fails, the default error message is logged (prefixed with 'prefix').
    """
    # parse arguments

    if not isinstance(obj, object):
        raise UnrecoverableError(f"Expected type of argument 'obj' to be 'object', but it is '{type(obj).__name__}'.")
    if not isinstance(match_types, bool):
        raise UnrecoverableError(f"Expected type of argument 'match_types' to be 'bool', but it is '{type(match_types).__name__}'.")
    if not isinstance(match_inherited_types, bool):
        raise UnrecoverableError(f"Expected type of argument 'match_inherited_types' to be 'bool', but it is '{type(match_inherited_types).__name__}'.")
    if not isinstance(match_types_as_values, bool):
        raise UnrecoverableError(f"Expected type of argument 'match_types_as_values' to be 'bool', but it is '{type(match_types_as_values).__name__}'.")
    if not isinstance(name, str):
        raise UnrecoverableError(f"Expected type of argument 'name' to be 'str', but it is '{type(name).__name__}'.")
    if not isinstance(prefix, str):
        raise UnrecoverableError(f"Expected type of argument 'prefix' to be 'str', but it is '{type(prefix).__name__}'.")
    from nimbro_api.utility.logger import Logger
    if not (logger is None or isinstance(logger, Logger)):
        raise UnrecoverableError(f"Expected type of argument 'logger' to be in {[type(None).__name__, Logger]}, but it is '{type(logger).__name__}'.")

    max_value_length = 50
    if isinstance(obj, type):
        value = obj.__name__
    elif len(str(obj)) > max_value_length:
        value = str(obj)[:max_value_length] + "..."
    else:
        value = str(obj)

    if isinstance(type_or_value, type({}.keys())):
        type_or_value = list(type_or_value)
    elif not isinstance(type_or_value, (tuple, list)):
        type_or_value = [type_or_value]
    if len(type_or_value) == 0:
        raise UnrecoverableError("Expected argument 'type_or_value' to be a value, a type, or a list thereof, but it is an empty list.")

    # collect matching targets
    types_list, type_names, values_list, value_names, value_types = [], [], [], [], []
    for item in type_or_value:
        if item is None or isinstance(item, type(None)):
            type_names.append("NoneType")
            values_list.append(None)
        elif isinstance(item, type):
            if not (match_types or match_types_as_values):
                raise UnrecoverableError("Expected either 'match_types' or 'match_types_as_values' to be True when 'type_or_value' contains an element of type 'type'.")
            if match_types:
                types_list.append(item)
                type_names.append(item.__name__)
            if match_types_as_values:
                values_list.append(item)
                value_names.append(item.__name__)
                value_types.append("type")
        else:
            values_list.append(item)
            value_names.append(f"{item}")
            value_types.append(type(item).__name__)

    # check validity and generate error text

    def _is_valid_type(o):
        for t in types_list:
            if t is int and isinstance(o, bool):
                continue
            if match_inherited_types:
                if isinstance(o, t):
                    return True
            else:
                if type(o) is t:
                    return True
        return False

    def _typed_in(value, collection):
        for item in collection:
            if type(value) is type(item):
                try:
                    if value == item:
                        return True
                except Exception:
                    pass
        return False

    valid_type = match_types and _is_valid_type(obj)
    valid_value = _typed_in(obj, values_list)
    valid = valid_type or valid_value

    if len(type_names) > 0 and len(value_names) > 0:
        if not valid:
            if len(type_names) == 1:
                if len(value_names) == 1:
                    _text = f"Expected type of {name} to be '{type_names[0]}', or its value to be '{value_names[0]}' of type '{value_types[0]}', but it is '{value}' of type '{type(obj).__name__}'."
                else:
                    _text = f"Expected type of {name} to be '{type_names[0]}', or its value to be in {value_names} with types {value_types}, but it is '{value}' of type '{type(obj).__name__}'."
            else:
                if len(value_names) == 1:
                    _text = f"Expected type of {name} to be in {type_names}, or its value to be '{value_names[0]}' of type '{value_types[0]}', but it is '{value}' of type '{type(obj).__name__}'."
                else:
                    _text = f"Expected type of {name} to be in {type_names}, or its value to be in {value_names} with types {value_types}, but it is '{value}' of type '{type(obj).__name__}'."
    elif len(type_names) > 0:
        if not valid:
            if len(type_names) == 1:
                _text = f"Expected type of {name} to be '{type_names[0]}', but it is '{type(obj).__name__}'."
            else:
                _text = f"Expected type of {name} to be in {type_names}, but it is '{type(obj).__name__}'."
    else:
        if not valid:
            if len(value_names) == 1:
                _text = f"Expected value of {name} to be '{value_names[0]}' of type '{value_types[0]}', but it is '{value}' of type '{type(obj).__name__}'."
            else:
                _text = f"Expected value of {name} to be in {value_names} with types {value_types}, but it is '{value}' of type '{type(obj).__name__}'."

    # log and raise
    if not valid:
        if logger is not None:
            logger.fatal(f"{prefix}{_text}")
        raise UnrecoverableError(_text)

def assert_keys(obj, keys, mode="whitelist", *, name="dictionary", text=None, logger=None):
    """
    Validates that a dictionary contains or omits specific keys, depending on the selected mode.

    Args:
        obj (dict):
            The dictionary to validate.
        keys (set | list | tuple | dict_keys):
            The collection of keys to check against.
        mode (str, optional):
            The validation mode. Must be one of:
            - "match": All and only the specified keys must be present in 'obj'.
            - "whitelist": Only the specified keys are permitted to be present in 'obj'.
            - "required": The specified keys must be present, but extra keys are allowed.
            - "blacklist": The specified keys must not be present in 'obj'.
            Defaults to "whitelist".
        name (str, optional):
            Name of the object, used in error messages. Defaults to "dictionary".
        text (str | None, optional):
            Custom assertion message to override the default one. Defaults to `None`.
        logger (nimbro_api.utility.logger.Logger | None, optional):
            If provided, logs default assertion message and 'text' (if set) before raising assertion. Defaults to `None`.

    Raises:
        UnrecoverableError: If input arguments are invalid or the 'obj' violates the key constraints defined by 'mode'.

    Notes:
        - In "match" mode, 'obj' must contain exactly the keys in 'keys'.
        - In "whitelist" mode, 'obj' must not contain any keys that are not in 'keys', but does not require any of them to be present.
        - In "required" mode, 'obj' must contain at least the keys in 'keys', but extra keys are allowed.
        - In "blacklist" mode, 'obj' must not contain any of the keys in 'keys'.
        - If 'logger' is provided and validation fails, the error message is logged using 'logger.error()'.
    """
    # parse arguments
    from nimbro_api.utility.logger import Logger
    assert_type_value(obj=logger, type_or_value=[Logger, None], name="argument 'logger'")
    assert_type_value(obj=obj, type_or_value=dict, name="argument 'obj'")
    assert_type_value(obj=keys, type_or_value=[set, list, tuple, type({}.keys())], name="argument 'keys'")
    assert_type_value(obj=mode, type_or_value=["match", "whitelist", "blacklist", "required"], name="argument 'mode'")
    assert_type_value(obj=name, type_or_value=str, name="argument 'name'")
    assert_type_value(obj=text, type_or_value=[str, None], name="argument 'text'")

    qm = "'"

    # missing keys
    if mode in ["match", "required"]:
        missing_keys = []
        for key in keys:
            if key not in obj:
                missing_keys.append(key)
        if len(missing_keys) > 0:
            if len(missing_keys) == len(keys):
                _text = f"Expected {name} to contain the key{(' ' + qm + str(list(keys)[0]) + qm) if len(keys) == 1 else ('s ' + str(list(keys)))} but it misses {'all of them' if len(missing_keys) > 1 else 'it'}."
            else:
                _text = f"Expected {name} to contain the key{(' ' + qm + str(list(keys)[0]) + qm) if len(keys) == 1 else ('s ' + str(list(keys)))} but it misses {str(missing_keys) if len(missing_keys) > 1 else (qm + str(missing_keys[0]) + qm)}."
            if logger is not None:
                logger.error(_text)
                if text is not None:
                    logger.error(text)
            if len(missing_keys) != 0:
                raise UnrecoverableError(_text if text is None else text)

    # excessive keys
    if mode in ["match", "whitelist"]:
        excessive_keys = []
        for key in obj:
            if key not in keys:
                excessive_keys.append(key)
        if len(excessive_keys) > 0:
            if len(excessive_keys) == len(keys):
                _text = f"Expected {name} to contain the key{(' ' + qm + str(list(keys)[0]) + qm) if len(keys) == 1 else ('s ' + str(list(keys)))} but it misses {'all of them' if len(excessive_keys) > 1 else 'it'}."
            else:
                _text = f"Expected {name} to contain only the key{(' ' + qm + str(list(keys)[0]) + qm) if len(keys) == 1 else ('s ' + str(list(keys)))} but it contains {str(excessive_keys) if len(excessive_keys) > 1 else (qm + str(excessive_keys[0]) + qm)}."
            if logger is not None:
                logger.error(_text)
                if text is not None:
                    logger.error(text)
            if len(excessive_keys) != 0:
                raise UnrecoverableError(_text if text is None else text)

    # forbidden keys
    elif mode == "blacklist":
        forbidden_keys = []
        for key in obj:
            if key in keys:
                forbidden_keys.append(key)
        if len(forbidden_keys) > 0:
            if len(forbidden_keys) == len(keys):
                _text = f"Expected {name} to contain the key{(' ' + qm + str(list(keys)[0]) + qm) if len(keys) == 1 else ('s ' + str(list(keys)))} but it misses {'all of them' if len(forbidden_keys) > 1 else 'it'}."
            else:
                _text = f"Expected {name} to not contain the key{(' ' + qm + str(list(keys)[0]) + qm) if len(keys) == 1 else ('s ' + str(list(keys)))} but it {('contains' + str(forbidden_keys)) if len(keys) > 1 else 'does'}."
            if logger is not None:
                logger.error(_text)
                if text is not None:
                    logger.error(text)
            if len(forbidden_keys) != 0:
                raise UnrecoverableError(_text if text is None else text)

def assert_log(expression, message, *, logger=None):
    """
    Assertion wrapper emitting the assertion message as a log before raising the assertion.

    Args:
        expression (any):
            The expression to be evaluated as `bool`, where `True` passes and `False` raises.
        message (str):
            The message emitted as log and assertion message.
        logger (nimbro_api.utility.logger.Logger | None, optional):
            If provided, logs the 'message' before raising the assertion. Defaults to `None`.

    Raises:
        UnrecoverableError: If input arguments are invalid or 'expression' evaluates to `False`.
    """
    # parse arguments
    from nimbro_api.utility.logger import Logger
    assert_type_value(obj=logger, type_or_value=[Logger, None], name="argument 'logger'")
    assert_type_value(obj=message, type_or_value=str, name="argument 'message'")

    # evaluate expression
    if not expression:
        if logger is not None:
            logger.error(message)
        raise UnrecoverableError(message)

# Object Management

def update_dict(old_dict, new_dict=None, *, key_name=None, logger=None, _prefix=""):
    """
    Update a dictionary by merging keys from an old dictionary into a new dictionary, filling in missing keys recursively.

    Args:
        old_dict (dict):
            The original dictionary containing default key-value pairs.
        new_dict (dict | None):
            The dictionary to update, or `None` to create a new one. Defaults to `None`.
        key_name (str | None, optional):
            Name to use for keys in log messages. Defaults to "key" if `None`.
        logger (nimbro_api.utility.logger.Logger | None, optional):
            If provided, logs default assertion message before raising assertion. Defaults to `None`.

    Raises:
        UnrecoverableError: If input arguments are invalid.

    Returns:
        dict: A new dictionary containing all keys from 'new_dict' (or an empty `dict` if `None`), updated with keys from 'old_dict' where missing.

    Notes:
        - If 'new_dict' is `None`, an empty `dict` is created.
        - If both old and new values for a key are dicts, they are merged recursively rather than overwritten.
        - Logging always occurs at the leaf level (deepest non-dict-vs-dict values), producing precise per-field messages.
        - If 'logger' is provided, changes are logged at `info` level and unchanged values at `debug` level.
    """
    # parse arguments
    if not _prefix:
        from nimbro_api.utility.logger import Logger
        assert_type_value(obj=logger, type_or_value=[Logger, None], name="argument 'logger'")
        assert_type_value(obj=old_dict, type_or_value=dict, name="argument 'old_dict'")
        assert_type_value(obj=new_dict, type_or_value=[dict, None], name="argument 'new_dict'")
        assert_type_value(obj=key_name, type_or_value=[str, None], name="argument 'key_name'")

    if key_name is None:
        key_name = "key"
    if new_dict is None:
        new_dict = {}

    def _full_key(key):
        return f"{_prefix}.{key}" if _prefix else key

    result = {}

    for key in dict.fromkeys(list(old_dict.keys()) + list(new_dict.keys())):
        fk = _full_key(key)
        in_old = key in old_dict
        in_new = key in new_dict

        if in_old and in_new:
            old_val = old_dict[key]
            new_val = new_dict[key]

            if isinstance(old_val, dict) and isinstance(new_val, dict):
                result[key] = update_dict(
                    old_val, new_val,
                    key_name=key_name, logger=logger,
                    _prefix=fk,
                )
            else:
                if logger is not None:
                    if old_val == new_val:
                        logger.debug(f"Kept {key_name} '{fk}' set to {format_obj(new_val)}.")
                    else:
                        logger.info(f"Set {key_name} '{fk}' from {format_obj(old_val)} to {format_obj(new_val)}.")
                result[key] = new_val
        elif in_old:
            if logger is not None:
                logger.debug(f"Kept {key_name} '{fk}' set to {format_obj(old_dict[key])}.")
            result[key] = old_dict[key]
        else:
            if logger is not None:
                logger.info(f"Set {key_name} '{fk}' to {format_obj(new_dict[key])}.")
            result[key] = new_dict[key]

    return result

def count_duplicates(iterable, *, include_unique=False):
    """
    Counts the occurrences of items in an iterable and returns the frequency of duplicates or all items.

    Args:
        iterable (iterable):
            A collection of hashable items to be counted.
        include_unique (bool, optional):
            If `True`, includes items that occur only once in the result. Defaults to `False`.

    Raises:
        UnrecoverableError: If input arguments are invalid.

    Returns:
        dict: A mapping of items to their respective occurrence counts.
    """
    # parse arguments
    if not iter(iterable):
        raise UnrecoverableError("Expected value of argument 'iterable' to be iterable.")
    assert_type_value(obj=include_unique, type_or_value=bool, name="argument 'include_unique'")

    # count duplicates
    count_dict = {}
    for item in iterable:
        try:
            hash(item)
        except Exception as e:
            assert_log(expression=False, message=f"Expected all elements in value of argument 'iterable' to be hashable: {repr(e)}")
        if item in count_dict:
            count_dict[item] += 1
        else:
            count_dict[item] = 1
    if include_unique:
        return count_dict
    return {key: value for key, value in count_dict.items() if value > 1}

# Printing

escape = {
    # ANSI escape codes for terminal text formatting, coloring, and control.

    # Attributes:
    #     Text Colors:
    #         - Standard: black, red, green, yellow, blue, magenta, cyan, white, gray
    #         - Dark: darkred, darkgreen, darkyellow, darkblue, darkmagenta, darkcyan, darkgray

    #     Background Colors:
    #         - Standard: bg_black, bg_red, bg_green, bg_yellow, bg_blue,
    #                     bg_magenta, bg_cyan, bg_white, bg_gray
    #         - Dark: bg_darkred, bg_darkgreen, bg_darkyellow, bg_darkblue,
    #                 bg_darkmagenta, bg_darkcyan, bg_darkgray

    #     Text Styles:
    #         - bold: Bold text
    #         - dim: Dim/faint text
    #         - italic: Italic text (may not be supported everywhere)
    #         - underline: Underlined text
    #         - blink: Blinking text
    #         - invert: Reverses foreground and background
    #         - hidden: Invisible text (still selectable)
    #         - strikethrough: Text with a line through it

    #     Miscellaneous Controls:
    #         - end: Reset all styles and colors to default
    #         - clear_line: Clears the current terminal line
    #         - clear_screen: Clears the entire terminal screen
    #         - carriage_return: Moves cursor to beginning of the line
    #         - bell: Triggers a terminal beep
    #         - cursor_hide: Hides the cursor
    #         - cursor_show: Shows the cursor

    # Usage:
    #     print(f"{escape['red']}Error:{escape['end']} Something went wrong.")
    #     print(f"{escape['bold']}{escape['green']}Success!{escape['end']}")
    #     print(f"{escape['clear_line']}{escape['carriage_return']}Overwriting this line...")

    # Foreground (text) colors
    'black': "\033[30m",
    'red': "\033[91m",
    'green': "\033[92m",
    'yellow': "\033[93m",
    'blue': "\033[94m",
    'magenta': "\033[95m",
    'cyan': "\033[96m",
    'white': "\033[97m",
    'gray': "\033[37m",
    'darkgray': "\033[90m",
    'darkred': "\033[31m",
    'darkgreen': "\033[32m",
    'darkyellow': "\033[33m",
    'darkblue': "\033[34m",
    'darkmagenta': "\033[35m",
    'darkcyan': "\033[36m",

    # Background colors
    'bg_black': "\033[40m",
    'bg_red': "\033[101m",
    'bg_green': "\033[102m",
    'bg_yellow': "\033[103m",
    'bg_blue': "\033[104m",
    'bg_magenta': "\033[105m",
    'bg_cyan': "\033[106m",
    'bg_white': "\033[107m",
    'bg_gray': "\033[47m",
    'bg_darkgray': "\033[100m",
    'bg_darkred': "\033[41m",
    'bg_darkgreen': "\033[42m",
    'bg_darkyellow': "\033[43m",
    'bg_darkblue': "\033[44m",
    'bg_darkmagenta': "\033[45m",
    'bg_darkcyan': "\033[46m",

    # Text styles
    'bold': "\033[1m",
    'dim': "\033[2m",
    'italic': "\033[3m",
    'underline': "\033[4m",
    'blink': "\033[5m",
    'invert': "\033[7m",
    'hidden': "\033[8m",
    'strikethrough': "\033[9m",

    # Miscellaneous controls
    'end': "\033[0m",
    'clear_line': "\033[2K",
    'clear_screen': "\033[2J",
    'carriage_return': "\r",
    'bell': "\a",
    'cursor_hide': "\033[?25l",
    'cursor_show': "\033[?25h",
}

def print_lines(string, *, prefix_first_line, prefix_next_lines, line_length, style):
    """
    Format and print a string to the terminal with line-wrapping and line-specific prefixes.

    Args:
        string (str):
            The text to be formatted and printed.
        prefix_first_line (str):
            The string prepended to the first line of the output followed by a colon.
        prefix_next_lines (str):
            The string prepended to all subsequent lines followed by a pipe character.
        line_length (int | None):
            The maximum number of characters allowed per line. If `None`, it defaults to the terminal width via `shutil.get_terminal_size()`.
        style (str):
            The ANSI escape sequence used to style the prefixes and line formatting.

    Raises:
        UnrecoverableError: If input arguments are invalid.

    Returns:
        tuple[list[str], str]: A tuple containing:
            - list[str]: A list of the wrapped lines (as raw strings).
            - str: The full formatted string including ANSI styles and prefixes.

    Notes:
        - The function preserves existing line breaks in 'string' and applies wrapping within those segments.
        - Leading indentation in lines is preserved during wrapping where possible.
        - Terminal width is determined with a fallback of infinity columns (no wrapping beyond line-breaks) if detection fails.
        - All printed lines are automatically encapsulated with reset escape codes.
    """
    # parse arguments
    assert_type_value(obj=string, type_or_value=str, name="argument 'string'")
    assert_type_value(obj=prefix_first_line, type_or_value=str, name="argument 'prefix_first_line'")
    assert_type_value(obj=prefix_next_lines, type_or_value=str, name="argument 'prefix_next_lines'")
    assert_type_value(obj=line_length, type_or_value=[int, None], name="argument 'line_length'")
    assert_type_value(obj=style, type_or_value=str, name="argument 'style'")

    if line_length is None:
        line_length = shutil.get_terminal_size(fallback=(float("inf"), 0)).columns
    first_line_length = max(line_length - len(prefix_first_line) - 2, 1)
    next_line_length = max(line_length - len(prefix_next_lines) - 2, 1)

    final_lines = []

    for line in string.splitlines():
        # If the line fits without wrapping, preserve it exactly (keeping all spaces)
        max_len = first_line_length if len(final_lines) == 0 else next_line_length
        if len(line) <= max_len:
            final_lines.append(line)
            continue

        stripped = line.lstrip()

        if len(stripped) > 0:
            leading_space = line[:len(line) - len(stripped)]
            words = stripped.split()
            parts = [leading_space] if len(leading_space) > 0 else []
            current_len = len(leading_space)
            has_content = False

            for word in words:
                while word:
                    max_len = first_line_length if len(final_lines) == 0 else next_line_length

                    if has_content:
                        space_left = max_len - current_len - 1
                    else:
                        space_left = max_len - current_len

                    if space_left <= 0:
                        if has_content:
                            final_lines.append("".join(parts))
                        remaining_len = current_len - max_len
                        parts = [" " * remaining_len] if remaining_len > 0 else []
                        current_len = remaining_len if remaining_len > 0 else 0
                        has_content = False
                    elif len(word) <= space_left:
                        if has_content:
                            parts.append(" ")
                            current_len += 1
                        parts.append(word)
                        current_len += len(word)
                        word = ""
                        has_content = True
                    elif not has_content or len(word) > max_len:
                        if has_content:
                            parts.append(" ")
                            current_len += 1
                        parts.append(word[:space_left])
                        current_len += space_left
                        word = word[space_left:]
                        final_lines.append("".join(parts))
                        parts = []
                        current_len = 0
                        has_content = False
                    else:
                        if has_content:
                            final_lines.append("".join(parts))
                        parts = []
                        current_len = 0
                        has_content = False

            if has_content:
                final_lines.append("".join(parts))
        else:
            final_lines.append("")

    printed = ""
    for i, line in enumerate(final_lines):
        if i == 0:
            if style == "":
                p = f"{prefix_first_line}: {line}"
            else:
                p = f"{style}{prefix_first_line}: {line}{escape['end']}"
        else:
            if style == "":
                p = f"{prefix_next_lines}| {line}"
            else:
                p = f"{style}{prefix_next_lines}| {line}{escape['end']}"
        print(p)
        printed = f"{printed}\n{p}"
    if len(printed) > 0:
        printed = printed[1:]

    return final_lines, printed

def format_obj(obj, *, cutoff=3000):
    """
    Formats a Python object into a string representation optimized for logging purposes.

    Args:
        obj (object):
            The object to be formatted. Typically a Python builtin type or a JSON-serializable object.
        cutoff (int | None, optional):
            The maximum character length allowed for strings or `dict` values before they are replaced
            with an excessive length placeholder. If `None`, no truncation is performed. Defaults to 3000.

    Returns:
        str: The formatted string representation of 'obj', wrapped in single quotes.

    Notes:
        - If 'obj' is a `str` and contains newlines, a newline is prepended to the result.
        - For `dict` objects, the function performs a `copy.deepcopy()` and replaces values
          exceeding 'cutoff' in length with a placeholder string starting with "<Excessive>".
        - Uses `json.dumps()` with an indentation of 2 for non-string objects to provide structured log output.
        - If an exception occurs during formatting, the function falls back to returning
          the standard `str()` representation of 'obj' wrapped in single quotes.
    """
    # parse arguments
    assert_type_value(obj=cutoff, type_or_value=[int, None], name="argument 'cutoff'")

    # format object
    try:
        if isinstance(obj, str):
            if cutoff is not None and len(obj) > cutoff: # TODO expose as core setting
                string = f"<Excessive>(type:'{type(obj).__name__}', str_length:'{len(obj)}')"
            else:
                string = obj
            if "\n" in obj:
                string = f"\n{string}"
            string = f"'{string}'"
        else:
            if isinstance(obj, dict):
                obj = copy.deepcopy(obj)
                for key in list(obj.keys()):
                    value = str(obj[key])
                    if cutoff is not None and len(value) > cutoff:
                        obj[key] = f"<Excessive>(type:'{type(obj[key]).__name__}', str_length:'{len(value)}')"
            if isinstance(obj, list):
                string = json.dumps(obj, indent=0)
                string = f"[{string[2:-2]}]".replace(",\n", ", ")
            else:
                string = json.dumps(obj, indent=2)
            if "\n" in string:
                string = f"\n{string}"
            string = f"'{string}'"
    except Exception:
        string = f"'{obj}'"
    return string
