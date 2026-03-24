#!/usr/bin/env python3

import time
import inspect
import datetime
import threading
import traceback
import importlib.util
from pathlib import Path

import nimbro_api

from ..utility.misc import UnrecoverableError, assert_type_value, assert_log, escape, print_lines

log_raw, log_pretty = "", ""
lock = threading.Lock()

def _report(prefix, text, style):
    global log_raw, log_pretty
    lock.acquire()

    stamp = datetime.datetime.now()
    # stamp = datetime.datetime.now(datetime.timezone.utc)
    stamp_str = f"[{stamp.isoformat()[:23]}]"
    stamp_str = stamp_str[:11] + " " + stamp_str[12:]
    # stamp_str = f"[{stamp.isoformat()}]"
    prefix_first_line = f"{stamp_str}[TEST]{prefix}"
    if nimbro_api.get_settings('logger_multi_line_prefix'):
        prefix_next_lines = prefix_first_line
    else:
        prefix_next_lines = " " * len(prefix_first_line)

    lines, printed = print_lines(
        string=text,
        prefix_first_line=prefix_first_line,
        prefix_next_lines=prefix_next_lines,
        line_length=nimbro_api.get_settings(name='logger_line_length'),
        style=style
    )

    for i, line in enumerate(lines):
        if i == 0:
            log_raw = f"{log_raw}\n{prefix_first_line}: {line}".lstrip()
        else:
            log_raw = f"{log_raw}\n{prefix_next_lines}| {line}".lstrip()

    log_pretty = f"{log_pretty}\n{printed}".lstrip()

    lock.release()

def _test_utilities():
    failures = 0
    _report("", "Testing utilities.", escape['darkcyan'])
    stamp = time.perf_counter()
    # TODO test utilities, e.g. download:
    # "https://www.nimbro.net/AVATAR/images/NimbRo_Avatar_2022_11_01_Team.jpg"
    # "https://download.samplelib.com/mp3/sample-15s.mp3"
    # "https://bitcoin.org/bitcoin.pdf"

    try:
        nimbro_api.__author__
    except Exception:
        _report("[utility]", "Package attribute __author__ is not set.", escape['yellow'])
        failures += 1
    try:
        nimbro_api.__version__
    except Exception:
        _report("[utility]", "Package attribute __version__ is not set.", escape['yellow'])
        failures += 1

    _report("", f"Tested utilities with '{failures}' failure{'' if failures == 1 else 's'} in '{time.perf_counter() - stamp:.3f}s'.", escape['darkyellow'] if failures > 0 else escape['cyan'])
    return failures

def _test_function(prefix, fun, args):
    name = fun.__name__[1:] if fun.__name__.startswith('_') else fun.__name__
    _report(prefix, f"Executing '{name}'.", escape['gray'])
    stamp = time.perf_counter()
    try:
        if args is None:
            response = fun()
        else:
            response = fun(**args)
    except Exception as e:
        if isinstance(e, UnrecoverableError):
            text = f"Failed in '{name}' after '{time.perf_counter() - stamp:.3f}s': {e}"
        else:
            text = f"Failed in '{name}' after '{time.perf_counter() - stamp:.3f}s':\n{traceback.format_exc()}"
        _report(prefix, text, escape['darkred'])
        return 1
    text = f"Succeeded '{name}' in '{time.perf_counter() - stamp:.3f}s'"
    text = f"{text}." if response is None else f"{text}: {response}"
    _report(prefix, text, escape['white'])
    return 0

def _test_module(prefix, path, function):
    failures = 0
    _report(prefix, "Processing module.", escape['darkblue'])
    stamp = time.perf_counter()
    try:
        mod = importlib.import_module(path)
    except Exception:
        _report(prefix, f"Failed to import module:\n{traceback.format_exc()}", escape['darkred'])
        return 1
    for name, fun in inspect.getmembers(mod, inspect.isfunction):
        if name.startswith("test_") and (function is None or name.find(function) > 0):
            failures += _test_function(prefix=prefix, fun=fun, args={})
    _report(prefix, f"Processed module with '{failures}' failure{'' if failures == 1 else 's'} in '{time.perf_counter() - stamp:.3f}s'.", escape['yellow'] if failures > 0 else escape['blue'])
    return failures

def _test_api(path, module, function):
    failures = 0
    _report(f"[{path.name}]", "Testing API.", escape['darkcyan'])
    stamp = time.perf_counter()
    package_path = path.parent.parent
    test_dir = path / "test"
    if test_dir.is_dir():
        for python_path in sorted(test_dir.glob("*.py")):
            if not python_path.name.startswith("_") and (module is None or python_path.name.find(module) > -1):
                module_path = ".".join((package_path.name, *python_path.relative_to(package_path).with_suffix("").parts))
                failures += _test_module(prefix=f"[{path.name}][{python_path.name[:-3]}]", path=module_path, function=function)
    else:
        failures += 1
        _report(f"[{path.name}]", f"Test directory '{test_dir}' does not exist.", escape['darkred'])
    _report(f"[{path.name}]", f"Tested API with '{failures}' failure{'' if failures == 1 else 's'} in '{time.perf_counter() - stamp:.3f}s'.", escape['darkyellow'] if failures > 0 else escape['cyan'])
    return failures

def _test_client(module, name):
    # init
    n = 100
    warning = 3
    timeout = 10
    tic = time.perf_counter()
    client = [getattr(module, name)() for _ in range(n)][-1]
    toc = time.perf_counter()
    duration = toc - tic
    assert_log(expression=duration < timeout, message=f"Failed to create '{n}' client{'' if n == 1 else 's'} in less than '{timeout:.3f}s'.")
    if duration > warning:
        _report(f"[{module.__name__.split('.')[-1]}][{name}]", f"Creating '{n}' client{'' if n == 1 else 's'} took longer than '{warning:.3f}s'.", escape['yellow'])

    # get_settings
    settings = client.get_settings()
    assert_type_value(obj=settings, type_or_value=dict, name="result get_settings()")

    # defaults
    assert_log(expression='logger_severity' in settings, message="Expected to find setting 'logger_severity'.")
    assert_type_value(obj=settings['logger_severity'], type_or_value=None, name="setting 'logger_severity'")
    assert_log(expression='logger_name' in settings, message="Expected to find setting 'logger_name'.")
    assert_type_value(obj=settings['logger_name'], type_or_value=str, name="setting 'logger_name'")
    assert_log(expression='retry' in settings, message="Expected to find setting 'retry'.")

    # set_settings
    success, message = client.set_settings(retry=True)
    assert_type_value(obj=success, type_or_value=bool, name="success")
    assert_type_value(obj=message, type_or_value=str, name="message")
    assert_log(expression=success, message=message)
    settings = client.get_settings()
    assert_type_value(obj=settings, type_or_value=dict, name="result get_settings()")
    assert_log(expression='retry' in settings, message="Expected to find setting 'retry'.")
    assert_type_value(obj=settings['retry'], type_or_value=True, name="setting 'retry'")
    success, message = client.set_settings(settings={'retry': -1})
    assert_type_value(obj=success, type_or_value=bool, name="success")
    assert_type_value(obj=message, type_or_value=str, name="message")
    assert_log(expression=not success, message=message)
    success, message = client.set_settings(settings={'retry': 1})
    assert_type_value(obj=success, type_or_value=bool, name="success")
    assert_type_value(obj=message, type_or_value=str, name="message")
    assert_log(expression=success, message=message)
    settings = client.get_settings()
    assert_type_value(obj=settings, type_or_value=dict, name="result get_settings()")
    assert_log(expression='retry' in settings, message="Expected to find setting 'retry'.")
    assert_type_value(obj=settings['retry'], type_or_value=1, name="setting 'retry'")

def _common_tests(api, module):
    failures = 0
    _report("", "Performing common tests.", escape['darkcyan'])
    stamp = time.perf_counter()
    for api_name in nimbro_api.api.__all__:
        if api is None or api_name.find(api) > -1:
            failures_api = 0
            module_obj = getattr(nimbro_api.api, api_name)
            _report(f"[{api_name}]", "Handling API.", escape['darkblue'])
            stamp_api = time.perf_counter()
            for name in module_obj.__all__:
                if module is None or name.find(module) > -1:
                    failures_api += _test_function(prefix=f"[{api_name}][{name}]", fun=_test_client, args={'module': module_obj, 'name': name})
            _report(f"[{api_name}]", f"Handled API with '{failures_api}' failure{'' if failures_api == 1 else 's'} in '{time.perf_counter() - stamp_api:.3f}s'.", escape['yellow'] if failures_api > 0 else escape['blue'])
            failures += failures_api
    _report("", f"Performed common tests with '{failures}' failure{'' if failures == 1 else 's'} in '{time.perf_counter() - stamp:.3f}s'.", escape['darkyellow'] if failures > 0 else escape['cyan'])
    return failures

def test(api=None, *, module=None, function="utilities", common=True, utilities=True, severity=None):
    """
    Executes a comprehensive suite of tests including common client checks and API-specific test modules.

    Args:
        api (str | None, optional):
            The name of a specific API area to test. If `None`, all discovered APIs are processed. Defaults to `None`.
        module (str | None, optional):
            A sub-string filter used to select specific test modules (files) within an API's test directory. Defaults to `None`.
        function (str | None, optional):
            A sub-string filter used to select specific test functions within a module. Only functions starting with "test_" are considered. Defaults to "utilities" which must not require any outgoing connection.
        common (bool, optional):
            If `True`, performs common client tests across the selected APIs. Defaults to `True`.
        utilities (bool, optional):
            If `True`, performs general level-package tests. Defaults to `True`.
        severity (int | None, optional):
            The logging severity level to apply during test execution (e.g., 10, 20, 30, 49, 50). If `None`, the logger is muted. Defaults to `None`.

    Raises:
        UnrecoverableError: If input arguments are invalid or if internal assertions fail during test preparation.

    Returns:
        int: The total number of test failures encountered during execution.

    Notes:
        - The function automatically manages the project's global logger settings, restoring them to their original state after completion.
        - Two log files are generated in the "test" sub-directory of the package: "test.txt" (raw text) and "test_pretty.txt" (with ANSI color codes).
        - Test discovery relies on the directory structure where each API resides in a sub-directory of the "api" folder containing a "test" directory.
    """
    # parse arguments
    assert_type_value(obj=api, type_or_value=[str, None], name="argument 'api'")
    assert_type_value(obj=module, type_or_value=[str, None], name="argument 'module'")
    assert_type_value(obj=function, type_or_value=[str, None], name="argument 'function'")
    assert_type_value(obj=common, type_or_value=bool, name="argument 'common'")
    assert_type_value(obj=utilities, type_or_value=bool, name="argument 'utilities'")
    assert_type_value(obj=severity, type_or_value=[10, 20, 30, 49, 50, None], name="argument 'severity'")

    # set logger
    logger_mute = nimbro_api.get_settings(name='logger_mute')
    keys_cache = nimbro_api.get_settings(name='keys_cache')
    if severity is None:
        success, message = nimbro_api.set_settings(settings={'logger_mute': True, 'keys_cache': False})
    else:
        logger_severity = nimbro_api.get_settings(name='logger_severity')
        success, message = nimbro_api.set_settings(settings={'logger_mute': False, 'logger_severity': severity, 'keys_cache': False})
    assert_log(expression=success, message=message)

    global log_raw, log_pretty
    log_raw, log_pretty = "", ""

    # run tests
    _report("", "Starting tests.", escape['darkgreen'])
    stamp = time.perf_counter()
    failures = 0

    if utilities:
        failures += _test_utilities()

    if common:
        failures += _common_tests(api=api, module=module)
    package_path = Path(nimbro_api.__path__[0])
    api_root_path = package_path / "api"
    api_paths = [api_root_path / api] if api else [p for p in api_root_path.iterdir() if p.is_dir()]
    for path in api_paths:
        if not path.name.startswith("_"):
            failures += _test_api(path=path, module=module, function=function)
    _report("", f"Finished tests with '{failures}' failure{'' if failures == 1 else 's'} in '{time.perf_counter() - stamp:.3f}s'.", escape['red'] if failures > 0 else escape['green'])

    # reset logger severity
    success, message = nimbro_api.set_settings(settings={'logger_mute': logger_mute, 'keys_cache': keys_cache})
    assert_log(expression=success, message=message)
    if severity is not None:
        success, message = nimbro_api.set_settings(settings={'logger_severity': logger_severity})
        assert_log(expression=success, message=message)

    # write files
    path = package_path / "test"
    with open(path / "test.txt", "w", encoding="utf-8") as file:
        file.write(f"{log_raw}\n")
    with open(path / "test_pretty.txt", "w", encoding="utf-8") as file:
        file.write(f"{log_pretty}\n")

    return failures

if __name__ == '__main__':
    test()
