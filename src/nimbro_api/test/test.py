#!/usr/bin/env python3

"""
Run common client checks, package utilities, and API-specific tests.

Install this checkout with Python 3.9+:
    pip install -e .

Run from the repository root:
    python src/nimbro_api/test/test.py
    python src/nimbro_api/test/test.py --provider openai --module classify
    python src/nimbro_api/test/test.py --no-common --no-utilities --severity debug

Call from Python with the same arguments and defaults:
    nimbro_api.test()
    nimbro_api.test(provider="openai", module="classify")

Defaults select offline API utility tests and also run common client checks and
package utilities. See --help for filters and logging controls. --function none
includes live tests requiring external services and credentials.

Test reports are always emitted. --severity controls underlying `nimbro_api` logs
and defaults to "off".

Results overwrite test.txt and test_pretty.txt in the package test directory.
The return value and script exit status are the total number of test failures.
"""

import sys
import time
import inspect
import argparse
import datetime
import threading
import traceback
import importlib.util
from pathlib import Path

import nimbro_api

from nimbro_api.utility.logger import Logger
from nimbro_api.utility.misc import UnrecoverableError, assert_type_value, assert_log, escape, print_lines

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

def _test_provider(path, module, function):
    failures = 0
    _report(f"[{path.name}]", "Testing API provider.", escape['darkcyan'])
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
    _report(f"[{path.name}]", f"Tested API provider with '{failures}' failure{'' if failures == 1 else 's'} in '{time.perf_counter() - stamp:.3f}s'.", escape['darkyellow'] if failures > 0 else escape['cyan'])
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

    # endpoint names
    if isinstance(settings.get('endpoint'), dict):
        endpoints = client.get_settings('endpoints')
        endpoint = settings['endpoint'].copy()
        for endpoint_name in ["", " ", "\t\n"]:
            registry = endpoints.copy()
            registry[endpoint_name] = {key: value for key, value in endpoint.items() if key != "name"}
            success, message = client.set_settings(endpoints=registry)
            assert_type_value(obj=success, type_or_value=bool, name="success")
            assert_type_value(obj=message, type_or_value=str, name="message")
            assert_log(expression=not success, message=f"Expected rejection of endpoint registry name '{endpoint_name}'.")
            assert_log(expression=client.get_settings() == settings, message="Expected invalid endpoint registry names to leave settings unchanged.")
            endpoint['name'] = endpoint_name
            success, message = client.set_settings(endpoint=endpoint)
            assert_type_value(obj=success, type_or_value=bool, name="success")
            assert_type_value(obj=message, type_or_value=str, name="message")
            assert_log(expression=not success, message=f"Expected rejection of inline endpoint name '{endpoint_name}'.")
            assert_log(expression=client.get_settings() == settings, message="Expected invalid inline endpoint names to leave settings unchanged.")
        endpoint['name'] = " test endpoint "
        success, message = client.set_settings(endpoint=endpoint)
        assert_log(expression=success, message=message)
        assert_type_value(obj=client.get_settings('endpoint'), type_or_value=" test endpoint ", name="preserved endpoint name")
        success, message = client.set_settings(settings)
        assert_log(expression=success, message=message)
        assert_log(expression=client.get_settings() == settings, message="Expected a settings round trip to preserve the selected endpoint.")

    # connect and read timeouts
    for setting in settings:
        if setting == "timeout_connect" or setting.startswith("timeout_read"):
            for value in [True, False, 0, -1, float('inf'), float('-inf'), float('nan')]:
                success, message = client.set_settings(**{setting: value})
                assert_type_value(obj=success, type_or_value=bool, name="success")
                assert_type_value(obj=message, type_or_value=str, name="message")
                assert_log(expression=not success, message=f"Expected rejection of setting '{setting}' with value '{value}'.")
                assert_log(expression=client.get_settings() == settings, message=f"Expected invalid setting '{setting}' to leave settings unchanged.")
            for value in [None, 1, 0.5]:
                success, message = client.set_settings(**{setting: value})
                assert_type_value(obj=success, type_or_value=bool, name="success")
                assert_type_value(obj=message, type_or_value=str, name="message")
                assert_log(expression=success, message=message)
                assert_type_value(obj=client.get_settings(setting), type_or_value=value, name=f"setting '{setting}'")
            success, message = client.set_settings(settings)
            assert_log(expression=success, message=message)

    # reset to initialization settings
    reset_client = getattr(module, name)(logger_name="Initial client")
    initial = reset_client.get_settings()
    registry = None
    if isinstance(initial.get('endpoint'), dict):
        registry = reset_client.get_settings('endpoints')
        endpoint = initial['endpoint'].copy()
        endpoint['name'] = "temporary endpoint"
        success, message = reset_client.set_settings(endpoint=endpoint)
        assert_log(expression=success, message=message)
    success, message = reset_client.set_settings(retry=True, logger_name="Changed client")
    assert_log(expression=success, message=message)
    success, message = reset_client.set_settings()
    assert_type_value(obj=success, type_or_value=bool, name="success")
    assert_type_value(obj=message, type_or_value=str, name="message")
    assert_log(expression=success, message=message)
    assert_log(expression=reset_client.get_settings() == initial, message="Expected resetting settings to restore all initialization settings.")
    if registry is not None:
        assert_log(expression=reset_client.get_settings('endpoints') == registry, message="Expected resetting settings to remove endpoint definitions added after initialization.")

def _common_tests(provider, module):
    failures = 0
    _report("", "Performing common tests.", escape['darkcyan'])
    stamp = time.perf_counter()
    for api_name in nimbro_api.api.__all__:
        if provider is None or api_name.find(provider) > -1:
            failures_api = 0
            module_obj = getattr(nimbro_api.api, api_name)
            _report(f"[{api_name}]", "Handling API provider.", escape['darkblue'])
            stamp_api = time.perf_counter()
            for name in module_obj.__all__:
                if module is None or name.find(module) > -1:
                    failures_api += _test_function(prefix=f"[{api_name}][{name}]", fun=_test_client, args={'module': module_obj, 'name': name})
            _report(f"[{api_name}]", f"Handled API provider with '{failures_api}' failure{'' if failures_api == 1 else 's'} in '{time.perf_counter() - stamp_api:.3f}s'.", escape['yellow'] if failures_api > 0 else escape['blue'])
            failures += failures_api
    _report("", f"Performed common tests with '{failures}' failure{'' if failures == 1 else 's'} in '{time.perf_counter() - stamp:.3f}s'.", escape['darkyellow'] if failures > 0 else escape['cyan'])
    return failures

def test(provider=None, *, module=None, function="utilities", common=True, utilities=True, severity="off"):
    """
    Executes a comprehensive suite of tests including common client checks and API-specific test modules.

    Args:
        provider (str | None, optional):
            The name of a specific API provider to test. If `None`, all discovered APIs are processed. Defaults to `None`.
        module (str | None, optional):
            A case-sensitive substring filter for API test filenames and common client class names.
            Use `None` to select all modules. Defaults to `None`.
        function (str | None, optional):
            A case-sensitive substring filter for API test functions. Only functions starting with
            "test_" are considered. Use `None` to select all functions, including live tests requiring
            services and credentials. Defaults to "utilities", which selects offline utility tests.
        common (bool, optional):
            If `True`, performs common client tests across the selected APIs. Defaults to `True`.
        utilities (bool, optional):
            If `True`, performs package utility checks. Defaults to `True`.
        severity (str, optional):
            Logger severity in ["debug", "info", "warn", "error", "fatal", "off"] for underlying `nimbro_api` logs.
            Use "off" to mute all underlying logs. Test reports are always emitted. Defaults to "off".

    Raises:
        UnrecoverableError: If input arguments are invalid or if internal assertions fail during test preparation.

    Returns:
        int: The total number of test failures encountered during execution.

    Notes:
        - Two log files are generated in the "test" sub-directory of the package: "test.txt" (raw text) and "test_pretty.txt" (with ANSI color codes).
        - Test discovery relies on the directory structure where each API provider resides in a sub-directory of the "api" folder containing a "test" directory.
        - Logging and credential cache settings changed by the runner are restored after execution.
    """
    # parse arguments
    assert_type_value(obj=provider, type_or_value=[str, None], name="argument 'provider'")
    assert_type_value(obj=module, type_or_value=[str, None], name="argument 'module'")
    assert_type_value(obj=function, type_or_value=[str, None], name="argument 'function'")
    assert_type_value(obj=common, type_or_value=bool, name="argument 'common'")
    assert_type_value(obj=utilities, type_or_value=bool, name="argument 'utilities'")
    assert_type_value(obj=severity, type_or_value=[str, int], name="argument 'severity'")
    severity = Logger(severity=severity).get_settings(name='severity')

    # set logger
    original_settings = {name: nimbro_api.get_settings(name=name) for name in ['logger_mute', 'logger_severity', 'keys_cache']}
    global log_raw, log_pretty
    log_raw, log_pretty = "", ""
    package_path = Path(nimbro_api.__path__[0])
    try:
        success, message = nimbro_api.set_settings(settings={
            'logger_mute': severity == "off",
            'logger_severity': severity,
            'keys_cache': False
        })
        assert_log(expression=success, message=message)

        # run tests
        _report("", "Starting tests.", escape['darkgreen'])
        stamp = time.perf_counter()
        failures = 0

        if utilities:
            failures += _test_utilities()

        if common:
            failures += _common_tests(provider=provider, module=module)
        api_root_path = package_path / "api"
        api_paths = [api_root_path / provider] if provider else [p for p in api_root_path.iterdir() if p.is_dir()]
        for path in api_paths:
            if not path.name.startswith("_"):
                failures += _test_provider(path=path, module=module, function=function)
        _report("", f"Finished tests with '{failures}' failure{'' if failures == 1 else 's'} in '{time.perf_counter() - stamp:.3f}s'.", escape['red'] if failures > 0 else escape['green'])

        # ensure deferred jobs are drained
        nimbro_api.execute_deferred_jobs()
    finally:
        # restore settings
        success, message = nimbro_api.set_settings(settings=original_settings, mute=True)
        assert_log(expression=success, message=message)

        # write files
        path = package_path / "test"
        with open(path / "test.txt", "w", encoding="utf-8") as file:
            file.write(f"{log_raw}\n")
        with open(path / "test_pretty.txt", "w", encoding="utf-8") as file:
            file.write(f"{log_pretty}\n")

    return failures

def _optional_filter(value):
    return None if value.lower() == "none" else value

def _severity(value):
    return int(value) if value.isdigit() else value

def _arguments():
    parser = argparse.ArgumentParser(description="Run NimbRo API tests. Default utility tests do not contact providers.", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--provider", type=_optional_filter, help="API provider name. Use 'none' to select all providers.")
    parser.add_argument("--module", type=_optional_filter, help="Test module/class substring filter, or 'none'.")
    parser.add_argument("--function", type=_optional_filter, help="Test function substring filter. Use 'none' to include live tests requiring services and credentials.")
    parser.add_argument("--common", action=argparse.BooleanOptionalAction, help="Run common client checks.")
    parser.add_argument("--utilities", action=argparse.BooleanOptionalAction, help="Run package utility checks.")
    parser.add_argument("--severity", type=_severity, choices=["debug", "info", "warn", "error", "fatal", "off", 0, 10, 20, 30, 40, 50], metavar="{debug,info,warn,error,fatal,off}", help="Logger severity for underlying nimbro_api logs. Use 'off' to mute those logs. Test reports are always emitted.")
    parser.set_defaults(**{name: parameter.default for name, parameter in inspect.signature(test).parameters.items()})
    return vars(parser.parse_args())

if __name__ == "__main__":
    sys.exit(test(**_arguments()))
