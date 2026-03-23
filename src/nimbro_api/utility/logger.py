import time
import copy
import inspect
import datetime
import threading

import nimbro_api
from .misc import escape, assert_type_value, assert_keys, assert_log, print_lines, update_dict

lock = threading.Lock()

default_settings = {
    # Logger severity in [10, 20, 30, 40, 50] (int) or None to adopt global process-wide severity.
    'severity': None,
    # Logger name (str).
    'name': None
}

class Logger:
    """
    Easily configurable Logger with severity, name, suffix, and various filters.
    """

    def __init__(self, settings=None, *, core_settings=None):
        """
        Initialize the Logger.

        Args:
            settings (dict | None, optional):
                Configuration settings to override defaults. Defaults to `None`.
            core_settings (dict | None, optional):
                Global process-wide configuration settings.
                If provided, they are not pulled from the nimbro_api module. Defaults to `None`.

        Raises:
            UnrecoverableError: If input arguments are invalid.
        """
        # state
        self._settings = default_settings
        self._core_settings = core_settings
        self._name, self._severity = None, None
        self._once_fired = set()
        self._skip_first_seen = set()
        self._last_throttle_time = {}
        self._bypass = False
        self._method_severity = {
            'debug': 10,
            'info': 20,
            'warn': 30,
            'error': 40,
            'fatal': 50
        }
        self._method_escape = {
            'debug': f"{escape['darkgreen']}",
            'info': "",
            'warn': f"{escape['yellow']}",
            'error': f"{escape['darkred']}",
            'fatal': f"{escape['red']}"
        }
        self._method_prefix = {
            'debug': "[DEBUG]",
            'info': "[INFO ]",
            'warn': "[WARN ]",
            'error': "[ERROR]",
            'fatal': "[FATAL]"
        }

        settings = update_dict(old_dict=default_settings, new_dict=settings)
        self.set_settings(settings=settings)

    def _log(self, method, text, *, once=False, skip_first=False, throttle=None, suffix=None, core_settings=None):
        """
        Internal logging method applying filters and thread safety.

        Args:
            method (str):
                Logging method name ("debug", "info", "warn", "error", "fatal").
            text (object):
                The content to be logged.
            once (bool, optional):
                If `True`, log only the first occurrence for this call-site. Defaults to `False`.
            skip_first (bool, optional):
                If `True`, skip the first occurrence for this call-site. Defaults to `False`.
            throttle (int | float | None, optional):
                Minimum interval in seconds between logs for this call-site. Defaults to `None`.
            suffix (str | None, optional):
                Optional string appended to the log prefix. Defaults to `None`.
            core_settings (dict | None, optional):
                Global process-wide configuration settings.
                If not provided, they are pulled from the `nimbro_api` module. Defaults to `None`.

        Raises:
            UnrecoverableError: If input arguments are invalid.

        Returns:
            bool: `True` if the message was logged, `False` otherwise.

        Notes:
            - This method is thread-safe using a global `lock`.
            - Filters ('once', 'skip_first', 'throttle') are tracked per call-site (filename, function name, line number).
            - If severity level in 'core_settings' or local '_settings' is higher than 'method', the log is suppressed.
            - If 'text' fails to cast to `str`, a warning is logged instead.
        """
        stamp = datetime.datetime.now()
        # stamp = datetime.datetime.now(datetime.timezone.utc)

        # bypass
        if self._bypass:
            return False

        # parse arguments
        assert_type_value(obj=method, type_or_value=list(self._method_severity.keys()), name="argument 'method'")
        assert_type_value(obj=once, type_or_value=bool, name="argument 'once'")
        assert_type_value(obj=skip_first, type_or_value=bool, name="argument 'skip_first'")
        assert_type_value(obj=throttle, type_or_value=[int, float, None], name="argument 'throttle'")
        assert_type_value(obj=suffix, type_or_value=[None, str], name="argument 'suffix'")
        if isinstance(suffix, str):
            assert_log(len(suffix) > 0, "Expected argument 'suffix' provided as 'str' to be non-empty.")

        lock.acquire()

        if self._core_settings is None:
            core_settings = nimbro_api.get_settings()
        else:
            core_settings = self._core_settings

        # check severity and global mute
        severity = self._settings['severity']
        if severity is None:
            severity = core_settings['logger_severity']
        mute = core_settings['logger_mute']
        if mute or severity > self._method_severity[method]:
            lock.release()
            return False

        # identify call-site key
        frame = inspect.currentframe().f_back
        key = (frame.f_code.co_name, frame.f_code.co_filename, frame.f_lineno)

        # apply filter "once"
        if once:
            if key in self._once_fired:
                lock.release()
                return False
            self._once_fired.add(key)

        # apply filter "skip_first"
        if skip_first:
            if key not in self._skip_first_seen:
                self._skip_first_seen.add(key)
                lock.release()
                return False

        # apply filter "throttle"
        if throttle is not None:
            now = time.time()
            last = self._last_throttle_time.get(key)
            if last is not None and (now - last) < throttle:
                lock.release()
                return False
            self._last_throttle_time[key] = now

        # cast text to string
        try:
            text_str = str(text)
        except Exception:
            if method in ["debug", "info"]:
                method = "warn"
            text_str = f"Failed to cast log to string ({key})."
            once = False
            skip_first = False
            throttle = None

        # log
        stamp_str = f"[{stamp.isoformat()[:23]}]"
        stamp_str = stamp_str[:11] + " " + stamp_str[12:]
        # stamp_str = f"[{stamp.isoformat()}]"
        name_str = f"[{self._settings['name']}]" if self._settings['name'] else ""
        suffix_str = f"[{suffix}]" if suffix else ""
        prefix_first_line = f"{stamp_str}{self._method_prefix[method]}{name_str}{suffix_str}"
        if core_settings['logger_multi_line_prefix']:
            prefix_next_lines = prefix_first_line
        else:
            prefix_next_lines = " " * len(prefix_first_line)

        print_lines(
            string=text_str,
            prefix_first_line=prefix_first_line,
            prefix_next_lines=prefix_next_lines,
            line_length=core_settings['logger_line_length'],
            style=self._method_escape[method]
        )

        lock.release()
        return True

    # settings

    def get_settings(self):
        """
        Retrieve the current settings of the Logger.

        Returns:
            dict: A deep copy of the current settings.
        """
        return copy.deepcopy(self._settings)

    def set_settings(self, settings):
        """
        Update settings of the Logger.

        Args:
            settings (dict):
                New settings to apply.

        Raises:
            UnrecoverableError: If input arguments or provided settings are invalid.
        """
        assert_type_value(obj=settings, type_or_value=[dict, None], name="argument 'settings'")
        settings = update_dict(old_dict=self._settings, new_dict=settings)
        assert_keys(obj=settings, keys=default_settings.keys(), mode="match", name="settings")

        assert_type_value(obj=settings['severity'], type_or_value=[10, 20, 30, 40, 50, None], name="setting 'severity'")
        assert_type_value(obj=settings['name'], type_or_value=[str, None], name="setting 'name'")
        if isinstance(settings['name'], str) and len(settings['name']) == 0:
            settings['name'] = None

        self._settings = settings

    # log

    def debug(self, *args, **kwargs):
        """
        Log a message with severity "debug" and optional filters.

        See `_log` for further details.
        """
        return self._log("debug", *args, **kwargs)

    def info(self, *args, **kwargs):
        """
        Log a message with severity "info" and optional filters.

        See `_log` for further details.
        """
        return self._log("info", *args, **kwargs)

    def warn(self, *args, **kwargs):
        """
        Log a message with severity "warn" and optional filters.

        See `_log` for further details.
        """
        return self._log("warn", *args, **kwargs)

    def error(self, *args, **kwargs):
        """
        Log a message with severity "error" and optional filters.

        See `_log` for further details.
        """
        return self._log("error", *args, **kwargs)

    def fatal(self, *args, **kwargs):
        """
        Log a message with severity "fatal" and optional filters.

        See `_log` for further details.
        """
        return self._log("fatal", *args, **kwargs)
