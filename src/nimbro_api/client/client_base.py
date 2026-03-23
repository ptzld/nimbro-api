import copy
import threading
import traceback

from ..utility.logger import Logger
from ..utility.misc import UnrecoverableError, assert_type_value, assert_keys, assert_log, update_dict, format_obj

class ClientBase:
    """
    Base class providing the core logic for API implementations, including
    thread-safe settings management, automated retries, and unified logging.

    To be used in conjunction with `nimbro_api.client.Client`.
    """

    def __init__(self, settings, default_settings, **kwargs):
        """
        Initialize the the implementation underlying an Client.

        Args:
            settings (dict | None):
                Settings initializing the Client. Missing settings are drawn from 'default_settings'.
                See the documentation of `get_settings()` for of comprehensive list of all available settings.
                Use `None` to use default settings. Defaults to `None`.
            default_settings (dict):
                The default settings of the Client.
            **kwargs:
                The initial settings (see `get_settings()`) can also be configured via keyword arguments.
                When doing so, 'settings' must be None or an empty `dict`.

        Raises:
            UnrecoverableError: If input arguments are invalid or initialization fails.

        Notes:
            - The class inheriting from `ClientBase` is responsible for setting
              `self._initialized = True` when its `__init__()` function terminates.
            - The class inheriting from `ClientBase` must overwrite `set_settings()` as described in the template shown below.
            - The `wrap()` function serves as a bridge between user-exposed functions in the class inheriting from `Client`
              and their implementation inheriting from `ClientBase` (see `example_function()` in both classes).
        """
        self._initialized = False

        # locks
        self._lock_api = threading.Lock()
        self._lock_settings = threading.Lock()

        # logger
        if len(kwargs) > 0:
            assert_log(expression=settings is None or len(settings) == 0, message="Failed during initialization: Expected settings to be passed as either a dictionary via the 'settings' argument or as keyword arguments, but not both.")
            settings = kwargs
        from ..core import CoreBase
        if isinstance(self, CoreBase):
            self._logger_settings = {
                'logger_mute': default_settings['logger_mute'],
                'logger_line_length': default_settings['logger_line_length'],
                'logger_multi_line_prefix': default_settings['logger_multi_line_prefix']
            }
            self._logger = Logger(
                settings={'severity': default_settings['logger_severity'], 'name': default_settings['logger_name']},
                core_settings=self._logger_settings
            )
            self._logger.debug(f"Initializing '{type(self).__name__}' object.")
        else:
            severity = default_settings['logger_severity']
            name = default_settings['logger_name']
            if isinstance(settings, dict):
                if settings.get('logger_severity') in [10, 20, 30, 40, 50, None]:
                    severity = settings.get('logger_severity')
                if 'logger_name' in settings and isinstance(settings['logger_name'], (str, type(None))):
                    name = settings.get('logger_name')
            self._logger = Logger(
                settings={'severity': severity, 'name': name}
            )
            self._logger.debug(f"Initializing '{type(self).__name__}' object.")

        # settings
        assert_type_value(obj=settings, type_or_value=[dict, None], name="argument 'settings'", logger=self._logger, prefix="Failed during initialization: ")
        assert_type_value(obj=default_settings, type_or_value=dict, name="argument 'default_settings'", logger=self._logger, prefix="Failed during initialization: ")
        for key in ['logger_severity', 'logger_name', 'retry']:
            assert_log(expression=key in default_settings, message=f"Failed during initialization: Expected default settings to contain the mandatory key '{key}'.")
        self._assert_no_dots_in_keys(settings=default_settings)

        if settings is None or len(settings) == 0:
            self._initial_settings = copy.deepcopy(default_settings)
        else:
            self._initial_settings = copy.deepcopy(settings)
        self._default_settings = copy.deepcopy(default_settings)
        self._settings = {}

        self.wrap(0, self.set_settings, default_settings if isinstance(self, CoreBase) else settings, "init")

    def wrap(self, responses, function, *args, **kwargs):
        """
        Wrap an implementation function to provide a standardized execution
        environment for all user-exposed API calls of the overlaying Client.

        This is intended to be used in the overlying Client wrapping the corresponding implementation
        in the corresponding ClientBase, providing a standardized execution environment and executing
        tasks including thread safety, exception handling, retry behavior, configuring settings, and logging.

        Args:
            responses (int):
                The number of additional response values expected from 'function'
                beyond the standard success flag `bool` and message `str`.
            function (callable):
                The implementation function to be executed. It must return a tuple
                of length 2 plus 'responses', where the first two elements must be
                a success flag `bool` and a natural language message `str`.
            *args:
                Positional arguments are passed directly to 'function'.
            **kwargs:
                All settings (see `get_settings()`) can also be configured via keyword arguments from here.
                Additionally, there are special keyword arguments that can be provided:
                    persist (bool):
                        If `True`, settings changed via keyword arguments from this function call are persistent.
                        If `False`, they are reverted to the state before the function call, once this function
                        terminates, regardless of success. Defaults to `False`.
                    mute (bool):
                        If `True`, all logs emitted by this function are muted. Defaults to `False`.

        Raises:
            UnrecoverableError: When a 'function' fails before this `ClientBase` object has finished initialization.

        Returns:
            tuple[bool, str, ...]: A tuple containing:
                - bool: `True` if the operation succeeded, `False` otherwise.
                - str: A descriptive message about the operation result.
                ...
        """
        if self._lock_api.locked():
            try:
                self._logger.debug(f"Waiting for API lock in '{function.__name__}()'.")
            except Exception:
                self._logger.debug("Waiting for API lock in 'wrap()'.")
        self._lock_api.acquire()

        skip = False
        toggle_bypass = False
        raise_error = False
        reset_logger = False

        if not callable(function):
            # check function
            if self._initialized:
                message = f"Failed in '{function.__name__}()': Expected argument 'function' to be a callable but got type '{type(function).__name__}'."
                self._logger.error(message)
                response = False, message, *[None] * responses
            else:
                message = f"Failed in '{function.__name__}()' during initialization: Expected argument 'function' to be a callable but got type '{type(function).__name__}'."
                self._logger.fatal(message)
                raise_error = True
            skip = True

        if not skip and not isinstance(responses, int) and responses >= 0:
            # check responses
            if self._initialized:
                message = f"Failed in '{function.__name__}()': Expected argument 'responses' to be a non-negative 'int' but got '{responses}'."
                self._logger.error(message)
                response = False, message, *[None] * responses
            else:
                message = f"Failed in '{function.__name__}()' during initialization: Expected argument 'responses' to be a non-negative 'int' but got '{responses}'."
                self._logger.fatal(message)
                raise_error = True
            skip = True

        if not skip:
            if len(kwargs) > 0:
                # acknowledge keyword argument 'mute'
                mute = kwargs.pop('mute', False)
                if isinstance(mute, bool):
                    if mute and not self._logger._bypass:
                        self._logger._bypass = True
                        toggle_bypass = True
                else:
                    if self._initialized:
                        message = f"Failed in '{function.__name__}()': Expected argument 'mute' to be of type 'bool' but got '{type(mute).__name__}'."
                        self._logger.error(message)
                        response = False, message, *[None] * responses
                    else:
                        message = f"Failed in '{function.__name__}()' during initialization: Expected argument 'mute' to be of type 'bool' but got '{type(mute).__name__}'."
                        self._logger.fatal(message)
                        raise_error = True
                    skip = True

        if not skip:
            if function.__name__ == "set_settings":
                persist = True
                # assert len(args) == 1, args
                assert args[0] is None or isinstance(args[0], dict), args[0]
                if len(kwargs) > 0:
                    if args[0] is None or len(args[0]) == 0:
                        args = (copy.deepcopy(kwargs),)
                        kwargs = {}
                    else:
                        if self._initialized:
                            message = f"Failed in '{function.__name__}()': Expected settings to be passed as either a dictionary via the 'settings' argument or as keyword arguments, but not both."
                            self._logger.error(message)
                            response = False, message, *[None] * responses
                        else:
                            message = f"Failed in '{function.__name__}()' during initialization: Expected settings to be passed as either a dictionary via the 'settings' argument or as keyword arguments, but not both."
                            self._logger.fatal(message)
                            raise_error = True
                        skip = True
                if not skip and isinstance(args[0], dict):
                    logger_settings_reset = self._logger.get_settings()
                    logger_settings_target = self._logger.get_settings()
                    if 'logger_name' in args[0] and isinstance(args[0]['logger_name'], str):
                        logger_settings_target['name'] = args[0]['logger_name']
                    if 'logger_severity' in args[0] and args[0]['logger_severity'] in [None, 10, 20, 30, 40, 50]:
                        logger_settings_target['severity'] = args[0]['logger_severity']
                    if logger_settings_reset != logger_settings_target:
                        reset_logger = True
                        self._logger.debug("Fast tracking logger settings within 'set_settings()'.")
                        self._logger.set_settings(settings=logger_settings_target)
            elif len(kwargs) == 0:
                persist = True
            else:
                # acknowledge keyword argument 'persist'
                persist = kwargs.pop('persist', False)
                if not isinstance(persist, bool):
                    if self._initialized:
                        message = f"Failed in '{function.__name__}()': Expected argument 'persist' to be of type 'bool' but got '{type(persist).__name__}'."
                        self._logger.error(message)
                        response = False, message, *[None] * responses
                    else:
                        message = f"Failed in '{function.__name__}()' during initialization: Expected argument 'persist' to be of type 'bool' but got '{type(persist).__name__}'."
                        self._logger.fatal(message)
                        raise_error = True
                    skip = True

        if not skip:
            # sort retries
            if 'retry' in kwargs:
                retry = kwargs['retry']
                if not (isinstance(retry, bool) or (isinstance(retry, int) and retry >= 0)):
                    if self._initialized:
                        message = f"Failed in '{function.__name__}()': Expected argument 'retry' to be of type 'bool' or a non-negative 'int' but got '{retry}' of type '{type(retry).__name__}'."
                        self._logger.error(message)
                        response = False, message, *[None] * responses
                    else:
                        message = f"Failed in '{function.__name__}()' during initialization: Expected argument 'retry' to be of type 'bool' or a non-negative 'int' but got '{retry}' of type '{type(retry).__name__}'."
                        self._logger.fatal(message)
                        raise_error = True
                    skip = True
            else:
                retry = self._settings.get('retry', False)
            if not skip:
                if isinstance(retry, bool):
                    if retry:
                        retry = -1
                    else:
                        retry = 0

        if not skip:
            # attempt function

            if retry == -1:
                self._logger.debug(f"Wrapping '{function.__name__}()' with infinite retries.")
            elif retry == 0:
                self._logger.debug(f"Wrapping '{function.__name__}()' without retries.")
            else:
                self._logger.debug(f"Wrapping '{function.__name__}()' with '{retry}' retr{'y' if retry == 1 else 'ies'}.")

            if len(kwargs) > 0 and function.__name__ != "set_settings":
                settings = self.get_settings(name=None)
                for kwarg in copy.deepcopy(kwargs):
                    if kwarg in settings and settings[kwarg] == kwargs[kwarg]:
                        kwargs.pop(kwarg)
                if len(kwargs) > 0:
                    stage = 0
                else:
                    stage = 1
            else:
                stage = 1

            attempt = 1

            while True:

                if stage == 0:
                    # apply keyword arguments as settings
                    try:
                        success, message = self.set_settings(settings=kwargs, mode="set" if persist else "temp")
                    except Exception as e:
                        if isinstance(e, UnrecoverableError):
                            if self._initialized:
                                if attempt == 1:
                                    message = f"Unrecoverable error while {'' if persist else 'temporarily '}overwriting settings before '{function.__name__}()': {e}"
                                else:
                                    message = f"Unrecoverable error while {'' if persist else 'temporarily '}overwriting settings before '{function.__name__}()' after attempt '{attempt}': {e}"
                                self._logger.error(message)
                                response = False, message, *[None] * responses
                            else:
                                if attempt == 1:
                                    message = f"Unrecoverable error while {'' if persist else 'temporarily '}overwriting settings before '{function.__name__}()' during initialization: {e}"
                                else:
                                    message = f"Unrecoverable error while {'' if persist else 'temporarily '}overwriting settings before '{function.__name__}()' after attempt '{attempt}' during initialization: {e}"
                                self._logger.fatal(message)
                                raise_error = True
                        else:
                            if self._initialized:
                                if attempt == 1:
                                    message = f"Unexpected error while {'' if persist else 'temporarily '}overwriting settings before '{function.__name__}()':\n{traceback.format_exc()}"
                                else:
                                    message = f"Unexpected error while {'' if persist else 'temporarily '}overwriting settings before '{function.__name__}()' after attempt '{attempt}':\n{traceback.format_exc()}"
                                self._logger.fatal(message)
                                response = False, message, *[None] * responses
                            else:
                                if attempt == 1:
                                    message = f"Unexpected error while {'' if persist else 'temporarily '}overwriting settings before '{function.__name__}()' during initialization:\n{traceback.format_exc()}"
                                else:
                                    message = f"Unexpected error while {'' if persist else 'temporarily '}overwriting settings before '{function.__name__}()' after attempt '{attempt}' during initialization:\n{traceback.format_exc()}"
                                self._logger.fatal(message)
                                raise_error = True
                        break
                    else:
                        if not isinstance(success, bool):
                            message = f"Expected 'set_settings()' to return a tuple where the first element is of type 'bool' but got '{type(success).__name__}'."
                            success = False
                        elif not isinstance(message, str):
                            success = False
                            message = f"Expected 'set_settings()' to return a tuple where the second element is of type 'str' but got '{type(message).__name__}'."
                        if success:
                            stage += 1
                            self._logger.debug(f"Temporarily overwritten settings: {list(kwargs.keys())}: {message}")
                        else:
                            if retry == -1 or retry >= attempt:
                                if retry == -1:
                                    self._logger.warn(f"Retrying to {'' if persist else 'temporarily '}overwrite settings before '{function.__name__}()' until successful after attempt '{attempt}': {message}")
                                else:
                                    retries_left = retry - attempt + 1
                                    self._logger.warn(f"Retrying to {'' if persist else 'temporarily '}overwrite settings before '{function.__name__}()' for '{retries_left}' more time{'' if retries_left == 1 else 's'} after failed attempt '{attempt}': {message}")
                                attempt += 1
                            else:
                                if self._initialized:
                                    if attempt == 1:
                                        message = f"Failed to {'' if persist else 'temporarily '}overwrite settings before '{function.__name__}()': {message}"
                                    else:
                                        message = f"Failed to {'' if persist else 'temporarily '}overwrite settings before '{function.__name__}()' after attempt '{attempt}': {message}"
                                    self._logger.error(message)
                                    response = False, message, *[None] * responses
                                else:
                                    if attempt == 1:
                                        message = f"Failed to {'' if persist else 'temporarily '}overwrite settings before '{function.__name__}()' during initialization: {message}"
                                    else:
                                        message = f"Failed to {'' if persist else 'temporarily '}overwrite settings before '{function.__name__}()' after attempt '{attempt}' during initialization: {message}"
                                    self._logger.fatal(message)
                                    raise_error = True
                                break

                if stage == 1:
                    # execute target
                    self._logger.debug(f"Executing '{function.__name__}()'.")
                    try:
                        response = function(*args)
                    except Exception as e:
                        if isinstance(e, UnrecoverableError):
                            if self._initialized:
                                if attempt == 1:
                                    message = f"Unrecoverable error in '{function.__name__}()': {e}"
                                else:
                                    message = f"Unrecoverable error in '{function.__name__}()' after attempt '{attempt}': {e}"
                                self._logger.error(message)
                                response = False, message, *[None] * responses
                            else:
                                if attempt == 1:
                                    message = f"Unrecoverable error in '{function.__name__}()' during initialization: {e}"
                                else:
                                    message = f"Unrecoverable error in '{function.__name__}()' after attempt '{attempt}' during initialization: {e}"
                                self._logger.fatal(message)
                                raise_error = True
                        else:
                            if self._initialized:
                                if attempt == 1:
                                    message = f"Unexpected error in '{function.__name__}()':\n{traceback.format_exc()}"
                                else:
                                    message = f"Unexpected error in '{function.__name__}()' after attempt '{attempt}':\n{traceback.format_exc()}"
                                self._logger.fatal(message)
                                response = False, message, *[None] * responses
                            else:
                                if attempt == 1:
                                    message = f"Unexpected error in '{function.__name__}()' during initialization:\n{traceback.format_exc()}"
                                else:
                                    message = f"Unexpected error in '{function.__name__}()' after attempt '{attempt}' during initialization:\n{traceback.format_exc()}"
                                self._logger.fatal(message)
                                raise_error = True
                        if persist:
                            break
                        else:
                            stage += 1
                    else:
                        valid = False
                        if not isinstance(response, tuple):
                            message = f"Expected '{function.__name__}()' to return a tuple but got '{type(response).__name__}'."
                        elif not len(response) == 2 + responses:
                            message = f"Expected '{function.__name__}()' to return a tuple of length '{2 + responses}' but got '{len(response)}'."
                        elif not isinstance(response[0], bool):
                            message = f"Expected '{function.__name__}()' to return a tuple where the first element is of type 'bool' but got '{type(response[0]).__name__}'."
                        elif not isinstance(response[1], str):
                            message = f"Expected '{function.__name__}()' to return a tuple where the second element is of type 'str' but got '{type(response[1]).__name__}'."
                        else:
                            valid = True
                        if valid:
                            if response[0]:
                                if function.__name__ != "set_settings":
                                    self._logger.info(f"{response[1]}")
                                if persist:
                                    if attempt == 1:
                                        message = f"Succeeded '{function.__name__}()'."
                                    else:
                                        message = f"Succeeded '{function.__name__}()' after attempt '{attempt}'."
                                    self._logger.debug(message)
                                    break
                                else:
                                    stage += 1
                            else:
                                if retry == -1 or retry >= attempt:
                                    if retry == -1:
                                        self._logger.warn(f"Retrying '{function.__name__}()' until successful after attempt '{attempt}': {response[1]}")
                                    else:
                                        retries_left = retry - attempt + 1
                                        self._logger.warn(f"Retrying '{function.__name__}()' for '{retries_left}' more time{'' if retries_left == 1 else 's'} after failed attempt '{attempt}': {response[1]}")
                                    attempt += 1
                                else:
                                    if self._initialized:
                                        if attempt == 1:
                                            message = f"Failed in '{function.__name__}()': {response[1]}"
                                        else:
                                            message = f"Failed in '{function.__name__}()' after attempt '{attempt}': {response[1]}"
                                        self._logger.error(message)
                                        # response = False, message, *[None] * responses
                                    else:
                                        if attempt == 1:
                                            message = f"Failed in '{function.__name__}()' during initialization: {response[1]}"
                                        else:
                                            message = f"Failed in '{function.__name__}()' after attempt '{attempt}' during initialization: {response[1]}"
                                        self._logger.fatal(message)
                                        raise_error = True
                                    if persist:
                                        break
                                    else:
                                        stage += 1
                        else:
                            if self._initialized:
                                if attempt == 1:
                                    message = f"Unexpected error in '{function.__name__}()': {message}"
                                else:
                                    message = f"Unexpected error in '{function.__name__}()' after attempt '{attempt}': {message}"
                                response = False, message, *[None] * responses
                            else:
                                if attempt == 1:
                                    message = f"Unexpected error in '{function.__name__}()' during initialization: {message}"
                                else:
                                    message = f"Unexpected error in '{function.__name__}()' after attempt '{attempt}' during initialization: {message}"
                                raise_error = True
                            self._logger.fatal(message)
                            if persist:
                                break
                            else:
                                stage += 1

                if stage == 2:
                    # revert settings
                    settings = {key: settings[key] for key in kwargs}
                    try:
                        success, message = self.set_settings(settings=settings, mode="revert")
                    except Exception as e:
                        if isinstance(e, UnrecoverableError):
                            if self._initialized:
                                if attempt == 1:
                                    message = f"Unrecoverable error while reverting settings after '{function.__name__}()': {e}"
                                    self._logger.error(message)
                                else:
                                    message = f"Unrecoverable error while reverting settings after '{function.__name__}()' after attempt '{attempt}': {e}"
                                self._logger.error(message)
                                # response = False, message, *[None] * responses
                                response[0] = False
                                response[1] = message
                            else:
                                if attempt == 1:
                                    message = f"Unrecoverable error while reverting settings after '{function.__name__}()' during initialization: {e}"
                                else:
                                    message = f"Unrecoverable error while reverting settings after '{function.__name__}()' after attempt '{attempt}' during initialization: {e}"
                                self._logger.fatal(message)
                                raise_error = True
                        else:
                            if self._initialized:
                                if attempt == 1:
                                    message = f"Unexpected error while reverting settings after '{function.__name__}()':\n{traceback.format_exc()}"
                                else:
                                    message = f"Unexpected error while reverting settings after '{function.__name__}()' after attempt '{attempt}':\n{traceback.format_exc()}"
                                self._logger.fatal(message)
                                # response = False, message, *[None] * responses
                                response[0] = False
                                response[1] = message
                            else:
                                if attempt == 1:
                                    message = f"Unexpected error while reverting settings after '{function.__name__}()' during initialization:\n{traceback.format_exc()}"
                                else:
                                    message = f"Unexpected error while reverting settings after '{function.__name__}()' after attempt '{attempt}' during initialization:\n{traceback.format_exc()}"
                                self._logger.fatal(message)
                                raise_error = True
                        break
                    else:
                        if not isinstance(success, bool):
                            message = f"Expected 'set_settings()' to return a tuple where the first element is of type 'bool' but got '{type(success).__name__}'."
                            success = False
                        elif not isinstance(message, str):
                            success = False
                            message = f"Expected 'set_settings()' to return a tuple where the second element is of type 'str' but got '{type(message).__name__}'."
                        if success:
                            self._logger.debug(f"Reverted settings: {list(kwargs.keys())}: {message}")
                            if response[0]:
                                if attempt == 1:
                                    message = f"Succeeded '{function.__name__}()'."
                                else:
                                    message = f"Succeeded '{function.__name__}()' after attempt '{attempt}'."
                            else:
                                if attempt == 1:
                                    message = f"Failed in '{function.__name__}()'."
                                else:
                                    message = f"Failed in '{function.__name__}()' after attempt '{attempt}'."
                            self._logger.debug(message)
                            break
                        else:
                            if retry == -1 or retry >= attempt:
                                if retry == -1:
                                    self._logger.warn(f"Retrying to revert settings after '{function.__name__}()' until successful after attempt '{attempt}': {message}")
                                else:
                                    retries_left = retry - attempt + 1
                                    self._logger.warn(f"Retrying to revert settings after '{function.__name__}()' for '{retries_left}' more time{'' if retries_left == 1 else 's'} after failed attempt '{attempt}': {message}")
                                attempt += 1
                            else:
                                if self._initialized:
                                    if attempt == 1:
                                        message = f"Failed to revert settings after '{function.__name__}()': {message}"
                                    else:
                                        message = f"Failed to revert settings after '{function.__name__}()' after attempt '{attempt}': {message}"
                                    self._logger.error(message)
                                    # response = False, message, *[None] * responses
                                    response[0] = False
                                    response[1] = message
                                else:
                                    if attempt == 1:
                                        message = f"Failed to revert settings after '{function.__name__}()' during initialization: {message}"
                                    else:
                                        message = f"Failed to revert settings after '{function.__name__}()' after attempt '{attempt}' during initialization: {message}"
                                    self._logger.fatal(message)
                                    raise_error = True
                                break

        if reset_logger and (raise_error or not response[0]):
            self._logger.debug("Resetting fast tracked logger settings after failure in 'set_settings()'.")
            self._logger.set_settings(settings=logger_settings_reset)
        if toggle_bypass:
            self._logger._bypass = not self._logger._bypass

        if self._lock_settings.locked():
            self._lock_settings.release()
        self._lock_api.release()
        if raise_error:
            raise UnrecoverableError(message)
        return response

    def get_settings(self, name):
        """
        Retrieve all current settings or a specific one.

        Args:
            name (str | None):
                If provided, the one setting with this name is returned directly.
                Use `None` to return all settings as a dictionary. Defaults to `None`.

        Raises:
            UnrecoverableError: If 'name' is provided and does not refer to an existing setting.

        Returns:
            any: A deep copy of the current settings (`dict`) or a single setting when providing 'name' (`any`).
        """
        assert_type_value(obj=name, type_or_value=list(self._default_settings.keys()) + [None], name="argument 'name'", logger=self._logger, prefix="Failed in 'get_settings()': ")

        if self._lock_settings.locked():
            self._logger.debug("Waiting for settings lock in 'get_settings()'.")
        self._lock_settings.acquire()

        if name is None:
            settings = copy.deepcopy(self._settings)
            # consolidate active endpoint (implying base implements set(get()) contract by accepting 'endpoint' as dict)
            if {'endpoints', 'endpoint'}.issubset(set(settings.keys())):
                if isinstance(settings['endpoints'], dict):
                    if isinstance(settings['endpoint'], str):
                        if settings['endpoint'] in settings['endpoints']:
                            if isinstance(settings['endpoints'][settings['endpoint']], dict):
                                if 'name' not in settings['endpoints'][settings['endpoint']]:
                                    settings['endpoint'] = {'name': settings['endpoint'], ** settings['endpoints'][settings['endpoint']]}
                                    del settings['endpoints']
        else:
            settings = copy.deepcopy(self._settings[name])

        self._lock_settings.release()
        return settings

    def set_settings(self, settings, mode="set"):
        """
        Configure all settings or a subset of them.

        Args:
            settings (dict | None, optional):
                New settings to apply. Settings not contained are kept.
                See the documentation of `get_settings()` for a comprehensive list of all available settings.
                Use `None` to reset all settings to their initial values. Defaults to `None`.
            mode (str):
                Signals where this function is used and configures the logging messages it emits.
                Must be in `["set", "temp", "revert", "reset"]`. Defaults to "set".
            **kwargs:
                All settings (see `get_settings()`) can also be configured via keyword arguments.
                When doing so, 'settings' must be `None` or an empty `dict`.

        Returns:
            tuple[bool, str]: A tuple containing:
                - bool: `True` if the operation succeeded, `False` otherwise.
                - str: A descriptive message about the operation result.

        Notes:
            - Overwrite this function in the class inheriting from `ClientBase` and address all its settings according to this template.
            - This function must not raise any Exception except `UnrecoverableError`, which prevents `wrap()` executing this function to consider retries.
            - This functions performs the following steps:
                1. Perform logging and obtain the full set of target settings via `_introduce_settings()`.
                2. Validate each individual setting and their combination, and raise an `UnrecoverableError` if the target settings are not valid.
                3. Apply the validated settings by updating member variables, if required, and then forward and return the settings via `_apply_settings()`.
            - The current settings can be accessed safely from within the object inheriting from `ClientBase` via `self._settings['...']`.
            - The pattern shown here ensures that `self._settings` is a validated and complete set of settings,
              and that it remains constant while a user-exposed function is executed using `wrap()`.
        """
        settings = self._introduce_settings(settings=settings, mode=mode)

        # Validate all settings

        # logger_severity
        assert_type_value(obj=settings['logger_severity'], type_or_value=[10, 20, 30, 40, 50, None], name="setting 'logger_severity'")

        # logger_name
        assert_type_value(obj=settings['logger_name'], type_or_value=[str, None], name="setting 'logger_name'")

        # retry
        assert_type_value(obj=settings['retry'], type_or_value=[int, bool], name="setting 'retry'")
        if not isinstance(settings['retry'], bool):
            assert_log(
                expression=settings['retry'] >= 0,
                message=f"Expected setting 'retry' to be of type 'bool' or a non-negative 'int' but got '{settings['retry']}'."
            )
            if settings['retry'] == 0:
                settings['retry'] = False

        # my_setting
        # assert_type_value(obj=settings['my_setting'], type_or_value=int, name="setting 'my_setting'")

        # Accommodate any member variable according to the new settings
        # self.my_variable = settings['my_setting']

        return self._apply_settings(settings, mode)

    # def example_function(self, number):
    #     # parse arguments
    #     assert_type_value(obj=number, type_or_value=int, name="argument 'number'")
    #     assert_log(expression=number > 0, message=f"Expected argument 'number' to be non-negative but got '{number}'.")

    #     # use API (check out `nimbro_api.utility.api` for some HTTP helpers useful for communicating with a real API)
    #     number = -number

    #     # conclude response
    #     success = True
    #     message = "successfully inverted the number."

    #     return success, message, number

    def _assert_no_dots_in_keys(self, settings, prefix=""):
        for key in settings:
            fk = f"{prefix}.{key}" if prefix else key
            assert_log(expression="." not in key, message=f"Failed during initialization: Expected the names of default settings to not contain '.' but got '{fk}'")
            if isinstance(settings[key], dict):
                self._assert_no_dots_in_keys(settings[key], prefix=fk)

    def _expand_dotted_keys(self, settings):
        result = {}
        leaf_paths = {}

        for key, value in settings.items():
            parts = key.split(".")
            path_so_far = ""

            d = result
            for i, part in enumerate(parts[:-1]):
                path_so_far = f"{path_so_far}.{part}" if path_so_far else part

                if part not in d:
                    d[part] = {}
                elif not isinstance(d[part], dict):
                    assert_log(expression=False, message=f"Conflicting settings: '{key}' tries to nest into '{path_so_far}', which was already set to a leaf value by '{leaf_paths[path_so_far]}'.")
                d = d[part]

            final_part = parts[-1]
            final_path = f"{path_so_far}.{final_part}" if path_so_far else final_part

            if final_part in d:
                if isinstance(d[final_part], dict) and not isinstance(value, dict):
                    conflicting = [p for p, k in leaf_paths.items() if p.startswith(final_path + ".")]
                    assert_log(expression=False, message=f"Conflicting settings: '{key}' tries to set '{final_path}' to a leaf value, but it already contains nested keys set by {[leaf_paths[c] for c in conflicting]}.")
                elif isinstance(d[final_part], dict) and isinstance(value, dict):
                    def _check_overlap(existing, incoming, path, source_key):
                        for k, v in incoming.items():
                            sub_path = f"{path}.{k}"
                            if k in existing:
                                if isinstance(existing[k], dict) and isinstance(v, dict):
                                    _check_overlap(existing[k], v, sub_path, source_key)
                                else:
                                    # Find which previous key set this
                                    prev_key = leaf_paths.get(sub_path, "a nested dict")
                                    assert_log(expression=False, message=f"Conflicting settings: '{source_key}' and '{prev_key}' both set '{sub_path}'.")

                    _check_overlap(d[final_part], value, final_path, key)

                    def _merge_and_track(existing, incoming, path, source_key):
                        for k, v in incoming.items():
                            sub_path = f"{path}.{k}"
                            if k in existing and isinstance(existing[k], dict) and isinstance(v, dict):
                                _merge_and_track(existing[k], v, sub_path, source_key)
                            else:
                                existing[k] = v
                                if not isinstance(v, dict):
                                    leaf_paths[sub_path] = source_key

                    _merge_and_track(d[final_part], value, final_path, key)
                    continue
                elif not isinstance(d[final_part], dict) and not isinstance(value, dict):
                    prev_key = leaf_paths.get(final_path, "a nested dict")
                    assert_log(
                        expression=False,
                        message=f"Conflicting settings: '{key}' and '{prev_key}' both set '{final_path}'."
                    )
                else:
                    assert_log(expression=False, message=f"Conflicting settings: '{key}' tries to set '{final_path}' to a nested dict, but it was already set to a leaf value by '{leaf_paths[final_path]}'.")

            d[final_part] = value

            if isinstance(value, dict):
                def _track_leaves(val, path, source_key):
                    for k, v in val.items():
                        sub_path = f"{path}.{k}"
                        if isinstance(v, dict):
                            _track_leaves(v, sub_path, source_key)
                        else:
                            leaf_paths[sub_path] = source_key
                _track_leaves(value, final_path, key)
            else:
                leaf_paths[final_path] = key

        return result

    def _count_leaves(self, settings, prevent_dots=False):
        count = 0
        for value in settings.values():
            if isinstance(value, dict):
                count += self._count_leaves(value)
            else:
                count += 1
        return count

    def _leaf_keys(self, settings, prefix=""):
        keys = []
        for key, value in settings.items():
            fk = f"{prefix}.{key}" if prefix else key
            if isinstance(value, dict):
                keys.extend(self._leaf_keys(value, fk))
            else:
                keys.append(fk)
        return keys

    def _walk_settings(self, old_dict, new_dict, *, mode, tense, prefix=""):
        for key in new_dict:
            fk = f"{prefix}.{key}" if prefix else key
            old_exists = key in old_dict
            old_val = old_dict[key] if old_exists else None
            new_val = new_dict[key]
            if old_exists and isinstance(old_val, dict) and isinstance(new_val, dict):
                self._walk_settings(old_val, new_val, mode=mode, tense=tense, prefix=fk)
            else:
                self._log_settings_leaf(fk, old_val, new_val, mode=mode, tense=tense, old_exists=old_exists)

    def _log_settings_leaf(self, prefix, old_val, new_val, *, mode, tense, old_exists=True):
        if mode == "init":
            if tense == "present":
                self._logger.debug(f"Initializing '{prefix}' to {format_obj(new_val)}.")
            else:
                if old_exists and old_val != new_val:
                    self._logger.info(f"Initialized '{prefix}' to {format_obj(new_val)}.")
                else:
                    self._logger.debug(f"Initialized '{prefix}' to {format_obj(new_val)}.")
            return

        if old_val == new_val:
            return

        from_str = format_obj(old_val)
        to_str = format_obj(new_val)

        if tense == "present":
            verbs = {"set": "Setting", "temp": "Temporarily setting", "revert": "Reverting", "reset": "Resetting"}
            self._logger.debug(f"{verbs[mode]} '{prefix}' from {from_str} to {to_str}.")
        else:
            verbs = {"set": "Set", "temp": "Temporarily set", "revert": "Reverted", "reset": "Reset"}
            self._logger.info(f"{verbs[mode]} '{prefix}' from {from_str} to {to_str}.")

    def _introduce_settings(self, settings, mode):
        if self._lock_settings.locked():
            self._logger.debug("Waiting for settings lock in 'set_settings()'.")
        self._lock_settings.acquire()
        assert_type_value(obj=settings, type_or_value=[dict, None], name="argument 'settings'")
        assert_type_value(obj=mode, type_or_value=["set", "temp", "revert", "reset", "init"], name="argument 'mode'")
        if settings is None:
            settings = copy.deepcopy(self._initial_settings)
            if self._initialized:
                mode = "reset"
            else:
                mode = "init"
        else:
            settings = self._expand_dotted_keys(settings)

        n_leaves = self._count_leaves(settings)
        n_total = self._count_leaves(self._default_settings, prevent_dots=True)
        leaf_keys = self._leaf_keys(settings)

        if mode == "set":
            self._logger.debug(f"Applying '{n_leaves}' of '{n_total}' setting{'' if n_total == 1 else 's'}: {leaf_keys}")
        elif mode == "temp":
            self._logger.debug(f"Temporarily applying '{n_leaves}' of '{n_total}' setting{'' if n_total == 1 else 's'}: {leaf_keys}")
        elif mode == "revert":
            self._logger.debug(f"Reverting '{n_leaves}' of '{n_total}' setting{'' if n_total == 1 else 's'}: {leaf_keys}")
        elif mode == "reset":
            self._logger.debug(f"Resetting all '{n_leaves}' setting{'' if n_total == 1 else 's'}: {leaf_keys}")
        elif mode == "init":
            self._logger.debug(f"Initializing all '{n_leaves}' setting{'' if n_total == 1 else 's'}: {leaf_keys}")

        assert_keys(obj=settings, keys=['mute', 'persist'], mode="blacklist", name="argument 'settings'") # reserved for `wrap()`
        assert_keys(obj=settings, keys=self._default_settings.keys(), mode="whitelist", name="argument 'settings'")
        if self._initialized:
            assert_type_value(obj=mode, type_or_value=["set", "temp", "revert", "reset"], name="argument 'mode'")
            if self._settings.keys() != self._default_settings.keys():
                raise RuntimeError("Unexpected mismatch between keys of settings and default settings.")
            self._walk_settings(self._settings, settings, mode=mode, tense="present")
            settings = update_dict(old_dict=self._settings, new_dict=settings)
        else:
            assert_log(expression=self._settings == {}, message="Expected internal settings to be empty before initializing.")
            merged = update_dict(old_dict=self._default_settings, new_dict=settings)
            self._walk_settings(self._default_settings, merged, mode="init", tense="present")
            settings = merged
        self._logger.debug("Validating settings.")

        # required settings
        assert_type_value(obj=settings['logger_severity'], type_or_value=[10, 20, 30, 40, 50, None], name="setting 'logger_severity'")
        assert_type_value(obj=settings['logger_name'], type_or_value=[str, None], name="setting 'logger_name'")
        assert_type_value(obj=settings['retry'], type_or_value=[int, bool], name="setting 'retry'")
        if not isinstance(settings['retry'], bool):
            assert_log(
                expression=settings['retry'] >= 0,
                message=f"Expected setting 'retry' to be of type 'bool' or a non-negative 'int' but got '{settings['retry']}'."
            )
            if settings['retry'] == 0:
                settings['retry'] = False

        return copy.deepcopy(settings)

    def _apply_settings(self, settings, mode):
        self._logger.debug("Applying settings.")
        assert_type_value(obj=settings, type_or_value=dict, name="argument 'settings'")
        if settings.keys() != self._default_settings.keys():
            self._lock_settings.release()
            raise RuntimeError("Unexpected mismatch between keys of settings and default settings.")
        assert_type_value(obj=mode, type_or_value=["set", "temp", "revert", "reset", "init"], name="argument 'mode'")
        if self._initialized:
            self._walk_settings(self._settings, settings, mode=mode, tense="past")
        else:
            self._walk_settings(self._default_settings, settings, mode="init", tense="past")
        self._logger.set_settings({'severity': settings['logger_severity'], 'name': settings['logger_name']})
        self._settings = settings
        self._lock_settings.release()
        if mode == "set":
            message = "Settings applied."
        elif mode == "temp":
            message = "Settings temporarily applied."
        elif mode == "revert":
            message = "Settings reverted."
        elif mode == "reset":
            message = "Settings reset."
        elif mode == "init":
            message = "Settings initialized."

        return True, message
