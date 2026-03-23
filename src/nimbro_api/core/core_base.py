import os
import time
import copy
import signal
import datetime
import threading
import traceback
from queue import SimpleQueue

from ..client import ClientBase
from ..utility.misc import UnrecoverableError, assert_type_value, assert_log

class CoreBase(ClientBase):

    def __init__(self, default_settings):
        super().__init__(settings=None, default_settings=default_settings)
        self._cache = {}
        self._defer_queue = SimpleQueue()
        self._defer_timer = None
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
        self._logger.debug("Core initialized.")
        self._initialized = True

    def _signal_handler(self, signum, frame):
        _signal_names = {v: k for k, v in signal.__dict__.items() if k.startswith("SIG") and not k.startswith("SIG_")}
        name = _signal_names.get(signum, f"Unknown ({signum})")
        filename = frame.f_code.co_filename
        lineno = frame.f_lineno
        self._logger.info(f"Terminating after receiving signal '{name}' at '{filename}:{lineno}'.")
        self._logger.debug("Traceback:\n" + "".join(traceback.format_stack(frame)))

        if self._defer_timer:
            self._logger.debug("Cancelling deferred job.")
            self._defer_timer.cancel()
        self._deferred_thread()

        signal.signal(signum, signal.SIG_DFL)
        os.kill(os.getpid(), signum)

    # settings

    def set_settings(self, settings, mode="set"):
        settings = self._introduce_settings(settings=settings, mode=mode)

        # validate settings
        assert_type_value(obj=settings['logger_severity'], type_or_value=[10, 20, 30, 40, 50], name="setting 'logger_severity'")
        assert_type_value(obj=settings['logger_name'], type_or_value=[str, None], name="setting 'logger_name'")
        assert_type_value(obj=settings['logger_mute'], type_or_value=bool, name="setting 'logger_mute'")
        assert_type_value(obj=settings['logger_line_length'], type_or_value=[int, None], name="setting 'logger_mute'")
        if isinstance(settings['logger_line_length'], int):
            assert_log(expression=settings['logger_line_length'] > 0, message=f"Expected setting 'logger_line_length' provided as 'int' to be greater zero but got '{settings['logger_line_length']}'.")
        assert_type_value(obj=settings['logger_multi_line_prefix'], type_or_value=bool, name="setting 'logger_multi_line_prefix'")
        assert_type_value(obj=settings['keys_hide'], type_or_value=bool, name="setting 'keys_hide'")
        assert_type_value(obj=settings['keys_cache'], type_or_value=bool, name="setting 'keys_cache'")
        assert_type_value(obj=settings['defer_delay'], type_or_value=[int, float], name="setting 'defer_delay'")
        assert_log(expression=settings['defer_delay'] >= 0, message=f"Expected setting 'defer_delay' to be non-negative but got '{settings['defer_delay']}'.")
        assert_type_value(obj=settings['retry'], type_or_value=[int, bool], name="setting 'retry'")
        if isinstance(settings['retry'], int):
            assert_log(
                expression=settings['retry'] >= 0,
                message=f"Expected setting 'retry' provided as 'int' to be non-negative but got '{settings['retry']}'."
            )

        # apply settings
        self._logger_settings['logger_mute'] = settings['logger_mute']
        self._logger_settings['logger_line_length'] = settings['logger_line_length']
        self._logger_settings['logger_multi_line_prefix'] = settings['logger_multi_line_prefix']
        return self._apply_settings(settings, mode)

    # API keys

    def get_api_key(self, name, skip_cache_update=False):
        # parse arguments
        assert_type_value(obj=name, type_or_value=[None, str], name="argument 'name'")

        # obtain key from cache
        success, message, data = self.query_cache(
            category="keys",
            identifier=name,
            age=None
        )
        self._logger.debug(message)

        if success:
            if isinstance(name, str):
                data = data['data']
                if self._settings['keys_hide']:
                    message = f"Found API key '{name}' in cache."
                else:
                    message = f"Found API key '{name}' ('{data}') in cache."
            else:
                message = f"Obtained all '{len(data)}' cached API keys."
        elif isinstance(name, str):
            # read key from environment variable
            try:
                data = os.getenv(name)
            except Exception as e:
                success = False
                message = f"Failed to read API key '{name}' from environment variables: {repr(e)}"
                data = None
            else:
                if data is None:
                    success = False
                    message = f"Failed to read API key '{name}' from unset environment variable."
                else:
                    success = True
                    if self._settings['keys_cache'] and not skip_cache_update:
                        self.update_cache(category="keys", identifier=name, data=data)
                        if self._settings['keys_hide']:
                            message = f"Read API key '{name}' from environment variables and cached it."
                        else:
                            message = f"Read API key '{name}' ('{data}') from environment variables and cached it."
                    else:
                        if self._settings['keys_hide']:
                            message = f"Read API key '{name}' from environment variables."
                        else:
                            message = f"Read API key '{name}' ('{data}') from environment variables."
        else:
            message = "There are no API keys in cache."

        return success, message, data

    def set_api_key(self, name, key):
        # parse arguments
        assert_type_value(obj=name, type_or_value=str, name="argument 'name'")
        assert_log(expression=len(name) > 0, message="Expected argument 'name' to be a non-empty string.")
        assert_type_value(obj=key, type_or_value=str, name="argument 'key'")
        assert_log(expression=len(key) > 0, message="Expected argument 'key' to be a non-empty string.")

        # obtain key from cache
        success = True
        _, message, key_as_is = self.get_api_key(name=name, skip_cache_update=True)
        self._logger.debug(message)

        # cache key
        if self._settings['keys_cache']:
            self.update_cache(category="keys", identifier=name, data=key)

        # write key to environment variable
        if key != key_as_is:
            try:
                os.environ[name] = key
            except Exception as e:
                success = False
                message = f"Failed to write API key '{name}' to environment variable: {repr(e)}"
            else:
                if self._settings['keys_hide']:
                    self._logger.debug(f"Written API key '{name}' to environment variable.")
                else:
                    self._logger.debug(f"Written API key '{name}' ('{key}') to environment variable.")

        # construct message
        if success:
            if self._settings['keys_hide']:
                if key_as_is is None:
                    message = f"Set API key '{name}'."
                elif key_as_is == key:
                    message = f"Kept API key '{name}' as is."
                else:
                    message = f"Updated API key '{name}'."
            else:
                if key_as_is is None:
                    message = f"Set API key '{name}' to '{key}'."
                elif key_as_is == key:
                    message = f"Kept API key '{name}' set to '{key}'."
                else:
                    message = f"Updated API key '{name}' to '{key}'."

        return success, message

    # caching

    def query_cache(self, category, identifier, age):
        # parse arguments
        assert_type_value(obj=category, type_or_value=[str, None], name="argument 'category'")
        assert_type_value(obj=identifier, type_or_value=[str, None], name="argument 'identifier'")
        assert_type_value(obj=age, type_or_value=[float, int, None], name="argument 'age'")
        if age is not None:
            assert_log(expression=age >= 0, message=f"Expected argument 'age' provided as '{type(age).__name__}' to be non-negative but got '{age}'.")
        if identifier is None:
            assert_log(expression=age is None, message=f"Expected argument 'age' to be None when 'identifier' is not provided but got '{age}'.")
        if category is None:
            assert_log(expression=identifier is None, message=f"Expected argument 'identifier' to be None when 'category' is not provided but got '{identifier}'.")

        # forward
        if category is None:
            success = True
            message = "Obtained cache."
            data = copy.deepcopy(self._cache)
        else:
            success = False
            if category in self._cache:
                if identifier is None:
                    success = True
                    message = f"Obtained cache category '{category}'."
                    data = copy.deepcopy(self._cache[category])
                elif identifier in self._cache[category]:
                    if age is None:
                        success = True
                        message = f"Obtained cached identifier '{identifier}' from category '{category}'."
                        data = copy.deepcopy(self._cache[category][identifier])
                    else:
                        cached_stamp = datetime.datetime.fromisoformat(self._cache[category][identifier]['stamp'])
                        now_stamp = datetime.datetime.now(datetime.timezone.utc)
                        delta = (now_stamp - cached_stamp).total_seconds()
                        if delta <= age:
                            success = True
                            message = f"Cached identifier '{identifier}' from category '{category}' with age '{delta:.3f}s' <= '{age:.3f}s' is fresh enough."
                            data = copy.deepcopy(self._cache[category][identifier])
                        else:
                            message = f"Cached identifier '{identifier}' from category '{category}' with age '{delta:.3f}s' > '{age:.3f}s' is too old."
                            data = None
                else:
                    message = f"Identifier '{identifier}' is not cached in category '{category}'."
                    data = None
            else:
                message = f"Category '{category}' is not cached."
                data = None

        return success, message, data

    def update_cache(self, category, identifier, data):
        # parse arguments
        assert_type_value(obj=category, type_or_value=str, name="argument 'category'")
        assert_type_value(obj=identifier, type_or_value=str, name="argument 'identifier'")

        # update
        try:
            data = copy.deepcopy(data)
        except Exception as e:
            success = False
            message = f"Failed to copy the provided data before caching it: {repr(e)}"
        else:
            success = True
            now_stamp = datetime.datetime.now(datetime.timezone.utc)
            data = {
                'stamp': now_stamp.isoformat(),
                'data': data
            }
            if category in self._cache:
                if identifier in self._cache[category]:
                    cached_stamp = datetime.datetime.fromisoformat(self._cache[category][identifier]['stamp'])
                    delta = (now_stamp - cached_stamp).total_seconds()
                    message = f"Updated identifier '{identifier}' with age '{delta:.3f}s' in category '{category}'."
                else:
                    message = f"Cached identifier '{identifier}' in existing category '{category}'."
                self._cache[category][identifier] = data
            else:
                message = f"Cached identifier '{identifier}' in new category '{category}'."
                self._cache[category] = {}
                self._cache[category][identifier] = data

        return success, message

    def clear_cache(self, category, identifier, age):
        # parse arguments
        assert_type_value(obj=category, type_or_value=[str, None], name="argument 'category'")
        assert_type_value(obj=identifier, type_or_value=[str, None], name="argument 'identifier'")
        assert_type_value(obj=age, type_or_value=[float, int, None], name="argument 'age'")
        if age is not None:
            assert_log(expression=age >= 0, message=f"Expected argument 'age' provided as '{type(age).__name__}' to be non-negative but got '{age}'.")
            now_stamp = datetime.datetime.now(datetime.timezone.utc)

        # delete

        deletions = 0

        if category is not None:
            cat_list = [category] if category in self._cache else []
        else:
            cat_list = list(self._cache.keys())

        for cat_key in cat_list:
            if identifier is not None:
                sub_list = [identifier] if identifier in self._cache[cat_key] else []
            else:
                sub_list = list(self._cache[cat_key].keys())

            for id_key in sub_list:
                item = self._cache[cat_key][id_key]

                should_delete = False
                if age is None:
                    should_delete = True
                else:
                    cached_stamp = datetime.datetime.fromisoformat(item['stamp'])
                    delta = (now_stamp - cached_stamp).total_seconds()
                    if delta > age:
                        should_delete = True

                if should_delete:
                    del self._cache[cat_key][id_key]
                    deletions += 1

            # Clean up empty category to prevent memory bloat
            if len(self._cache[cat_key]) == 0:
                del self._cache[cat_key]

        return True, f"Deleted '{deletions}' item{'' if deletions == 1 else 's'} from cache."

    # deferred jobs

    def _deferred_thread(self):
        stamp_thread = time.perf_counter()
        self._logger.debug("Deferred thread started.")
        jobs, errors, = 0, 0
        while not self._defer_queue.empty():
            # execute job
            stamp_job = time.perf_counter()
            job = self._defer_queue.get_nowait()
            self._logger.debug(f"Executing deferred job '{job[0].__name__}'.")
            try:
                job[0](job[1])
            except Exception as e:
                errors += 1
                if isinstance(e, UnrecoverableError):
                    self._logger.error(f"Deferred job '{job[0].__name__}' failed after '{time.perf_counter() - stamp_job:.3f}s': {e}")
                else:
                    self._logger.fatal(f"Deferred job '{job[0].__name__}' failed after '{time.perf_counter() - stamp_job:.3f}s':\n{traceback.format_exc()}")
            else:
                self._logger.debug(f"Deferred job '{job[0].__name__}' finished after '{time.perf_counter() - stamp_job:.3f}s'.")
            jobs += 1
        duration = f"'{time.perf_counter() - stamp_thread:.3f}s'"
        self._logger.debug(f"Deferred thread finished '{jobs}' job{'' if jobs == 1 else 's'} with '{errors}' failure{'' if errors == 1 else 's'} in {duration}.")
        return True, f"Executed '{jobs}' deferred job{'' if jobs == 1 else 's'} with '{errors}' failure{'' if errors == 1 else 's'} in {duration}."

    def register_deferred_job(self, job):
        # parse arguments
        assert_type_value(obj=job, type_or_value=[tuple, list], name="argument 'job'")
        assert_log(expression=len(job), message=f"Expected argument 'job' to be a tuple or list that contains exactly '2' elements but got '{len(job)}'.")
        assert_log(expression=callable(job[0]), message=f"Expected first element in argument 'job' to be callable but got '{type(job[0]).__name__}'.")
        assert_type_value(obj=job[1], type_or_value=dict, name="second element in argument 'job'")

        self._defer_queue.put_nowait(job)
        if self._defer_timer:
            self._defer_timer.cancel()
        self._defer_timer = threading.Timer(self._settings['defer_delay'], self._deferred_thread)
        message = "Registered deferred job."
        self._logger.debug(message)
        self._defer_timer.start()

        return True, message

    def execute_deferred_jobs(self):
        if self._defer_timer:
            self._logger.debug("Cancelling deferred thread timer.")
            self._defer_timer.cancel()
        return self._deferred_thread()
