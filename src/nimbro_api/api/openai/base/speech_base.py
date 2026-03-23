import os
import time
import random
import string
import datetime
from pathlib import Path

import nimbro_api
from nimbro_api.client import ClientBase
from nimbro_api.utility.io import read_json, write_json, encode_b64, get_cache_location, acquire_lock, release_lock
from nimbro_api.utility.api import get_api_key, validate_endpoint, post_request
from nimbro_api.utility.misc import UnrecoverableError, assert_type_value, assert_keys, assert_log
from ..utility import validate_connection, get_models

class SpeechBase(ClientBase):

    def __init__(self, settings, default_settings, **kwargs):
        super().__init__(settings=settings, default_settings=default_settings, **kwargs)
        self.get_api_key = get_api_key.__get__(self)
        self.validate_connection = validate_connection.__get__(self)
        self.get_models = get_models.__get__(self)
        self._logger.debug(f"Initialized '{type(self).__name__}' object.")
        self._initialized = True

    def set_settings(self, settings, mode="set"):
        settings = self._introduce_settings(settings=settings, mode=mode)

        # endpoints
        assert_type_value(obj=settings['endpoints'], type_or_value=dict, name="setting 'endpoints'")
        assert_log(expression=len(settings['endpoints']) > 0, message="Expected setting 'endpoints' to define at least one endpoint.")
        for endpoint in settings['endpoints']:
            assert_type_value(obj=endpoint, type_or_value=str, name="all endpoint names in setting 'endpoints'")
            assert_log(expression=len(endpoint) > 0, message="Expected all endpoint names in setting 'endpoints' to be non-empty.")
            validate_endpoint(endpoint=settings['endpoints'][endpoint], flavors=None, require_key=True, require_name=False, setting_name=f"endpoint '{endpoint}' in setting 'endpoints'")

        # endpoint
        if isinstance(settings['endpoint'], dict):
            validate_endpoint(endpoint=settings['endpoint'], flavors=None, require_key=True, require_name=True, setting_name="endpoint provided through setting 'endpoint'")
            settings['endpoints'][settings['endpoint']['name']] = settings['endpoint']
            settings['endpoint'] = settings['endpoint']['name']
            del settings['endpoints'][settings['endpoint']]['name']
        else:
            assert_type_value(obj=settings['endpoint'], type_or_value=list(settings['endpoints'].keys()), name="setting 'endpoint'")

        # model
        assert_type_value(obj=settings['model'], type_or_value=str, name="setting 'model'")

        # validate_model
        assert_type_value(obj=settings['validate_model'], type_or_value=[float, int, bool], name="setting 'validate_model'")
        if isinstance(settings['validate_model'], (int, float)):
            assert_log(
                expression=settings['validate_model'] >= 0,
                message=f"Expected setting 'validate_model' provided as '{type(settings['validate_model']).__name__}' to be non-negative but got '{settings['validate_model']}'."
            )

        # voice_presets
        assert_type_value(obj=settings['voice_presets'], type_or_value=[str, None], name="setting 'voice_presets'")
        if settings['voice_presets'] is None:
            voice_presets = {}
        elif self._settings.get('voice_presets') == settings['voice_presets']:
            voice_presets = self._voice_presets
        else:
            if settings['voice_presets'] == "":
                settings['voice_presets'] = str(Path(__file__).resolve().parent.parent / "voice_presets.json")
            success, message, voice_presets = read_json(file_path=settings['voice_presets'], name="voice-presets", logger=self._logger)
            if success:
                self._logger.debug(message)
            else:
                raise UnrecoverableError(message)
        if settings['voice'] in voice_presets:
            if settings['instructions'] == "":
                self._logger.debug(f"Applying instructions from preset '{settings['voice']}'.")
                settings['instructions'] = voice_presets[settings['voice']]['instructions']
            self._logger.debug(f"Applying voice from preset '{settings['voice']}'.")
            settings['voice'] = voice_presets[settings['voice']]['voice']
        elif settings['instructions'] in voice_presets:
            if settings['voice'] == "":
                self._logger.debug(f"Applying voice from preset '{settings['instructions']}'.")
                settings['voice'] = voice_presets[settings['instructions']]['voice']
            self._logger.debug(f"Applying instructions from preset '{settings['instructions']}'.")
            settings['instructions'] = voice_presets[settings['instructions']]['instructions']

        # voice
        assert_type_value(obj=settings['voice'], type_or_value=str, name="setting 'voice'")

        # instructions
        assert_type_value(obj=settings['instructions'], type_or_value=str, name="setting 'instructions'")

        # speed
        assert_type_value(obj=settings['speed'], type_or_value=[float, int], name="setting 'speed'")
        speed_range = [0.25, 4.0]
        assert_log(
            expression=speed_range[0] <= settings['speed'] <= speed_range[1],
            message=f"Expected setting 'speed' to be in interval [{speed_range[0]}, {speed_range[1]}] but got '{settings['speed']}'."
        )
        if isinstance(settings['speed'], int):
            settings['speed'] = float(settings['speed'])

        # timeout_connect
        assert_type_value(obj=settings['timeout_connect'], type_or_value=[float, int, None], name="setting 'timeout_connect'")
        if settings['timeout_connect'] is not None:
            assert_log(
                expression=settings['timeout_connect'] > 0.0,
                message=f"Expected setting 'timeout_connect' to be None or greater zero but got '{settings['timeout_connect']}'."
            )

        # timeout_read
        assert_type_value(obj=settings['timeout_read'], type_or_value=[float, int, None], name="setting 'timeout_read'")
        if settings['timeout_read'] is not None:
            assert_log(
                expression=settings['timeout_read'] > 0.0,
                message=f"Expected setting 'timeout_read' to be None or greater zero but got '{settings['timeout_read']}'."
            )

        # cache_folder
        assert_type_value(obj=settings['cache_folder'], type_or_value=[str, None], name="setting 'cache_folder'")
        if settings['cache_folder'] in ["", None]:
            settings['cache_folder'] = os.path.join(get_cache_location(), "speech")
            self._logger.debug(f"Using cache-folder '{settings['cache_folder']}'.")

        # cache_file
        assert_type_value(obj=settings['cache_file'], type_or_value=str, name="setting 'cache_file'")

        # return_path
        assert_type_value(obj=settings['return_path'], type_or_value=bool, name="setting 'return_path'")

        # return_encoding
        assert_type_value(obj=settings['return_encoding'], type_or_value=["bytes", "base64"], name="setting 'return_encoding'")

        # cache_read
        assert_type_value(obj=settings['cache_read'], type_or_value=[bool, int, float], name="setting 'cache_read'")
        if isinstance(settings['cache_read'], (int, float)):
            assert_log(
                expression=settings['cache_read'] >= 0,
                message=f"Expected setting 'cache_read' provided as '{type(settings['cache_read']).__name__}' to be non-negative but got '{settings['cache_read']}'."
            )

        # cache_write
        assert_type_value(obj=settings['cache_write'], type_or_value=bool, name="setting 'cache_write'")

        # apply settings
        self._endpoint = settings['endpoints'][settings['endpoint']]
        self._voice_presets = voice_presets
        return self._apply_settings(settings, mode)

    def query_speech(self, text):
        # read cache-index if required
        if self._settings['cache_read'] is not False:
            index = None
            cache_path = os.path.join(self._settings['cache_folder'], self._settings['cache_file'])

            # determine if reading is required
            if isinstance(self._settings['cache_read'], bool):
                read_required = True
            else:
                # obtain key from cache
                success, message, cache = nimbro_api.query_cache(
                    category="speech",
                    identifier=cache_path,
                    age=None,
                    mute=True
                )
                self._logger.debug(message)
                if success:
                    cached_stamp = datetime.datetime.fromisoformat(cache['stamp'])
                    now_stamp = datetime.datetime.now(datetime.timezone.utc)
                    delta = (now_stamp - cached_stamp).total_seconds()
                    read_required = delta >= self._settings['cache_read']
                    if read_required:
                        self._logger.debug(f"Reading cache-index '{cache_path}' is required after '{delta:.3f}s' since last read (>= '{self._settings['cache_read']:.3f}s').")
                    else:
                        self._logger.debug(f"Reading cache-index '{cache_path}' is not required after '{delta:.3f}s' since last read (< '{self._settings['cache_read']:.3f}s').")
                        index = cache['data']
                else:
                    read_required = True

            if read_required:
                # check cache-index exists
                if not os.path.isfile(cache_path):
                    self._logger.debug(f"Cache-index '{cache_path}' doesn't exist.")
                else:
                    # read cache-index
                    success, message, data = read_json(file_path=cache_path, name="cache-index", logger=self._logger)
                    if success:
                        self._logger.debug(message)
                        index = data
                        success, message = nimbro_api.update_cache(category="speech", identifier=cache_path, data=index, mute=True)
                        if success:
                            self._logger.debug(message)
                        else:
                            self._logger.warn(message)
                    else:
                        self._logger.warn(message)

            if index is not None:
                # check required speech is cached
                speech_file = None
                try:
                    assert_type_value(obj=index, type_or_value=dict, name="cache-index")
                    if self._settings['endpoint'] in index:
                        assert_type_value(obj=index, type_or_value=dict, name="cache-index")
                        if self._settings['model'] in index[self._settings['endpoint']]:
                            assert_type_value(obj=index[self._settings['endpoint']][self._settings['model']], type_or_value=dict, name=f"value of cache-index key '{self._settings['model']}'")
                            if self._settings['voice'] in index[self._settings['endpoint']][self._settings['model']]:
                                assert_type_value(obj=index[self._settings['endpoint']][self._settings['model']][self._settings['voice']], type_or_value=dict, name=f"value of cache-index key '{self._settings['voice']}'")
                                if str(self._settings['speed']) in index[self._settings['endpoint']][self._settings['model']][self._settings['voice']]:
                                    assert_type_value(obj=index[self._settings['endpoint']][self._settings['model']][self._settings['voice']][str(self._settings['speed'])], type_or_value=dict, name=f"value of cache-index key '{self._settings['speed']}'")
                                    if self._settings['instructions'] in index[self._settings['endpoint']][self._settings['model']][self._settings['voice']][str(self._settings['speed'])]:
                                        assert_type_value(obj=index[self._settings['endpoint']][self._settings['model']][self._settings['voice']][str(self._settings['speed'])][self._settings['instructions']], type_or_value=dict, name=f"value of cache-index key '{self._settings['instructions']}'")
                                        if text in index[self._settings['endpoint']][self._settings['model']][self._settings['voice']][str(self._settings['speed'])][self._settings['instructions']]:
                                            assert_type_value(obj=index[self._settings['endpoint']][self._settings['model']][self._settings['voice']][str(self._settings['speed'])][self._settings['instructions']][text], type_or_value=str, name=f"value of cache-index key '{text}'")
                                            speech_file = index[self._settings['endpoint']][self._settings['model']][self._settings['voice']][str(self._settings['speed'])][self._settings['instructions']][text]
                except UnrecoverableError as e:
                    self._logger.warn(f"Cache-index '{cache_path}' is corrupted: {e}")
                else:
                    if speech_file is None:
                        self._logger.debug(f"Speech not found in cache-index '{cache_path}'.")
                    else:
                        self._logger.debug(f"Found required speech in cache-index '{cache_path}'.")
                        # check referenced file exists
                        speech_path = os.path.join(self._settings['cache_folder'], speech_file)
                        if self._settings['return_path'] is True:
                            if os.path.isfile(speech_path):
                                return speech_path
                            else:
                                self._logger.warn(f"Cache-index '{cache_path}' references audio-file '{speech_path}' that does not exist.")
                        else:
                            # open file
                            self._logger.debug(f"Reading referenced audio-file '{speech_path}'.")
                            try:
                                with open(speech_path, 'rb') as f:
                                    speech_bytes = f.read()
                            except Exception as e:
                                self._logger.warn(f"Failed to read audio-file '{speech_path}' referenced in cache-index '{cache_path}': {repr(e)}")
                            else:
                                if self._settings['return_encoding'] == "bytes":
                                    return speech_bytes
                                elif self._settings['return_encoding'] == "base64":
                                    success, message, speech_b64 = encode_b64(obj=speech_bytes, name=f"audio-file '{speech_path}' referenced in cache-index '{cache_path}'", logger=self._logger)
                                    if success:
                                        self._logger.debug(message)
                                        return speech_b64
                                    else:
                                        self._logger.warn(message)
                                else:
                                    raise NotImplementedError(f"Unknown encoding '{self._settings['return_encoding']}'.")

    def cache_speech(self, job):
        # validate job
        try:
            assert_keys(obj=job, keys=['type', 'folder', 'index', 'file', 'endpoint', 'model', 'voice', 'instructions', 'speed', 'text', 'speech'], mode="match", name="job")
            assert_type_value(obj=job['folder'], type_or_value=str, name="cache-job key 'folder'")
            assert_type_value(obj=job['index'], type_or_value=str, name="cache-job key 'index'")
            assert_type_value(obj=job['file'], type_or_value=str, name="cache-job key 'file'")
            assert_type_value(obj=job['endpoint'], type_or_value=str, name="cache-job key 'endpoint'")
            assert_type_value(obj=job['model'], type_or_value=str, name="cache-job key 'model'")
            assert_type_value(obj=job['voice'], type_or_value=str, name="cache-job key 'voice'")
            assert_type_value(obj=job['instructions'], type_or_value=str, name="cache-job key 'instructions'")
            assert_type_value(obj=job['speed'], type_or_value=float, name="cache-job key 'speed'")
            assert_type_value(obj=job['text'], type_or_value=str, name="cache-job key 'text'")
            assert_type_value(obj=job['speech'], type_or_value=[bytes, None], name="cache-job key 'speech'")
        except UnrecoverableError as e:
            success = False
            message = f"Job arguments are invalid: {e}"
        else:
            success = True

        # create cache folder
        if success and job['speech'] is not None:
            if not os.path.exists(job['folder']):
                self._logger.debug(f"Creating cache directory '{job['folder']}'.")
                try:
                    os.makedirs(job['folder'])
                except Exception as e:
                    success = False
                    message = f"Failed to create directory '{job['folder']}': {repr(e)}"
            elif not os.path.isdir(job['folder']):
                success = False
                message = f"Expected path '{job['folder']}' to either not exist or be a directory."

        # write speech file
        if success and isinstance(job['speech'], bytes):
            speech_path = os.path.join(job['folder'], job['file'])
            try:
                with open(speech_path, mode='bw') as f:
                    f.write(job['speech'])
            except Exception as e:
                success = False
                message = f"Failed to write generated speech to file '{speech_path}': {repr(e)}"

        # acquire file lock for cache-index
        if success:
            index_path = os.path.join(job['folder'], job['index'])
            lock_path = f"{index_path}.lock"
            self._logger.debug(f"Locking cache-index with lock-file '{lock_path}'.")
            lock_resource = acquire_lock(path=lock_path)
            is_locked = True
        else:
            is_locked = False

        # open cache-index
        if success:
            if not os.path.exists(index_path):
                self._logger.debug(f"Cache-index '{index_path}' doesn't exist.")
                cache = {}
            else:
                success, message, cache = read_json(file_path=index_path, name="cache-index", logger=self._logger)
                if success:
                    self._logger.debug(message)

        # add data to cache-index
        if success:
            try:
                assert_type_value(obj=cache, type_or_value=dict, name="cache-index'")

                if job['endpoint'] in cache:
                    assert_type_value(obj=cache[job['endpoint']], type_or_value=dict, name=f"value of cache-index key '{job['endpoint']}'")
                else:
                    cache[job['endpoint']] = {}

                if job['model'] in cache[job['endpoint']]:
                    assert_type_value(obj=cache[job['endpoint']][job['model']], type_or_value=dict, name=f"value of cache-index key '{job['model']}'")
                else:
                    cache[job['endpoint']][job['model']] = {}

                if job['voice'] in cache[job['endpoint']][job['model']]:
                    assert_type_value(obj=cache[job['endpoint']][job['model']][job['voice']], type_or_value=dict, name=f"value of cache-index key '{job['voice']}'")
                else:
                    cache[job['endpoint']][job['model']][job['voice']] = {}

                job['speed'] = str(job['speed'])

                if job['speed'] in cache[job['endpoint']][job['model']][job['voice']]:
                    assert_type_value(obj=cache[job['endpoint']][job['model']][job['voice']][job['speed']], type_or_value=dict, name=f"value of cache-index key '{job['speed']}'")
                else:
                    cache[job['endpoint']][job['model']][job['voice']][job['speed']] = {}

                if job['instructions'] in cache[job['endpoint']][job['model']][job['voice']][job['speed']]:
                    assert_type_value(obj=cache[job['endpoint']][job['model']][job['voice']][job['speed']][job['instructions']], type_or_value=dict, name=f"value of cache-index key '{job['instructions']}'")
                else:
                    cache[job['endpoint']][job['model']][job['voice']][job['speed']][job['instructions']] = {}

                if job['text'] in cache[job['endpoint']][job['model']][job['voice']][job['speed']][job['instructions']]:
                    assert_type_value(obj=cache[job['endpoint']][job['model']][job['voice']][job['speed']][job['instructions']][job['text']], type_or_value=str, name=f"value of cache-index key '{job['text']}'")
                    now = cache[job['endpoint']][job['model']][job['voice']][job['speed']][job['instructions']][job['text']]
                    if now == job['file']:
                        self._logger.debug(f"Aborting redundant cache-job for file '{job['file']}'.")
                        release_lock(lock_resource)
                        return
                    else:
                        self._logger.warn(f"Updating entry in cache-index '{index_path}' from '{now}' to '{job['file']}'.")
                else:
                    cache[job['endpoint']][job['model']][job['voice']][job['speed']][job['instructions']][job['text']] = {}

                cache[job['endpoint']][job['model']][job['voice']][job['speed']][job['instructions']][job['text']] = job['file']
            except UnrecoverableError as e:
                success = False
                message = f"Cache-index '{index_path}' is corrupted: {e}"

        # write temporary cache-index
        if success:
            temp_folder = os.path.join(job['folder'], "temp")
            temp_index_path = os.path.join(temp_folder, job['index'])
            success, message = write_json(file_path=temp_index_path, json_object=cache, indent=True, name="temporary cache-index", logger=self._logger)
            if success:
                self._logger.debug(message)

        # replace real cache-index
        if success:
            self._logger.debug(f"Replacing cache-index '{index_path}' with temporary cache-index '{temp_index_path}'.")
            try:
                os.replace(src=temp_index_path, dst=index_path)
            except Exception as e:
                success = False
                message = f"Failed to replace cache-index '{index_path}' with temporary cache-index '{temp_index_path}': {repr(e)}"

        # delete temporary cache-index
        if success:
            if os.path.isfile(temp_index_path):
                self._logger.debug(f"Deleting temporary cache-index '{temp_index_path}'.")
                try:
                    os.remove(path=temp_index_path)
                except Exception as e:
                    success = False
                    message = f"Failed to delete temporary cache-index '{temp_index_path}': {repr(e)}"

        # delete temporary folder
        if success:
            if os.path.isdir(temp_folder):
                self._logger.debug(f"Deleting temporary folder '{temp_folder}'.")
                try:
                    os.rmdir(path=temp_folder)
                except Exception as e:
                    success = False
                    message = f"Failed to delete temporary folder '{temp_folder}': {repr(e)}"

        # release file lock for cache-index
        if is_locked:
            self._logger.debug(f"Releasing lock-file '{lock_path}' for cache-index '{index_path}'.")
            release_lock(resource=lock_resource)

        # log
        if success:
            self._logger.debug("Finished cache-job successfully.")
        else:
            self._logger.warn(f"Failed cache-job: {message}")

    def get_speech(self, text):
        stamp = time.perf_counter()

        # parse arguments
        assert_type_value(obj=text, type_or_value=str, name="argument 'text'")
        assert_log(expression=len(text) > 0, message="Expected argument 'text' to be a non-empty text.")

        # retrieve speech from cache
        speech = self.query_speech(text=text)
        if speech is not None:
            return True, f"Found cached speech for text: '{text}'", speech

        # retrieve API key
        api_key = self.get_api_key()[2]

        # validate connection
        success, message = self.validate_connection(api_key=api_key)
        if not success:
            return False, message, None

        # construct payload
        headers = {
            'Content-Type': "application/json",
            'Authorization': f"Bearer {api_key}"
        }
        data = {
            'input': text,
            'model': self._settings['model'],
            'voice': self._settings['voice'],
            'speed': self._settings['speed'],
            'response_format': "wav"
        }
        if self._settings['instructions'] != "":
            data['instructions'] = self._settings['instructions']

        # use API
        success, message, response = post_request(
            api_name="Speech API",
            api_url=self._endpoint['api_url'],
            headers=headers,
            data=data,
            files=None,
            timeout=(self._settings['timeout_connect'], self._settings['timeout_read']),
            logger=self._logger
        )

        # parse response
        if success:
            try:
                response.json()
            except Exception:
                speech_bytes = response.content
            else:
                success = False

        # generate file name and path
        if success:
            if self._settings['cache_write'] or self._settings['return_path']:
                date = datetime.datetime.now().strftime("%Y_%m_%d_T_%H_%M_%S_%f")
                suffix = ''.join(random.choices(string.ascii_letters, k=4))
                speech_file = f"{date}_{suffix}.wav"
                speech_path = os.path.join(self._settings['cache_folder'], speech_file)

        # convert to target encoding
        if success:
            if self._settings['return_path']:
                # create cache folder
                if not os.path.exists(self._settings['cache_folder']):
                    self._logger.debug(f"Creating directory '{self._settings['cache_folder']}'.")
                    try:
                        os.makedirs(self._settings['cache_folder'])
                    except Exception as e:
                        success = False
                        message = f"Failed to create directory '{self._settings['cache_folder']}': {repr(e)}"
                elif not os.path.isdir(self._settings['cache_folder']):
                    success = False
                    message = f"Expected path '{self._settings['cache_folder']}' to either not exist or be a directory."
                # save file
                if success:
                    self._logger.debug(f"Writing generated speech to file '{speech_path}'.")
                    try:
                        with open(speech_path, mode='bw') as f:
                            f.write(speech_bytes)
                    except Exception as e:
                        success = False
                        message = f"Failed to write generated speech to file '{speech_path}': {repr(e)}"
                    else:
                        speech = speech_path
            elif self._settings['return_encoding'] == "bytes":
                speech = speech_bytes
            elif self._settings['return_encoding'] == "base64":
                success, _message, speech = encode_b64(obj=speech_bytes, name="speech", logger=self._logger)
                if not success:
                    message = _message
            else:
                raise NotImplementedError(f"Unknown encoding '{self._settings['return_encoding']}'.")

        if success:
            message = f"Generated speech in '{time.perf_counter() - stamp:.3f}s': '{text}'"
            if self._settings['cache_write']:
                # cache generated speech
                cache_job = {
                    'type': "speech",
                    'folder': self._settings['cache_folder'],
                    'index': self._settings['cache_file'],
                    'file': speech_file,
                    'endpoint': self._settings['endpoint'],
                    'model': self._settings['model'],
                    'voice': self._settings['voice'],
                    'instructions': self._settings['instructions'],
                    'speed': self._settings['speed'],
                    'text': text,
                    'speech': None if self._settings['return_path'] else speech_bytes
                }
                nimbro_api.register_deferred_job(job=(self.cache_speech, cache_job), mute=True)
        else:
            speech = None

        return success, message, speech
