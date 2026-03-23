import os
import time
import random
import string
import datetime

import nimbro_api
from nimbro_api.client import ClientBase
from nimbro_api.utility.io import read_json, write_json, encode_b64, decode_b64, get_cache_location, acquire_lock, release_lock
from nimbro_api.utility.api import get_api_key, validate_endpoint, post_request
from nimbro_api.utility.misc import UnrecoverableError, assert_type_value, assert_keys, assert_log
from ..utility import validate_connection, get_models

class ImagesBase(ClientBase):

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
        assert_type_value(obj=settings['model'], type_or_value=["gpt-image-1", "dall-e-3", "dall-e-2"], name="setting 'model'")

        # validate_model
        assert_type_value(obj=settings['validate_model'], type_or_value=[float, int, bool], name="setting 'validate_model'")
        if isinstance(settings['validate_model'], (int, float)):
            assert_log(
                expression=settings['validate_model'] >= 0,
                message=f"Expected setting 'validate_model' provided as '{type(settings['validate_model']).__name__}' to be non-negative but got '{settings['validate_model']}'."
            )

        # quality, style, size
        if settings['model'] == "gpt-image-1":
            assert_type_value(obj=settings['quality'], type_or_value=["auto", "high", "medium", "low"], name=f"setting 'quality' for model '{settings['model']}'")
            assert_type_value(obj=settings['style'], type_or_value="", name=f"setting 'style' for model '{settings['model']}'")
            assert_type_value(obj=settings['size'], type_or_value=["auto", "1024x1024", "1536x1024", "1024x1536"], name=f"setting 'size' for model '{settings['model']}'")
        elif settings['model'] == "dall-e-3":
            assert_type_value(obj=settings['quality'], type_or_value=["auto", "hd", "standard"], name=f"setting 'quality' for model '{settings['model']}'")
            assert_type_value(obj=settings['style'], type_or_value=["vivid", "natural"], name=f"setting 'style' for model '{settings['model']}'")
            assert_type_value(obj=settings['size'], type_or_value=["1024x1024", "1792x1024", "1024x1792"], name=f"setting 'size' for model '{settings['model']}'")
        elif settings['model'] == "dall-e-2":
            assert_type_value(obj=settings['quality'], type_or_value="", name=f"setting 'quality' for model '{settings['model']}'")
            assert_type_value(obj=settings['style'], type_or_value="", name=f"setting 'style' for model '{settings['model']}'")
            assert_type_value(obj=settings['size'], type_or_value=["256x256", "512x512", "1024x1024"], name=f"setting 'size' for model '{settings['model']}'")
        else:
            raise NotImplementedError(f"Unknown model name '{self._settings['model']}'.")

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
            settings['cache_folder'] = os.path.join(get_cache_location(), "images")
            self._logger.debug(f"Using cache-folder '{settings['cache_folder']}'.")

        # cache_file
        assert_type_value(obj=settings['cache_file'], type_or_value=str, name="setting 'cache_file'")

        # return_path
        assert_type_value(obj=settings['return_path'], type_or_value=bool, name="setting 'return_path'")

        # return_encoding
        assert_type_value(obj=settings['return_encoding'], type_or_value=["bytes", "base64"], name="setting 'return_encoding'")

        # cache_read
        assert_type_value(obj=settings['cache_read'], type_or_value=[bool, float, int], name="setting 'cache_read'")
        if isinstance(settings['cache_read'], (int, float)):
            assert_log(
                expression=settings['cache_read'] >= 0,
                message=f"Expected setting 'cache_read' provided as '{type(settings['cache_read']).__name__}' to be non-negative but got '{settings['cache_read']}'."
            )

        # cache_write
        assert_type_value(obj=settings['cache_write'], type_or_value=bool, name="setting 'cache_write'")

        # apply settings
        self._endpoint = settings['endpoints'][settings['endpoint']]
        return self._apply_settings(settings, mode)

    def query_image(self, prompt):
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
                    category="images",
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
                        success, message = nimbro_api.update_cache(category="images", identifier=cache_path, data=index, mute=True)
                        if success:
                            self._logger.debug(message)
                        else:
                            self._logger.warn(message)
                    else:
                        self._logger.warn(message)

            if index is not None:
                # check required image is cached
                image_file = None
                try:
                    assert_type_value(obj=index, type_or_value=dict, name="cache-index")
                    if self._settings['endpoint'] in index:
                        assert_type_value(obj=index, type_or_value=dict, name="cache-index")
                        if self._settings['model'] in index[self._settings['endpoint']]:
                            assert_type_value(obj=index[self._settings['endpoint']][self._settings['model']], type_or_value=dict, name=f"value of cache-index key '{self._settings['model']}'")
                            if self._settings['quality'] in index[self._settings['endpoint']][self._settings['model']]:
                                assert_type_value(obj=index[self._settings['endpoint']][self._settings['model']][self._settings['quality']], type_or_value=dict, name=f"value of cache-index key '{self._settings['quality']}'")
                                if self._settings['style'] in index[self._settings['endpoint']][self._settings['model']][self._settings['quality']]:
                                    assert_type_value(obj=index[self._settings['endpoint']][self._settings['model']][self._settings['quality']][self._settings['style']], type_or_value=dict, name=f"value of cache-index key '{self._settings['style']}'")
                                    if self._settings['size'] in index[self._settings['endpoint']][self._settings['model']][self._settings['quality']][self._settings['style']]:
                                        assert_type_value(obj=index[self._settings['endpoint']][self._settings['model']][self._settings['quality']][self._settings['style']][self._settings['size']], type_or_value=dict, name=f"value of cache-index key '{self._settings['size']}'")
                                        if prompt in index[self._settings['endpoint']][self._settings['model']][self._settings['quality']][self._settings['style']][self._settings['size']]:
                                            assert_type_value(obj=index[self._settings['endpoint']][self._settings['model']][self._settings['quality']][self._settings['style']][self._settings['size']][prompt], type_or_value=str, name=f"value of cache-index key '{prompt}'")
                                            image_file = index[self._settings['endpoint']][self._settings['model']][self._settings['quality']][self._settings['style']][self._settings['size']][prompt]
                except UnrecoverableError as e:
                    self._logger.warn(f"Cache-index '{cache_path}' is corrupted: {e}")
                else:
                    if image_file is None:
                        self._logger.debug(f"Image not found in cache-index '{cache_path}'.")
                    else:
                        self._logger.debug(f"Found required image in cache-index '{cache_path}'.")
                        # check referenced file exists
                        image_path = os.path.join(self._settings['cache_folder'], image_file)
                        if self._settings['return_path'] is True:
                            if os.path.isfile(image_path):
                                return image_path
                            else:
                                self._logger.warn(f"Cache-index '{cache_path}' references image-file '{image_path}' that does not exist.")
                        else:
                            # open file
                            self._logger.debug(f"Reading referenced image-file '{image_path}'.")
                            try:
                                with open(image_path, 'rb') as f:
                                    image_bytes = f.read()
                            except Exception as e:
                                self._logger.warn(f"Failed to read image-file '{image_path}' referenced in cache-index '{cache_path}': {repr(e)}")
                            else:
                                if self._settings['return_encoding'] == "bytes":
                                    return image_bytes
                                elif self._settings['return_encoding'] == "base64":
                                    success, message, image_b64 = encode_b64(obj=image_bytes, name=f"image-file '{image_path}' referenced in cache-index '{cache_path}'", logger=self._logger)
                                    if success:
                                        self._logger.debug(message)
                                        return image_b64
                                    else:
                                        self._logger.warn(message)
                                else:
                                    raise NotImplementedError(f"Unknown encoding '{self._settings['return_encoding']}'.")

    def cache_image(self, job):
        # validate job
        try:
            assert_keys(obj=job, keys=['type', 'folder', 'index', 'file', 'endpoint', 'model', 'quality', 'style', 'size', 'prompt', 'image'], mode="match", name="job")
            assert_type_value(obj=job['folder'], type_or_value=str, name="cache-job key 'folder'")
            assert_type_value(obj=job['index'], type_or_value=str, name="cache-job key 'index'")
            assert_type_value(obj=job['file'], type_or_value=str, name="cache-job key 'file'")
            assert_type_value(obj=job['endpoint'], type_or_value=str, name="cache-job key 'endpoint'")
            assert_type_value(obj=job['model'], type_or_value=str, name="cache-job key 'model'")
            assert_type_value(obj=job['quality'], type_or_value=str, name="cache-job key 'quality'")
            assert_type_value(obj=job['style'], type_or_value=str, name="cache-job key 'style'")
            assert_type_value(obj=job['size'], type_or_value=str, name="cache-job key 'size'")
            assert_type_value(obj=job['prompt'], type_or_value=str, name="cache-job key 'prompt'")
            assert_type_value(obj=job['image'], type_or_value=[str, bytes, None], name="cache-job key 'image'")
        except UnrecoverableError as e:
            success = False
            message = f"Job arguments are invalid: {e}"
        else:
            success = True

        # create cache folder
        if success and job['image'] is not None:
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

        # decode base64 to bytes
        if success and isinstance(job['image'], str):
            success, message, job['image'] = decode_b64(string=job['image'], name="image", logger=self._logger)
            if success:
                self._logger.debug(message)

        # write image file
        if success and job['image'] is not None:
            image_path = os.path.join(job['folder'], job['file'])
            try:
                with open(image_path, mode='bw') as f:
                    f.write(job['image'])
            except Exception as e:
                success = False
                message = f"Failed to write generated image to file '{image_path}': {repr(e)}"

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

                if job['quality'] in cache[job['endpoint']][job['model']]:
                    assert_type_value(obj=cache[job['endpoint']][job['model']][job['quality']], type_or_value=dict, name=f"value of cache-index key '{job['quality']}'")
                else:
                    cache[job['endpoint']][job['model']][job['quality']] = {}

                if job['style'] in cache[job['endpoint']][job['model']][job['quality']]:
                    assert_type_value(obj=cache[job['endpoint']][job['model']][job['quality']][job['style']], type_or_value=dict, name=f"value of cache-index key '{job['style']}'")
                else:
                    cache[job['endpoint']][job['model']][job['quality']][job['style']] = {}

                if job['size'] in cache[job['endpoint']][job['model']][job['quality']][job['style']]:
                    assert_type_value(obj=cache[job['endpoint']][job['model']][job['quality']][job['style']][job['size']], type_or_value=dict, name=f"value of cache-index key '{job['size']}'")
                else:
                    cache[job['endpoint']][job['model']][job['quality']][job['style']][job['size']] = {}

                if job['prompt'] in cache[job['endpoint']][job['model']][job['quality']][job['style']][job['size']]:
                    assert_type_value(obj=cache[job['endpoint']][job['model']][job['quality']][job['style']][job['size']][job['prompt']], type_or_value=str, name=f"value of cache-index key '{job['prompt']}'")
                    now = cache[job['endpoint']][job['model']][job['quality']][job['style']][job['size']][job['prompt']]
                    if now == job['file']:
                        self._logger.debug(f"Aborting redundant cache-job for file '{job['file']}'.")
                        release_lock(lock_resource)
                        return
                    else:
                        self._logger.warn(f"Updating entry in cache-index '{index_path}' from '{now}' to '{job['file']}'.")
                else:
                    cache[job['endpoint']][job['model']][job['quality']][job['style']][job['size']][job['prompt']] = {}

                cache[job['endpoint']][job['model']][job['quality']][job['style']][job['size']][job['prompt']] = job['file']
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

    def get_image(self, prompt):
        stamp = time.perf_counter()

        # parse arguments
        assert_type_value(obj=prompt, type_or_value=str, name="argument 'prompt'")
        assert_log(expression=len(prompt) > 0, message="Expected argument 'prompt' to be a non-empty prompt.")

        # retrieve image from cache
        image = self.query_image(prompt=prompt)
        if image is not None:
            return True, f"Found cached image for prompt: '{prompt}'", image

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
            'prompt': prompt,
            'model': self._settings['model'],
            'size': self._settings['size'],
            'n': 1
        }
        if self._settings['model'] == "gpt-image-1":
            data['quality'] = self._settings['quality']
            data['background'] = "auto"
            data['moderation'] = "low"
            data['output_format'] = "png"
            data['stream'] = False
        elif self._settings['model'] == "dall-e-3":
            data['quality'] = self._settings['quality']
            data['style'] = self._settings['style']
            data['response_format'] = "b64_json"
        elif self._settings['model'] == "dall-e-2":
            data['response_format'] = "b64_json"
        else:
            raise NotImplementedError(f"Unknown model name '{self._settings['model']}'.")

        # use API
        success, message, response = post_request(
            api_name="Images API",
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
                response = response.json()
                # self._logger.warn(f"Keys: {response.keys()}")                                  # [2025-11-17T18:08:57.624515]: Keys: dict_keys(['created', 'data'])
                # self._logger.warn(f"Created: {response['created']}")                           # [2025-11-17T18:08:57.624515]: Created: 1763399336
                # self._logger.warn(f"type(response['data']): {type(response['data'])}")         # [2025-11-17T18:08:57.624515]: type(response['data']): <class 'list'>
                # self._logger.warn(f"len(response['data']): {len(response['data'])}")           # [2025-11-17T18:08:57.624515]: len(response['data']): 1
                # self._logger.warn(f"type(response['data'][0]): {type(response['data'][0])}")   # [2025-11-17T18:08:57.624515]: type(response['data'][0]): <class 'dict'>
                # self._logger.warn(f"response['data'][0].keys(): {response['data'][0].keys()}") # [2025-11-17T18:08:57.624515]: response['data'][0].keys(): dict_keys(['b64_json'])
                image_b64 = response['data'][0]['b64_json']
            except Exception as e:
                success = False
                message = f"Failed to parse response from Images API '{self._endpoint['api_url']}': {repr(e)}"

        # generate file name and path
        if success:
            if self._settings['cache_write'] or self._settings['return_path']:
                date = datetime.datetime.now().strftime("%Y_%m_%d_T_%H_%M_%S_%f")
                suffix = ''.join(random.choices(string.ascii_letters, k=4))
                image_file = f"{date}_{suffix}.png"
                image_path = os.path.join(self._settings['cache_folder'], image_file)

        # convert to target encoding
        if success:
            if self._settings['return_path'] or self._settings['return_encoding'] == "bytes":
                success, _message, image_bytes = decode_b64(string=image_b64, name="image", logger=self._logger) # TODO mute error
                if not success:
                    message = _message
                else:
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
                            self._logger.debug(f"Writing generated image to file '{image_path}'.")
                            try:
                                with open(image_path, mode='bw') as f:
                                    f.write(image_bytes)
                            except Exception as e:
                                success = False
                                message = f"Failed to write generated image to file '{image_path}': {repr(e)}"
                            else:
                                image = image_path
                    elif self._settings['return_encoding'] == "bytes":
                        image = image_bytes
                    else:
                        raise NotImplementedError(f"Unknown encoding '{self._settings['return_encoding']}'.")
            elif self._settings['return_encoding'] == "base64":
                image = image_b64
            else:
                raise NotImplementedError(f"Unknown encoding '{self._settings['return_encoding']}'.")

        if success:
            message = f"Generated image in '{time.perf_counter() - stamp:.3f}s': '{prompt}'"
            if self._settings['cache_write']:
                # cache generated image
                cache_job = {
                    'type': "images",
                    'folder': self._settings['cache_folder'],
                    'index': self._settings['cache_file'],
                    'file': image_file,
                    'endpoint': self._settings['endpoint'],
                    'model': self._settings['model'],
                    'quality': self._settings['quality'],
                    'style': self._settings['style'],
                    'size': self._settings['size'],
                    'prompt': prompt,
                    'image': None if self._settings['return_path'] else image
                }
                nimbro_api.register_deferred_job(job=(self.cache_image, cache_job), mute=True)
        else:
            image = None

        return success, message, image
