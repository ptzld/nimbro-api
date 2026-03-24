import os
import copy
import time
import datetime

import nimbro_api
from nimbro_api.client import ClientBase
from nimbro_api.utility.io import read_json, write_json, get_cache_location, acquire_lock, release_lock
from nimbro_api.utility.api import get_api_key, validate_endpoint, post_request
from nimbro_api.utility.misc import UnrecoverableError, assert_type_value, assert_keys, assert_log, format_obj
from ..utility import validate_connection, get_models

class EmbeddingsBase(ClientBase):

    def __init__(self, settings, default_settings, **kwargs):
        super().__init__(settings=settings, default_settings=default_settings, **kwargs)
        self.get_api_key = get_api_key.__get__(self)
        self.validate_connection = validate_connection.__get__(self)
        self.get_models = get_models.__get__(self)
        self._logger.debug(f"Initialized '{type(self).__name__}' object.")
        self._initialized = True

    def set_settings(self, settings, mode="set"):
        settings = self._introduce_settings(settings=settings, mode=mode)

        # logger_info_requests
        assert_type_value(obj=settings['logger_info_requests'], type_or_value=bool, name="setting 'logger_info_requests'")

        # logger_info_progress
        assert_type_value(obj=settings['logger_info_progress'], type_or_value=bool, name="setting 'logger_info_progress'")

        # logger_info_cutoff
        assert_type_value(obj=settings['logger_info_cutoff'], type_or_value=int, name="setting 'logger_info_cutoff'")
        assert_log(expression=settings['logger_info_cutoff'] > 0, message=f"Expected setting 'logger_info_cutoff' to be greater zero but got '{settings['logger_info_cutoff']}'.")

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

        # max_batch_size
        assert_type_value(obj=settings['max_batch_size'], type_or_value=[int, None], name="setting 'max_batch_size'")
        assert_log(
            expression=settings['max_batch_size'] > 0,
            message=f"Expected setting 'max_batch_size' to be greater zero but got '{settings['max_batch_size']}'."
        )

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
            settings['cache_folder'] = os.path.join(get_cache_location(), "embeddings")
            self._logger.debug(f"Using cache-folder '{settings['cache_folder']}'.")

        # cache_file
        assert_type_value(obj=settings['cache_file'], type_or_value=str, name="setting 'cache_file'")
        assert_log(expression="index" in settings['cache_file'], message=f"Expected setting 'cache_file' to contain the sub-string 'index' but got '{settings['cache_file']}'.")

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

    def query_embeddings(self, texts):
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
                    category="embeddings",
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
                        success, message = nimbro_api.update_cache(category="embeddings", identifier=cache_path, data=index, mute=True)
                        if success:
                            self._logger.debug(message)
                        else:
                            self._logger.warn(message)
                    else:
                        self._logger.warn(message)

            if index is not None:
                # check required embeddings are cached
                num_texts = len(texts)
                sub_jobs = {}
                found = 0
                try:
                    assert_type_value(obj=index, type_or_value=dict, name="cache-index")
                    if 'texts' in index:
                        assert_type_value(obj=index['texts'], type_or_value=dict, name="value of cache-index key 'texts'")
                        if 'files' in index:
                            assert_type_value(obj=index['files'], type_or_value=list, name="value of cache-index key 'files'")
                            if self._settings['endpoint'] in index['texts']:
                                assert_type_value(obj=index['texts'], type_or_value=dict, name=f"value of cache-index key 'texts'>'{self._settings['endpoint']}'")
                                if self._settings['model'] in index['texts'][self._settings['endpoint']]:
                                    assert_type_value(obj=index['texts'][self._settings['endpoint']][self._settings['model']], type_or_value=dict, name=f"value of cache-index key 'texts'>'{self._settings['endpoint']}'>'{self._settings['model']}'")
                                    for i, text in enumerate(texts):
                                        if text in index['texts'][self._settings['endpoint']][self._settings['model']]:
                                            assert_type_value(obj=index['texts'][self._settings['endpoint']][self._settings['model']][text], type_or_value=list, name=f"value of cache-index key 'texts'>'{self._settings['endpoint']}'>'{self._settings['model']}'>'{text}'")
                                            file_and_idx = copy.deepcopy(index['texts'][self._settings['endpoint']][self._settings['model']][text])
                                            assert_log(expression=len(index['files']) > file_and_idx[0], message=f"Expected value of cache-index key 'files' be list that contains at least '{file_and_idx[0] + 1}' element{'' if file_and_idx[0] + 1 == 1 else 's'} but got '{len(index['files'])}'.")
                                            assert_type_value(obj=index['files'][file_and_idx[0]], type_or_value=list, name=f"value of element '{file_and_idx[0]}' in value of cache-index key '{text}'")
                                            assert_log(expression=len(index['files'][file_and_idx[0]]) == 2, message=f"Expected value of element '{file_and_idx[0]}' in value of cache-index key '{text}' to be a list of length '2' but got '{len(index['files'][file_and_idx[0]])}'.")
                                            assert_type_value(obj=index['files'][file_and_idx[0]][0], type_or_value=str, name=f"first element in element '{file_and_idx[0]}' in value of cache-index key '{text}'")
                                            assert_type_value(obj=index['files'][file_and_idx[0]][1], type_or_value=int, name=f"second element in element '{file_and_idx[0]}' in value of cache-index key '{text}'")
                                            assert_log(expression=index['files'][file_and_idx[0]][1] > file_and_idx[1], message=f"Expected value of second element in element '{file_and_idx[0]}' in value of cache-index key '{text}' to be an int larger '{file_and_idx[1]}' but got '{index['files'][file_and_idx[0]][1]}'.")
                                            file_and_idx[0] = index['files'][file_and_idx[0]][0]
                                            sub_job = (i, file_and_idx[1])
                                            if file_and_idx[0] not in sub_jobs:
                                                sub_jobs[file_and_idx[0]] = []
                                            sub_jobs[file_and_idx[0]].append(sub_job)
                                            found += 1
                except UnrecoverableError as e:
                    self._logger.warn(f"Cache-index '{cache_path}' is corrupted: {e}")
                else:
                    if len(sub_jobs) == 0:
                        self._logger.debug(f"Text{'' if num_texts == 1 else 's'} not referenced in cache-index '{cache_path}'.")
                    else:
                        self._logger.debug(f"Found '{found}' of '{num_texts}' text{'' if num_texts == 1 else 's'} referenced in cache-index '{cache_path}'.")
                        embeddings = [None] * num_texts
                        for file in sub_jobs:
                            # check referenced file exists
                            file_path = os.path.join(self._settings['cache_folder'], file)
                            if not os.path.isfile(file_path):
                                self._logger.warn(f"Cache-index '{cache_path}' references embeddings-file '{file_path}' that does not exist.")
                            else:
                                # open file
                                success, message, file_data = read_json(file_path=file_path, name="referenced embeddings-file", logger=self._logger)
                                if success:
                                    self._logger.debug(message)
                                    # obtain embeddings required from file
                                    obtained = 0
                                    try:
                                        assert_type_value(obj=file_data, type_or_value=list, name="embeddings-file")
                                        for sub_job in sub_jobs[file]:
                                            assert_log(expression=len(file_data) > sub_job[1], message=f"Expected embeddings-file '{file_path}' to contain at least '{sub_job[1]}' element{'' if sub_job[1] == 1 else 's'} but got '{len(file_data)}'.")
                                            embeddings[sub_job[0]] = file_data[sub_job[1]]
                                            obtained += 1
                                    except UnrecoverableError as e:
                                        self._logger.warn(f"Embeddings-file '{file_path}' is corrupted: {e}")
                                    self._logger.debug(f"Obtained '{obtained}' embedding{'' if obtained == 1 else 's'} from embeddings-file '{file_path}'.")

                                else:
                                    self._logger.warn(f"Failed to open embeddings-file '{file_path}' referenced in cache-index '{cache_path}': {message}")
                        obtained = num_texts - embeddings.count(None)
                        if obtained > 0:
                            self._logger.debug(f"Obtained '{obtained}' of '{num_texts}' required embedding{'' if num_texts == 1 else 's'}.")
                            return embeddings
        return [None] * len(texts)

    def cache_embeddings(self, job):
        # validate job
        try:
            assert_keys(obj=job, keys=['type', 'folder', 'index', 'endpoint', 'model', 'texts', 'embeddings'], mode="match", name="job")
            assert_type_value(obj=job['folder'], type_or_value=str, name="cache-job key 'folder'")
            assert_type_value(obj=job['index'], type_or_value=str, name="cache-job key 'index'")
            assert_log(expression="index" in job['index'], message=f"Expected value of cache-job key 'index' to contain the sub-string 'index' but got '{job['index']}'.")
            assert_type_value(obj=job['endpoint'], type_or_value=str, name="cache-job key 'endpoint'")
            assert_type_value(obj=job['model'], type_or_value=str, name="cache-job key 'model'")
            assert_type_value(obj=job['texts'], type_or_value=list, name="cache-job key 'texts'")
            for i, item in enumerate(job['texts']):
                assert_type_value(obj=item, type_or_value=str, name=f"element '{i}' in value of cache-job key 'texts'")
            assert_type_value(obj=job['embeddings'], type_or_value=list, name="cache-job key 'embeddings'")
            for i, item in enumerate(job['embeddings']):
                assert_type_value(obj=item, type_or_value=list, name=f"element '{i}' in value of cache-job key 'embeddings'")
                for j, sub_item in enumerate(item):
                    assert_type_value(obj=sub_item, type_or_value=float, name=f"element '{i}' in element '{j}' in value of cache-job key 'embeddings'")
            assert_log(expression=len(job['texts']) == len(job['embeddings']), message=f"Expected values of cache-job keys 'texts' and 'embeddings' be list of equal length but got '{len(job['texts'])}' and '{len(job['embeddings'])}'.")
            assert_log(expression=len(job['texts']) > 0, message="Expected values of cache-job keys 'texts' and 'embeddings' to be non-empty lists.")
        except UnrecoverableError as e:
            success = False
            message = f"Job arguments are invalid: {e}"
        else:
            success = True

        # create cache folder
        if success:
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
            max_embeddings_per_file = 200

            try:
                assert_type_value(obj=cache, type_or_value=dict, name="cache-index'")

                # extend incomplete structure of cache-index
                if 'texts' in cache:
                    assert_type_value(obj=cache['texts'], type_or_value=dict, name="value of cache-index key 'texts'")
                else:
                    cache['texts'] = {}
                if job['endpoint'] in cache['texts']:
                    assert_type_value(obj=cache['texts'][job['endpoint']], type_or_value=dict, name=f"value of cache-index key 'texts.{job['endpoint']}'")
                else:
                    cache['texts'][job['endpoint']] = {}
                if job['model'] in cache['texts'][job['endpoint']]:
                    assert_type_value(obj=cache['texts'][job['endpoint']][job['model']], type_or_value=dict, name=f"value of cache-index key 'texts.{job['endpoint']}.{job['model']}'")
                else:
                    cache['texts'][job['endpoint']][job['model']] = {}
                if 'files' in cache:
                    assert_type_value(obj=cache['files'], type_or_value=list, name="value of cache-index key 'files'")
                    for i, item in enumerate(cache['files']):
                        assert_type_value(obj=item, type_or_value=list, name=f"element '{i}' in value of cache-index key 'files'")
                        assert_log(expression=len(item) == 2, message=f"Expected element '{i}' in value of cache-index key 'files' to be list of length '2' but got '{len(item)}'.")
                        assert_type_value(obj=item[0], type_or_value=str, name=f"element '{i}.0' in value of cache-index key 'files'")
                        assert_type_value(obj=item[1], type_or_value=int, name=f"element '{i}.1' in value of cache-index key 'files'")
                else:
                    cache['files'] = []

                jobs = {}
                new_files = []
                for text, embedding in zip(job['texts'], job['embeddings']):
                    if text in cache['texts'][job['endpoint']][job['model']]:
                        assert_type_value(obj=cache['texts'][job['endpoint']][job['model']][text], type_or_value=list, name=f"value of cache-index key \"texts.{job['endpoint']}.{job['model']}.'{text}'\"")
                        assert_log(expression=len(cache['texts'][job['endpoint']][job['model']][text]) == 2, message=f"Expected value of cache-index key \"texts.{job['endpoint']}.{job['model']}.'{text}'\" to be a list of length '2' but got '{len(cache['texts'][job['endpoint']][job['model']][text])}'.")
                        assert_type_value(obj=cache['texts'][job['endpoint']][job['model']][text][0], type_or_value=int, name=f"value of cache-index key \"texts.{job['endpoint']}.{job['model']}.'{text}'.0\"")
                        assert_log(expression=cache['texts'][job['endpoint']][job['model']][text][0] >= 0, message=f"Expected value of cache-index key \"texts.{job['endpoint']}.{job['model']}.'{text}'.0\" to be non-negative but got '{cache['texts'][job['endpoint']][job['model']][text][0]}'.")
                        assert_type_value(obj=cache['texts'][job['endpoint']][job['model']][text][1], type_or_value=int, name=f"value of cache-index key \"texts.{job['endpoint']}.{job['model']}.'{text}'.1\"")
                        assert_log(expression=cache['texts'][job['endpoint']][job['model']][text][1] >= 0, message=f"Expected value of cache-index key \"texts.{job['endpoint']}.{job['model']}.'{text}'.1\" to be non-negative but got '{cache['texts'][job['endpoint']][job['model']][text][1]}'.")
                        assert_log(expression=len(cache['files']) > cache['texts'][job['endpoint']][job['model']][text][0], message=f"Expected value of cache-index key \"texts.{job['endpoint']}.{job['model']}.'{text}'.0\" to be less than length of cache-index key \"files\" but got '{cache['texts'][job['endpoint']][job['model']][text][0]}'.")
                        assert_log(expression=cache['files'][cache['texts'][job['endpoint']][job['model']][text][0]][1] > cache['texts'][job['endpoint']][job['model']][text][1], message=f"Expected value of cache-index key \"texts.{job['endpoint']}.{job['model']}.'{text}'.1\" to be less than value of cache-index key \"files.{cache['texts'][job['endpoint']][job['model']][text][0]}.1\" but got '{cache['texts'][job['endpoint']][job['model']][text][1]}'.")
                        if cache['files'][cache['texts'][job['endpoint']][job['model']][text][0]][0] not in jobs:
                            jobs[cache['files'][cache['texts'][job['endpoint']][job['model']][text][0]][0]] = []
                        jobs[cache['files'][cache['texts'][job['endpoint']][job['model']][text][0]][0]].append((text, embedding, cache['texts'][job['endpoint']][job['model']][text][1]))
                    else:
                        for i, item in enumerate(cache['files']):
                            assert_type_value(obj=item, type_or_value=list, name=f"element '{i}' in value of cache-index key 'files'")
                            assert_log(expression=len(item) == 2, message=f"Expected element '{i}' in value of cache-index key 'files' to be list of length '2' but got '{len(item)}'.")
                            assert_type_value(obj=item[0], type_or_value=str, name=f"element '{i}.0' in value of cache-index key 'files'")
                            assert_type_value(obj=item[1], type_or_value=int, name=f"element '{i}.1' in value of cache-index key 'files'")
                            if item[1] < max_embeddings_per_file:
                                if item[0] not in jobs:
                                    jobs[item[0]] = []
                                jobs[item[0]].append((text, embedding, item[1]))
                                cache['texts'][job['endpoint']][job['model']][text] = [i, item[1]]
                                cache['files'][i][1] += 1
                                break
                        else:
                            new_name = job['index'].replace("index", f"{len(cache['files'])}")
                            new_files.append(new_name)
                            assert_log(expression=new_name not in cache['files'], message=f"Expected name of new embeddings-file '{new_name}' to not be taken given that cache-index lists only '{len(cache['files'])}' embeddings-files.")
                            jobs[new_name] = [(text, embedding, 0)]
                            cache['files'].append([new_name, 1])
                            cache['texts'][job['endpoint']][job['model']][text] = [len(cache['files']) - 1, 0]
            except UnrecoverableError as e:
                success = False
                message = f"Cache-index '{index_path}' is corrupted: {e}"

        # add embeddings to embeddings-files
        if success:
            data = {}
            for file in jobs:
                if file in new_files:
                    data[file] = []
                else:
                    success, message, data[file] = read_json(file_path=os.path.join(job['folder'], file), name="embeddings-file", logger=self._logger)
                    if success:
                        self._logger.debug(message)
                        try:
                            assert_type_value(obj=data[file], type_or_value=list, name=f"embeddings-file '{file}'")
                        except UnrecoverableError as e:
                            success = False
                            message = f"{e}"
                            break
                    else:
                        break
                if success:
                    for item in jobs[file]:
                        if item[2] == len(data[file]):
                            self._logger.debug(f"Adding embedding of text '{format_obj(item[0])}' to embeddings-file '{file}' at index '{item[2]}'.")
                            data[file].append(item[1])
                        elif item[2] < len(data[file]):
                            if data[file][item[2]] != item[1]:
                                self._logger.warn(f"Updating embedding of text '{format_obj(item[0])}' in embeddings-file '{file}' at index '{item[2]}'.")
                                data[file][item[2]] = item[1]
                            else:
                                self._logger.debug(f"Ignoring redundant update of embedding of text '{format_obj(item[0])}' in embeddings-file '{file}' at index '{item[2]}'.")
                        else:
                            success = False
                            message = f"Expected index '{item[2]}' for embeddings-file '{file}' to be less than or equal to current length '{len(data[file])}' but got '{item[2]}'."
                            break
                if not success:
                    break

        # write temporary embeddings-files
        if success:
            temp_real_embeddings_paths = []
            for file in data:
                temp_folder = os.path.join(job['folder'], "temp")
                temp_embeddings_path = os.path.join(temp_folder, file)
                success, message = write_json(file_path=temp_embeddings_path, json_object=data[file], indent=True, name="temporary embeddings-file", logger=self._logger)
                if success:
                    self._logger.debug(message)
                else:
                    break
                real_real_embeddings_path = os.path.join(job['folder'], file)
                temp_real_embeddings_paths.append((temp_embeddings_path, real_real_embeddings_path))

        # write temporary cache-index
        if success:
            temp_folder = os.path.join(job['folder'], "temp")
            temp_index_path = os.path.join(temp_folder, job['index'])
            success, message = write_json(file_path=temp_index_path, json_object=cache, indent=True, name="temporary cache-index", logger=self._logger)
            if success:
                self._logger.debug(message)

        # replace real embeddings-files
        if success:
            for temp_path, real_path in temp_real_embeddings_paths:
                self._logger.debug(f"Replacing embeddings-file '{real_path}' with temporary embeddings-file '{temp_path}'.")
                try:
                    os.replace(src=temp_path, dst=real_path)
                except Exception as e:
                    success = False
                    message = f"Failed to replace embeddings-file '{real_path}' with temporary embeddings-file '{temp_path}': {repr(e)}"
                    break

        # replace real cache-index
        if success:
            self._logger.debug(f"Replacing cache-index '{index_path}' with temporary cache-index '{temp_index_path}'.")
            try:
                os.replace(src=temp_index_path, dst=index_path)
            except Exception as e:
                success = False
                message = f"Failed to replace cache-index '{index_path}' with temporary cache-index '{temp_index_path}': {repr(e)}"

        # delete temporary embeddings-files
        if success:
            for temp_path, _ in temp_real_embeddings_paths:
                if os.path.isfile(temp_path):
                    self._logger.debug(f"Deleting temporary embeddings-file '{temp_path}'.")
                    try:
                        os.remove(path=temp_path)
                    except Exception as e:
                        success = False
                        message = f"Failed to delete temporary embeddings-file '{temp_path}': {repr(e)}"

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

    def get_embedding(self, text):
        stamp = time.perf_counter()

        # parse text
        assert_type_value(obj=text, type_or_value=[str, list], name="argument 'text'")
        if isinstance(text, str):
            text = text.strip()
            assert_log(expression=len(text) > 0, message="Expected argument 'text' provided as 'str' to be non-empty.")
        else:
            for i, item in enumerate(text):
                assert_type_value(obj=item, type_or_value=str, name="all elements in argument 'text' provided as 'list'")
                text[i] = item.strip()
                assert_log(expression=len(item) > 0, message="Expected all elements in argument 'text' provided as 'list' to be non-empty strings.")

        if isinstance(text, str):
            input_is_list = False
            texts = [text]
        elif isinstance(text, list):
            input_is_list = True
            texts = text
        else:
            raise NotImplementedError(f"Unsupported 'text' type '{type(text).__name__}'.")

        # retrieve embeddings from cache
        embeddings = self.query_embeddings(texts=texts)
        num_cached = len(texts) - embeddings.count(None)
        if num_cached == len(texts):
            if input_is_list:
                if self._settings['logger_info_cutoff'] == 0:
                    suffix = "."
                else:
                    suffix = f"{texts}"
                    suffix = f": {suffix[:self._settings['logger_info_cutoff']]}...]" if len(suffix) > self._settings['logger_info_cutoff'] else f": {suffix}"
                prefix = f"Found cached embedding{'' if len(texts) == 1 else 's'} of '{len(texts)}' text{'' if len(texts) == 1 else 's'}"
                message = f"{prefix}{suffix}"
                embedding = embeddings
            else:
                prefix = "Found cached embedding"
                if self._settings['logger_info_cutoff'] == 0:
                    suffix = "."
                else:
                    suffix = texts[0]
                    suffix = f"{suffix[:self._settings['logger_info_cutoff']]}{'...' if len(suffix) > self._settings['logger_info_cutoff'] else ''}"
                    suffix = f":\n'\n{suffix}\n'" if "\n" in suffix else f": '{suffix}'"
                message = f"{prefix}{suffix}"
                embedding = embeddings[0]
            return True, message, embedding
        if num_cached == 0:
            if len(texts) == 1:
                self._logger.debug("Retrieving embedding from Embeddings API.")
            else:
                self._logger.debug(f"Retrieving all '{len(texts)}' embeddings from Embeddings API.")
        else:
            prefix = f"Found cached embedding{'' if num_cached == 1 else 's'} for '{num_cached}' of '{len(texts)}' text{'' if len(texts) == 1 else 's'}"
            if self._settings['logger_info_cutoff'] == 0:
                suffix = "."
            else:
                suffix = str([texts[i] for i in range(len(texts)) if embeddings[i] is not None])
                suffix = f": {suffix[:self._settings['logger_info_cutoff']]}...]" if len(suffix) > self._settings['logger_info_cutoff'] else f": {suffix}"
            message = f"{prefix}{suffix}"
            if self._settings['logger_info_progress']:
                self._logger.info(message)
            else:
                self._logger.debug(message)
            self._logger.debug(f"Retrieving '{num_cached}' missing embedding{'' if num_cached == 1 else 's'} from Embeddings API.")

        # associate missing embeddings
        missing_texts, missing_idx = [], []
        for i, string in enumerate(texts):
            if embeddings[i] is None:
                missing_texts.append(string)
                missing_idx.append(i)

        # retrieve API key
        api_key = self.get_api_key()[2]

        # validate connection
        success, message = self.validate_connection(api_key=api_key)
        if not success:
            return False, message, None

        # filter duplicates before requests
        missing_texts_unique = list(dict.fromkeys(missing_texts))
        num_filtered = len(missing_texts) - len(missing_texts_unique)
        if num_filtered > 0:
            self._logger.debug(f"Filtered '{num_filtered}' duplicate text{'' if num_filtered == 1 else 's'} before posting request(s).")

        # prepare batches
        text_ranges = []
        if self._settings['max_batch_size'] is None:
            max_batch_size = len(missing_texts_unique)
        else:
            max_batch_size = self._settings['max_batch_size']
        floor_mod = (len(missing_texts_unique) // max_batch_size, len(missing_texts_unique) % max_batch_size)
        for i in range(floor_mod[0]):
            text_ranges.append(missing_texts_unique[i * max_batch_size: (i + 1) * max_batch_size])
        if floor_mod[1] > 0:
            text_ranges.append(missing_texts_unique[floor_mod[0] * max_batch_size:])
        self._logger.debug(f"Distributing '{len(text_ranges)}' POST request{'' if len(text_ranges) == 1 else 's'}: {text_ranges}")

        # construct header
        headers = {
            'Content-Type': "application/json",
            'Authorization': f"Bearer {api_key}"
        }

        # retrieve batched embeddings
        new_embeddings = []
        for missing_texts_post in text_ranges:
            data = {
                'input': missing_texts_post,
                'model': self._settings['model'],
                'encoding_format': "float"
            }

            # log
            if input_is_list:
                if len(texts) == 1:
                    prefix = f"Generating embedding{'' if len(missing_texts_post) == 1 else 's'} of '{len(missing_texts_post)}' text"
                else:
                    prefix = f"Generating embedding{'' if len(missing_texts_post) == 1 else 's'} for '{len(missing_texts_post)}' of '{len(texts)}' texts"
                if self._settings['logger_info_cutoff'] == 0:
                    suffix = "."
                else:
                    suffix = f"{missing_texts_post}"
                    suffix = f": {suffix[:self._settings['logger_info_cutoff']]}...]" if len(suffix) > self._settings['logger_info_cutoff'] else f": {suffix}"
                message = f"{prefix}{suffix}"
                if self._settings['logger_info_requests']:
                    self._logger.info(message)
                else:
                    self._logger.debug(message)
            else:
                prefix = "Generating embedding"
                if self._settings['logger_info_cutoff'] == 0:
                    suffix = "."
                else:
                    suffix = f"{missing_texts_post[0]}"
                    suffix = f"{suffix[:self._settings['logger_info_cutoff']]}{'...' if len(suffix) > self._settings['logger_info_cutoff'] else ''}"
                    suffix = f":\n'\n{suffix}\n'" if "\n" in suffix else f": '{suffix}'"
                message = f"{prefix}{suffix}"
                if self._settings['logger_info_requests']:
                    self._logger.info(message)
                else:
                    self._logger.debug(message)

            # use API
            success, message, response = post_request(
                api_name="Embeddings API",
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
                    # self._logger.warn(f"Keys: {response.keys()}")                                        # [2025-11-14T15:37:54.660865]: Keys: dict_keys(['object', 'data', 'model', 'usage'])
                    # self._logger.warn(f"object: {response['object']}")                                   # [2025-11-14T15:37:54.660865]: object: list
                    # self._logger.warn(f"len(data): {len(response['data'])}")                             # [2025-11-14T15:37:54.660865]: len(data): 1
                    # self._logger.warn(f"response['data'][0].keys(): {response['data'][0].keys()}")       # [2025-11-14T15:37:54.660865]: response['data'][0].keys(): dict_keys(['object', 'index', 'embedding'])
                    # self._logger.warn(f"response['data'][0]['object']: {response['data'][0]['object']}") # [2025-11-14T15:37:54.660865]: response['data'][0]['object']: embedding
                    # self._logger.warn(f"response['data'][0]['index']: {response['data'][0]['index']}")   # [2025-11-14T15:37:54.660865]: response['data'][0]['index']: 0
                    # self._logger.warn(f"model: {response['model']}")                                     # [2025-11-14T15:37:54.660865]: model: text-embedding-3-large
                    # self._logger.warn(f"usage: {response['usage']}")                                     # [2025-11-14T15:37:54.660865]: usage: {'prompt_tokens': 32, 'total_tokens': 32}
                    new_embeddings += [response['data'][i]['embedding'] for i in range(len(response['data']))]
                except Exception as e:
                    success = False
                    message = f"Failed to parse response from Embeddings API '{self._endpoint['api_url']}': {repr(e)}"
                    embeddings = None

            if not success:
                break

        if success:
            # collect results
            map_text_embedding = dict(zip(missing_texts_unique, new_embeddings))
            new_embeddings = [map_text_embedding[t] for t in texts if t in missing_texts_unique]
            for i, embedding in enumerate(new_embeddings):
                embeddings[missing_idx[i]] = embedding
            if input_is_list:
                embedding = embeddings
            else:
                embedding = embeddings[0]

            # log
            if len(missing_idx) == len(embeddings):
                if len(texts) > 1:
                    prefix = f"Generated embeddings of '{len(texts)}' texts in '{time.perf_counter() - stamp:.3f}s'"
                    if self._settings['logger_info_cutoff'] == 0:
                        suffix = "."
                    else:
                        suffix = f"{missing_texts}"
                        suffix = f": {suffix[:self._settings['logger_info_cutoff']]}...]" if len(suffix) > self._settings['logger_info_cutoff'] else f": {suffix}"
                elif input_is_list:
                    prefix = f"Generated embedding of '{len(texts)}' text in '{time.perf_counter() - stamp:.3f}s'"
                    if self._settings['logger_info_cutoff'] == 0:
                        suffix = "."
                    else:
                        suffix = f"{missing_texts}"
                        suffix = f": {suffix[:self._settings['logger_info_cutoff']]}...]" if len(suffix) > self._settings['logger_info_cutoff'] else f": {suffix}"
                else:
                    prefix = f"Generated embedding in '{time.perf_counter() - stamp:.3f}s'"
                    if self._settings['logger_info_cutoff'] == 0:
                        suffix = "."
                    else:
                        suffix = missing_texts[0]
                        suffix = f"{suffix[:self._settings['logger_info_cutoff']]}{'...' if len(suffix) > self._settings['logger_info_cutoff'] else ''}"
                        suffix = f":\n'\n{suffix}\n'" if "\n" in suffix else f": '{suffix}'"
                message = f"{prefix}{suffix}"
            elif len(texts) > 1:
                prefix = f"Generated embeddings for '{len(missing_texts)}' of '{len(texts)}' texts"
                if self._settings['logger_info_cutoff'] == 0:
                    suffix = "."
                else:
                    suffix = f"{missing_texts}"
                    suffix = f": {suffix[:self._settings['logger_info_cutoff']]}...]" if len(suffix) > self._settings['logger_info_cutoff'] else f": {suffix}"
                message = f"{prefix}{suffix}"
                if self._settings['logger_info_progress']:
                    self._logger.info(message)
                else:
                    self._logger.debug(message)
                message = f"Obtained '{len(embeddings)}' embedding{'' if len(embeddings) == 1 else 's'} in '{time.perf_counter() - stamp:.3f}s'."

            if self._settings['cache_write']:
                # cache generated embeddings
                cache_job = {
                    'type': "embeddings",
                    'folder': self._settings['cache_folder'],
                    'index': self._settings['cache_file'],
                    'endpoint': self._settings['endpoint'],
                    'model': self._settings['model'],
                    'texts': missing_texts,
                    'embeddings': [embeddings[i] for i in missing_idx]
                }
                nimbro_api.register_deferred_job(job=(self.cache_embeddings, cache_job), mute=True)
        else:
            embedding = None

        return success, message, embedding
