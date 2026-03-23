import time
import requests

import nimbro_api
from .misc import UnrecoverableError, assert_type_value, assert_keys, assert_log, format_obj

def get_api_key(self):
    """
    Retrieves an API key from the environment variables for the current endpoint.

    Args:
        api_name (str):
            The name of the API for which the key is being retrieved. Used to construct
            the environment variable search key.
        logger (nimbro_api.utility.logger.Logger | None, optional):
            If provided, logs the success or failure of the environment variable retrieval.
            Defaults to `None`.

    Raises:
        UnrecoverableError: If the environment variable for 'api_name' is not set.

    Returns:
        str: The retrieved API key string.

    Notes:
        - The function expects the environment variable to follow a naming convention derived from 'api_name'.
        - If 'logger' is provided, the API key value itself is never logged; only the status of the retrieval is recorded.
    """
    if self._endpoint['key_type'] == "environment":
        success, message, api_key = nimbro_api.get_api_key(name=self._endpoint['key_value'], mute=True)
        if not success:
            raise UnrecoverableError(message)
    else:
        api_key = self._endpoint['key_value']
        if nimbro_api.get_settings(name='keys_hide'):
            message = "Using plain API key."
        else:
            message = f"Using plain API key '{api_key}'."
        self._logger.debug(message)

    return True, message, api_key

def validate_endpoint(endpoint, *, flavors, require_key, require_name, setting_name):
    """
    Validates the structure and content of an endpoint definition.

    Args:
        endpoint (dict):
            The dictionary representing an API endpoint to be validated.
        flavors (list | None):
            A list of valid values for the "api_flavor" key of the 'endpoint'.
            If `None` or an empty `list`, the "api_flavor" key of the 'endpoint' is not required.
        require_key (bool):
            Determines if the 'endpoint' requires the keys "key_type" and "key_value".
        require_name (bool):
            Determines if the 'endpoint' requires the key "name".
        setting_name (str):
            The name of the 'endpoint' being validated.

    Raises:
        UnrecoverableError: If input arguments are invalid or if 'endpoint' violates any constraint.

    Notes:
        - Required keys always include "api_url", "key_type", and "key_value".
        - If 'flavors' is provided and non-empty, "api_flavor" is also required and it's value must be in 'flavors'.
        - The only permitted optional key is "models_url".
        - All keys and values in 'endpoint' must be non-empty strings.
        - The value of 'key_type' must be either "environment" or "plain".
    """
    # parse arguments
    assert_type_value(obj=flavors, type_or_value=[list, None], name="argument 'flavors'")
    assert_type_value(obj=setting_name, type_or_value=str, name="argument 'setting_name'")

    if flavors is None or len(flavors) == 0:
        has_flavors = False
    else:
        has_flavors = True

    # required and optional keys
    assert_type_value(obj=endpoint, type_or_value=dict, name=setting_name)
    keys_required = ['api_url']
    if require_key:
        keys_required.append('key_type')
        keys_required.append('key_value')
    if require_name:
        keys_required.append('name')

    if has_flavors:
        keys_required.insert(1, 'api_flavor')
    assert_keys(obj=endpoint, keys=keys_required, mode="required", name=setting_name)
    keys_optional = ['models_url']
    assert_keys(obj=endpoint, keys=keys_required + keys_optional, mode="whitelist", name=setting_name)

    # keys and values are non-empty strings
    for key in endpoint:
        assert_type_value(obj=key, type_or_value=str, name=f"key '{key}' in {setting_name}")
        assert_log(expression=len(key) > 0, message=f"Expected key '{key}' in {setting_name} to be a non-empty string.")
        assert_type_value(obj=endpoint[key], type_or_value=str, name=f"key '{key}' in {setting_name}")
        assert_log(expression=len(endpoint[key]) > 0, message=f"Expected value of key '{key}' in {setting_name} to be a non-empty string.")

    if has_flavors:
        assert_type_value(obj=endpoint['api_flavor'], type_or_value=flavors, name=f"key 'api_flavor' in {setting_name}")
    if require_key:
        assert_type_value(obj=endpoint['key_type'], type_or_value=["environment", "plain"], name=f"key 'key_type' in {setting_name}")

def post_request(api_name, api_url, *, headers, data, files=None, timeout=(None, None), logger=None):
    """
    Sends an HTTP POST request to a specified API endpoint and processes the response.

    Args:
        api_name (str):
            The display name of the API being called, used for logging and error messages.
        api_url (str):
            The full URL of the endpoint to which the POST request is sent.
        headers (dict):
            A dictionary of HTTP headers to include in the request.
        data (dict | list | None):
            The payload to be sent in the body of the request.
        files (dict | None, optional):
            A dictionary of file objects to be uploaded in a multi-part POST request.
            Defaults to `None`.
        timeout (tuple[float | None, float | None], optional):
            A tuple containing the connect and read timeouts in seconds. Defaults to `(None, None)`.
        logger (nimbro_api.utility.logger.Logger | None, optional):
            If provided, logs the request attempt, the payload, and the received response. Defaults to `None`.

    Raises:
        UnrecoverableError: If input arguments are invalid.

    Returns:
        tuple[bool, str, requests.Response | None]: A tuple containing:
            - bool: A flag indicating whether the request was successful (status code "200").
            - str: A message describing the outcome, including the duration of the request.
            - requests.Response | None: The response object if a response was received, otherwise `None`.

    Notes:
        - If 'files' is `None`, 'data' is sent as a JSON-encoded body.
        - If 'files' is provided, the request is sent as "multipart/form-data".
        - If the request fails or returns a non-200 status code, the function attempts
          to extract error details from the response body, prioritizing JSON content.
    """
    # parse arguments
    from nimbro_api.utility.logger import Logger
    assert_type_value(obj=logger, type_or_value=[Logger, None], name="argument 'logger'")
    assert_type_value(obj=api_name, type_or_value=str, name="argument 'api_name'")
    assert_type_value(obj=api_url, type_or_value=str, name="argument 'api_url'")
    assert_type_value(obj=headers, type_or_value=dict, name="argument 'headers'")
    assert_type_value(obj=data, type_or_value=[dict, list, type(None)], name="argument 'data'")
    assert_type_value(obj=files, type_or_value=[dict, type(None)], name="argument 'files'")
    assert_type_value(obj=timeout, type_or_value=tuple, name="argument 'timeout'")
    assert_log(expression=len(timeout) == 2, message=f"Expected argument 'timeout' to be a tuple of length '2', but it has length '{len(timeout)}'.")
    for i, t in enumerate(timeout):
        assert_type_value(obj=t, type_or_value=[float, int, type(None)], name=f"element '{i}' in argument 'timeout'")

    if logger is not None:
        if data is None:
            logger.debug(f"Sending POST request to {api_name} '{api_url}'.")
        else:
            logger.debug(f"Sending POST request to {api_name} '{api_url}': {format_obj(data)}.")

    tic = time.perf_counter()
    try:
        if files is None:
            response = requests.post(
                api_url,
                headers=headers,
                json=data,
                stream=False,
                timeout=timeout
            )
        else:
            response = requests.post(
                api_url,
                headers=headers,
                files=files,
                data=data,
                stream=False,
                timeout=timeout
            )
    except Exception as e:
        duration = time.perf_counter() - tic
        success = False
        message = f"Failed to receive response from {api_name} '{api_url}' after '{duration:.3f}s': {repr(e)}"
        response = None
    else:
        duration = time.perf_counter() - tic

        status_code = response.status_code

        if status_code == 200:
            success = True
            message = f"Received response from {api_name} after '{duration:.3f}s'."
            if logger is not None:
                try:
                    response_str = response.json()
                except Exception:
                    response_str = response.text.strip()
                logger.debug(f"Received response {status_code} from {api_name} '{api_url}' after '{duration:.3f}s': {format_obj(response_str)}.")
        else:
            success = False
            try:
                response_str = response.json()
            except Exception:
                response_str = response.text.strip()
            else:
                while True:
                    if isinstance(response_str, dict):
                        if response_str.get('code') == response.status_code:
                            del response_str['code']
                        if len(response_str) == 1:
                            response_str = response_str[list(response_str.keys())[0]]
                        else:
                            break
                    else:
                        break
            if len(str(response_str).strip()) == 0:
                message = f"{api_name} '{api_url}' responded with status code '{status_code}' after '{duration:.3f}s'."
            else:
                message = f"{api_name} '{api_url}' responded with status code '{status_code}' after '{duration:.3f}s': {format_obj(response_str)}."
            # response = None

    return success, message, response

def get_request(api_name, api_url, *, headers, timeout=(None, None), logger=None):
    """
    Sends an HTTP GET request to a specified API endpoint and processes the response.

    Args:
        api_name (str):
            The display name of the API being called, used for logging and error messages.
        api_url (str):
            The full URL of the endpoint to which the GET request is sent.
        headers (dict):
            A dictionary of HTTP headers to include in the request.
        timeout (tuple[float | None, float | None], optional):
            A tuple containing the connect and read timeouts in seconds. Defaults to `(None, None)`.
        logger (nimbro_api.utility.logger.Logger | None, optional):
            If provided, logs the request attempt and the received response (including the response body). Defaults to `None`.

    Raises:
        UnrecoverableError: If input arguments are invalid.

    Returns:
        tuple[bool, str, requests.Response | None]: A tuple containing:
            - bool: A flag indicating whether the request was successful (status code "200").
            - str: A message describing the outcome, including the duration of the request.
            - requests.Response | None: The response object if a response was received, otherwise `None`.

    Notes:
        - If the request fails or returns a non-200 status code, the function attempts
          to extract error details from the response body, prioritizing JSON content.
    """

    # parse arguments
    from nimbro_api.utility.logger import Logger
    assert_type_value(obj=logger, type_or_value=[Logger, None], name="argument 'logger'")
    assert_type_value(obj=api_name, type_or_value=str, name="argument 'api_name'")
    assert_type_value(obj=api_url, type_or_value=str, name="argument 'api_url'")
    assert_type_value(obj=headers, type_or_value=dict, name="argument 'headers'")
    assert_type_value(obj=timeout, type_or_value=tuple, name="argument 'timeout'")
    assert_log(expression=len(timeout) == 2, message=f"Expected argument 'timeout' to be a tuple of length '2', but it has length '{len(timeout)}'.")
    for i, t in enumerate(timeout):
        assert_type_value(obj=t, type_or_value=[float, int, type(None)], name=f"element '{i}' in argument 'timeout'")

    if logger is not None:
        logger.debug(f"Sending GET request to {api_name} '{api_url}'.")

    tic = time.perf_counter()
    try:
        response = requests.get(
            api_url,
            headers=headers,
            stream=False,
            timeout=timeout
        )
    except Exception as e:
        duration = time.perf_counter() - tic
        success = False
        message = f"Failed to receive response from {api_name} '{api_url}' after '{duration:.3f}s': {repr(e)}"
        response = None
    else:
        duration = time.perf_counter() - tic
        if logger is not None:
            try:
                response_str = response.json()
            except Exception:
                response_str = response.text.strip()
            logger.debug(f"Received response from {api_name} '{api_url}' after '{duration:.3f}s': {format_obj(response_str)}.")

        status_code = response.status_code

        if status_code == 200:
            success = True
            message = f"Received response from {api_name} after '{duration:.3f}s'."
        else:
            success = False
            try:
                response_str = response.json()
            except Exception:
                response_str = response.text.strip()
            else:
                while True:
                    if isinstance(response_str, dict):
                        if response_str.get('code') == response.status_code:
                            del response_str['code']
                        if len(response_str) == 1:
                            response_str = response_str[list(response_str.keys())[0]]
                        else:
                            break
                    else:
                        break
            if len(str(response_str).strip()) == 0:
                message = f"{api_name} '{api_url}' responded with status code '{status_code}' after '{duration:.3f}s'."
            else:
                message = f"{api_name} '{api_url}' responded with status code '{status_code}' after '{duration:.3f}s': {format_obj(response_str)}."
            # response = None

    return success, message, response
