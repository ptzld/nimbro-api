import os
import time
import atexit
import asyncio
import threading
import importlib.util
import concurrent.futures

import httpx

import nimbro_api
from .misc import UnrecoverableError, assert_type_value, assert_keys, assert_log, format_obj

# internal

_HTTP2_AVAILABLE = importlib.util.find_spec("h2") is not None

class _HttpxStreamResponse:
    """
    Provides synchronous access to an asynchronous streaming HTTP response.
    """

    def __init__(self, runner, response, client, cancel_event):
        self._runner = runner
        self._response = response
        self._client = client
        self._cancel_event = cancel_event
        self._iterator = None
        self._future = None
        self._buffer = b""
        self._finished = False
        self._closed = False
        self._lock = threading.Lock()

    def __getattr__(self, name):
        if name.startswith("_"):
            raise AttributeError(name)
        if name in ("content", "text", "json"):
            raise AttributeError(f"'{name}' is not available on streaming responses. Use 'iter_content()' instead.")
        return getattr(self._response, name)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()

    async def _next_async(self):
        if self._iterator is None:
            self._iterator = self._response.aiter_bytes()

        return await anext(self._iterator)

    def _next(self):
        with self._lock:
            if self._closed or self._finished:
                raise StopIteration

            future = asyncio.run_coroutine_threadsafe(
                self._next_async(),
                self._runner._loop
            )
            self._future = future

        try:
            return self._runner._wait_future(
                future,
                cancel_event=self._cancel_event
            )
        except StopAsyncIteration as exc:
            self._finished = True
            raise StopIteration from exc
        finally:
            with self._lock:
                if self._future is future:
                    self._future = None

    def iter_content(self, chunk_size=1):
        if not isinstance(chunk_size, int) or chunk_size <= 0:
            raise ValueError("Expected argument 'chunk_size' to be a positive integer.")

        try:
            while True:
                while len(self._buffer) < chunk_size and not self._finished:
                    try:
                        self._buffer += self._next()
                    except StopIteration:
                        break

                if len(self._buffer) == 0:
                    break

                chunk = self._buffer[:chunk_size]
                self._buffer = self._buffer[chunk_size:]
                yield chunk
        finally:
            self.close()

    def close(self):
        with self._lock:
            if self._closed:
                return

            self._closed = True
            future = self._future

        if future is not None:
            future.cancel()

        try:
            close_future = asyncio.run_coroutine_threadsafe(
                self._runner._close_stream_async(self._response, self._client),
                self._runner._loop
            )
            close_future.result(timeout=1.0)
        except Exception:
            pass

class _HttpxRunner:
    """
    Runs a persistent asynchronous HTTP client in a background event loop.

    This preserves the synchronous public interface while allowing active requests
    to be cancelled immediately when the calling thread is interrupted.
    """

    def __init__(self):
        self._pid = os.getpid()
        self._loop = asyncio.new_event_loop()
        self._client = None
        self._active = {}
        self._retired = set()
        self._started = threading.Event()
        self._thread = threading.Thread(
            target=self._run,
            name="nimbro-api-httpx",
            daemon=True
        )
        self._thread.start()
        self._started.wait()

    def _run(self):
        asyncio.set_event_loop(self._loop)
        self._started.set()
        self._loop.run_forever()

    def _get_timeout(self, timeout):
        if timeout is None:
            connect = None
            read = None
        else:
            if timeout == "default":
                timeout = ("default", "default")
            if "default" in timeout:
                settings = nimbro_api.get_settings()
            connect = settings['http_timeout_connect'] if timeout[0] == "default" else timeout[0]
            read = settings['http_timeout_read'] if timeout[1] == "default" else timeout[1]
        return httpx.Timeout(
            connect=connect,
            read=read,
            write=read,
            pool=connect
        )

    def _get_client(self):
        if self._client is None:
            settings = nimbro_api.get_settings()
            self._client = httpx.AsyncClient(
                follow_redirects=settings['http_follow_redirects'],
                http2=settings['http_use_http2'],
                limits=httpx.Limits(
                    max_connections=settings['http_max_connections'],
                    max_keepalive_connections=settings['http_max_keepalive_connections'],
                    keepalive_expiry=settings['http_keepalive_expiry']
                )
            )
        return self._client

    def _wait_future(self, future, cancel_event):
        done = threading.Event()
        future.add_done_callback(lambda _: done.set())
        try:
            while True:
                if cancel_event is not None and cancel_event.is_set():
                    raise HttpRequestCancelled(
                        "HTTP request was cancelled."
                    )
                if not done.wait(timeout=0.1):
                    continue
                try:
                    return future.result()
                except concurrent.futures.CancelledError as exc:
                    if cancel_event is not None and cancel_event.is_set():
                        raise HttpRequestCancelled("HTTP request was cancelled.") from exc
                    raise
        except BaseException:
            if not future.cancel():
                self._discard_result(future)
            raise

    def _discard_result(self, future):
        if not future.done() or future.cancelled() or future.exception() is not None:
            return

        result = future.result()
        if isinstance(result, tuple):
            coroutine = self._close_stream_async(*result)
        elif hasattr(result, "aclose"):
            coroutine = result.aclose()
        else:
            return

        try:
            close_future = asyncio.run_coroutine_threadsafe(
                coroutine,
                self._loop
            )
            close_future.result(timeout=1.0)
        except Exception:
            pass

    def _acquire_client(self):
        client = self._get_client()
        self._active[client] = self._active.get(client, 0) + 1

        return client

    async def _release_client(self, client):
        self._active[client] -= 1
        if self._active[client] == 0:
            del self._active[client]
            if client in self._retired:
                self._retired.discard(client)
                await client.aclose()

    async def _request_async(self, method, api_url, *, headers, json, data, files, timeout):
        client = self._acquire_client()

        try:
            return await client.request(
                method=method,
                url=api_url,
                headers=headers,
                json=json,
                data=data,
                files=files,
                timeout=self._get_timeout(timeout)
            )
        finally:
            await self._release_client(client)

    async def _stream_request_async(self, method, api_url, *, headers, json, data, files, timeout):
        client = self._acquire_client()

        try:
            request = client.build_request(
                method=method,
                url=api_url,
                headers=headers,
                json=json,
                data=data,
                files=files,
                timeout=self._get_timeout(timeout)
            )

            response = await client.send(
                request=request,
                stream=True
            )
        except BaseException:
            await self._release_client(client)
            raise

        return response, client

    async def _close_stream_async(self, response, client):
        try:
            await response.aclose()
        finally:
            await self._release_client(client)

    async def _reload_client_async(self):
        client = self._client
        self._client = None
        if client is not None:
            if client in self._active:
                self._retired.add(client)
            else:
                await client.aclose()

    def reload_client(self):
        try:
            future = asyncio.run_coroutine_threadsafe(
                self._reload_client_async(),
                self._loop
            )
            future.result(timeout=1.0)
        except Exception:
            pass

    def request(self, method, api_url, *, headers, json=None, data=None, files=None, timeout="default", cancel_event=None):
        coroutine = self._request_async(
            method=method,
            api_url=api_url,
            headers=headers,
            json=json,
            data=data,
            files=files,
            timeout=timeout
        )
        future = asyncio.run_coroutine_threadsafe(
            coroutine,
            self._loop
        )

        return self._wait_future(
            future,
            cancel_event=cancel_event
        )

    def stream_request(self, method, api_url, *, headers, json=None, data=None, files=None, timeout="default", cancel_event=None):
        coroutine = self._stream_request_async(
            method=method,
            api_url=api_url,
            headers=headers,
            json=json,
            data=data,
            files=files,
            timeout=timeout
        )
        future = asyncio.run_coroutine_threadsafe(
            coroutine,
            self._loop
        )

        response, client = self._wait_future(
            future,
            cancel_event=cancel_event
        )

        return _HttpxStreamResponse(
            runner=self,
            response=response,
            client=client,
            cancel_event=cancel_event
        )

    async def _stop_async(self):
        for task in asyncio.all_tasks():
            if task is not asyncio.current_task():
                task.cancel()

        self._loop.call_soon(self._loop.stop)

    async def _close_async(self):
        clients = set(self._retired)
        if self._client is not None:
            clients.add(self._client)
        self._client = None
        self._retired.clear()

        for client in clients:
            await client.aclose()

    def close(self):
        if not self._thread.is_alive():
            return

        try:
            future = asyncio.run_coroutine_threadsafe(
                self._close_async(),
                self._loop
            )
            future.result(timeout=1.0)
        except Exception:
            pass

        try:
            future = asyncio.run_coroutine_threadsafe(
                self._stop_async(),
                self._loop
            )
            future.result(timeout=1.0)
        except Exception:
            pass

        self._thread.join(timeout=1.0)

        if not self._thread.is_alive():
            try:
                self._loop.close()
            except Exception:
                pass

_HTTPX_RUNNER = None
_HTTPX_RUNNER_LOCK = threading.Lock()

def _get_httpx_runner():
    global _HTTPX_RUNNER

    pid = os.getpid()

    with _HTTPX_RUNNER_LOCK:
        if _HTTPX_RUNNER is None or _HTTPX_RUNNER._pid != pid:
            _HTTPX_RUNNER = _HttpxRunner()

    return _HTTPX_RUNNER

def _close_httpx_runner():
    global _HTTPX_RUNNER

    if _HTTPX_RUNNER is not None and _HTTPX_RUNNER._pid == os.getpid():
        _HTTPX_RUNNER.close()
        _HTTPX_RUNNER = None

def _reload_httpx_settings():
    with _HTTPX_RUNNER_LOCK:
        runner = _HTTPX_RUNNER
    if runner is not None and runner._pid == os.getpid():
        runner.reload_client()

atexit.register(_close_httpx_runner)

def _assert_timeout(timeout):
    assert_type_value(obj=timeout, type_or_value=[tuple, "default", None], name="argument 'timeout'")
    if isinstance(timeout, tuple):
        assert_log(expression=len(timeout) == 2, message=f"Expected argument 'timeout' to be a tuple of length '2', but it has length '{len(timeout)}'.")
        for i, t in enumerate(timeout):
            assert_type_value(obj=t, type_or_value=[float, int, "default", None], name=f"element '{i}' in argument 'timeout'")

def _process_response(api_name, api_url, response, duration, logger):
    status_code = response.status_code
    http_version = response.http_version

    try:
        response_str = response.json()
    except Exception:
        response_str = response.text.strip()
        is_json = False
    else:
        is_json = True

    if logger is not None:
        logger.debug(f"Received response with status code '{status_code}' via '{http_version}' from {api_name} '{api_url}' after '{duration:.3f}s': {format_obj(response_str)}.")

    if status_code == 200:
        success = True
        message = f"Received response from {api_name} after '{duration:.3f}s'."
    else:
        success = False
        if is_json:
            while True:
                if isinstance(response_str, dict):
                    if response_str.get('code') == status_code:
                        del response_str['code']
                    if len(response_str) == 1:
                        response_str = response_str[list(response_str.keys())[0]]
                    else:
                        break
                else:
                    break
        if len(str(response_str).strip()) == 0:
            message = f"{api_name.capitalize()} '{api_url}' responded with status code '{status_code}' via '{http_version}' after '{duration:.3f}s'."
        else:
            message = f"{api_name.capitalize()} '{api_url}' responded with status code '{status_code}' via '{http_version}' after '{duration:.3f}s': {format_obj(response_str)}."

    return success, message

# public

def get_api_key(self):
    """
    Retrieves the API key from the set endpoint.

    Raises:
        UnrecoverableError: If the environment variable for 'api_name' is not set or 'self' does not provide the expected attributes.

    Returns:
        tuple[bool, str, str]: A tuple containing:
            - bool: `True` if the operation succeeded, `False` otherwise.
            - str: A descriptive message about the operation result.
            - str: The API key for the set endpoint.

    Notes:
        - This function assumes a validated endpoint (see `validate_endpoint()`) stored in `self._endpoint`.
    """
    # parse arguments
    assert_log(expression=hasattr(self, "_endpoint"), message="Expected argument 'self' to provide attribute '_endpoint'.")
    assert_log(expression=hasattr(self, "_logger"), message="Expected argument 'self' to provide attribute '_logger'.")
    assert_type_value(obj=self._endpoint, type_or_value=dict, name="attribute '_endpoint' of argument 'self'")

    if self._endpoint['key_value'] == "":
        api_key = ""
        message = "Using no API key."
        self._logger.debug(message)
    elif self._endpoint['key_type'] == "environment":
        success, message, api_key = nimbro_api.get_api_key(name=self._endpoint['key_value'], mute=True)
        if not success:
            raise UnrecoverableError(message)
        self._logger.debug(message)
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
        - The required key is always "api_url".
        - "key_type" and "key_value" are required when argument 'require_key' is `True`.
        - "name" is required when argument 'require_name' is `True`.
        - If argument 'flavors' is provided and non-empty, "api_flavor" is also required and it's value must be in 'flavors'.
        - The only permitted optional key is "models_url".
        - All keys and values in 'endpoint' must be non-empty strings, except the value of key "key_value", which can be an empty string.
        - The value of "key_type" must be either "environment" or "plain".
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
        if key != "key_value":
            assert_log(expression=len(endpoint[key]) > 0, message=f"Expected value of key '{key}' in {setting_name} to be a non-empty string.")

    if has_flavors:
        assert_type_value(obj=endpoint['api_flavor'], type_or_value=flavors, name=f"key 'api_flavor' in {setting_name}")
    if require_key:
        assert_type_value(obj=endpoint['key_type'], type_or_value=["environment", "plain"], name=f"key 'key_type' in {setting_name}")

class HttpRequestCancelled(Exception):
    """
    Raised when an HTTP request is cancelled through its cancellation event.

    This exception indicates intentional cancellation rather than a network,
    protocol, or timeout failure. It is raised while establishing a connection,
    sending a request, waiting for response headers, or reading a streaming
    response when the provided `cancel_event` becomes set.
    """

def post_request(api_name, api_url, *, headers, data, files=None, timeout="default", logger=None):
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
        timeout (str | None | tuple[str | int | float | None, str | int | float | None], optional):
            An object specifying the connect and read timeouts in seconds:
            - Use "default" (`str`) to adopt core settings 'http_timeout_connect' and 'http_timeout_read'.
            - Use None to not time out when connecting or reading.
            - Use a tuple with two elements, where the first one specifies connection timeout, and the second one specifies the read timeout.
              Both values can be `int` or `float` in seconds, as well as "default" or None as described above.
            Defaults to "default".
        logger (nimbro_api.utility.logger.Logger | None, optional):
            If provided, logs the request attempt, the payload, and the received response. Defaults to `None`.

    Raises:
        UnrecoverableError: If input arguments are invalid.
        KeyboardInterrupt | SystemExit: Cancel the active asynchronous request and propagate to the caller.

    Returns:
        tuple[bool, str, httpx.Response | None]: A tuple containing:
            - bool: `True` if the operation succeeded (status code 200), `False` otherwise.
            - str: A descriptive message about the operation result.
            - httpx.Response | None: The response object if a response was received, otherwise `None`.

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
    _assert_timeout(timeout=timeout)

    if logger is not None:
        if data is None:
            logger.debug(f"Sending POST request to {api_name} '{api_url}'.")
        else:
            logger.debug(f"Sending POST request to {api_name} '{api_url}': {format_obj(data)}.")

    tic = time.perf_counter()
    try:
        if files is None:
            response = _get_httpx_runner().request(
                method="POST",
                api_url=api_url,
                headers=headers,
                json=data,
                timeout=timeout
            )
        else:
            response = _get_httpx_runner().request(
                method="POST",
                api_url=api_url,
                headers=headers,
                data=data,
                files=files,
                timeout=timeout
            )
    except Exception as e:
        duration = time.perf_counter() - tic
        success = False
        message = f"Failed to receive response from {api_name} '{api_url}' after '{duration:.3f}s': {repr(e)}."
        response = None
    else:
        duration = time.perf_counter() - tic
        success, message = _process_response(api_name=api_name, api_url=api_url, response=response, duration=duration, logger=logger)
        # if not success:
        #     response = None

    return success, message, response

def get_request(api_name, api_url, *, headers, timeout="default", logger=None):
    """
    Sends an HTTP GET request to a specified API endpoint and processes the response.

    Args:
        api_name (str):
            The display name of the API being called, used for logging and error messages.
        api_url (str):
            The full URL of the endpoint to which the GET request is sent.
        headers (dict):
            A dictionary of HTTP headers to include in the request.
        timeout (str | None | tuple[str | int | float | None, str | int | float | None], optional):
            An object specifying the connect and read timeouts in seconds:
            - Use "default" (`str`) to adopt core settings 'http_timeout_connect' and 'http_timeout_read'.
            - Use None to not time out when connecting or reading.
            - Use a tuple with two elements, where the first one specifies connection timeout, and the second one specifies the read timeout.
              Both values can be `int` or `float` in seconds, as well as "default" or None as described above.
            Defaults to "default".
        logger (nimbro_api.utility.logger.Logger | None, optional):
            If provided, logs the request attempt and the received response (including the response body). Defaults to `None`.

    Raises:
        UnrecoverableError: If input arguments are invalid.
        KeyboardInterrupt | SystemExit: Cancel the active asynchronous request and propagate to the caller.

    Returns:
        tuple[bool, str, httpx.Response | None]: A tuple containing:
            - bool: `True` if the operation succeeded (status code 200), `False` otherwise.
            - str: A descriptive message about the operation result.
            - httpx.Response | None: The response object if a response was received, otherwise `None`.

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
    _assert_timeout(timeout=timeout)

    if logger is not None:
        logger.debug(f"Sending GET request to {api_name} '{api_url}'.")

    tic = time.perf_counter()
    try:
        response = _get_httpx_runner().request(
            method="GET",
            api_url=api_url,
            headers=headers,
            timeout=timeout
        )
    except Exception as e:
        duration = time.perf_counter() - tic
        success = False
        message = f"Failed to receive response from {api_name} '{api_url}' after '{duration:.3f}s': {repr(e)}."
        response = None
    else:
        duration = time.perf_counter() - tic
        success, message = _process_response(api_name=api_name, api_url=api_url, response=response, duration=duration, logger=logger)
        # if not success:
        #     response = None

    return success, message, response

def http_request(method, api_url, *, headers, json=None, data=None, files=None, stream=False, timeout="default", cancel_event=None, logger=None):
    """
    Sends a buffered or streaming HTTP request through the shared HTTP client.

    Args:
        method (str):
            The HTTP request method.
        api_url (str):
            The full URL of the endpoint.
        headers (dict):
            A dictionary of HTTP headers to include in the request.
        json (dict | list | None, optional):
            The JSON-encoded request body. Defaults to `None`.
        data (dict | list | str | bytes | None, optional):
            The form or raw request body. Defaults to `None`.
        files (dict | None, optional):
            Files to include in a multi-part request. Defaults to `None`.
        stream (bool, optional):
            Whether to stream the response body. Defaults to `False`.
        timeout (str | None | tuple[str | int | float | None, str | int | float | None], optional):
            An object specifying the connect and read timeouts in seconds:
            - Use "default" (`str`) to adopt core settings 'http_timeout_connect' and 'http_timeout_read'.
            - Use None to not time out when connecting or reading.
            - Use a tuple with two elements, where the first one specifies connection timeout, and the second one specifies the read timeout.
              Both values can be `int` or `float` in seconds, as well as "default" or None as described above.
            Defaults to "default".
        cancel_event (threading.Event | None, optional):
            An event used to cancel the request. Defaults to `None`.
        logger (nimbro_api.utility.logger.Logger | None, optional):
            If provided, logs the request and received response headers.
            Defaults to `None`.

    Raises:
        UnrecoverableError: If input arguments are invalid.
        HttpRequestCancelled: If `cancel_event` becomes set while the request is active.
        httpx.HTTPError: If the request fails due to a network, protocol, or timeout error.
        KeyboardInterrupt | SystemExit: Cancel the active asynchronous request and propagate to the caller.

    Returns:
        httpx.Response | _HttpxStreamResponse: The buffered or streaming response.
    """
    # parse arguments
    from nimbro_api.utility.logger import Logger
    assert_type_value(obj=logger, type_or_value=[Logger, None], name="argument 'logger'")
    assert_type_value(obj=method, type_or_value=str, name="argument 'method'")
    assert_type_value(obj=api_url, type_or_value=str, name="argument 'api_url'")
    assert_type_value(obj=headers, type_or_value=dict, name="argument 'headers'")
    assert_type_value(obj=json, type_or_value=[dict, list, None], name="argument 'json'")
    assert_type_value(obj=data, type_or_value=[dict, list, str, bytes, None], name="argument 'data'")
    assert_type_value(obj=files, type_or_value=[dict, None], name="argument 'files'")
    assert_type_value(obj=stream, type_or_value=bool, name="argument 'stream'")
    _assert_timeout(timeout=timeout)
    assert_type_value(obj=cancel_event, type_or_value=[threading.Event, None], name="argument 'cancel_event'")

    method = method.upper()

    if logger is not None:
        logger.debug(f"Sending {method} request to '{api_url}'.")

    tic = time.perf_counter()

    try:
        runner = _get_httpx_runner()

        if stream:
            response = runner.stream_request(
                method,
                api_url,
                headers=headers,
                json=json,
                data=data,
                files=files,
                timeout=timeout,
                cancel_event=cancel_event
            )
        else:
            response = runner.request(
                method,
                api_url,
                headers=headers,
                json=json,
                data=data,
                files=files,
                timeout=timeout,
                cancel_event=cancel_event
            )
    except BaseException as e:
        duration = time.perf_counter() - tic
        if logger is not None:
            logger.debug(f"Failed to receive {method} response from '{api_url}' after '{duration:.3f}s': {repr(e)}.")
        raise

    duration = time.perf_counter() - tic

    if logger is not None:
        if stream:
            response_type = "streaming response"
        else:
            response_type = "response"
        logger.debug(f"Received {method} {response_type} with status code '{response.status_code}' via '{response.http_version}' from '{api_url}' after '{duration:.3f}s'.")

    return response
