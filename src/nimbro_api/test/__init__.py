import importlib

__all__ = ["benchmark", "test"]

def benchmark(*, processes=None, threads_per_process=10, duration=10.0, warmup=2.0, prompt_bytes=2048, response_bytes=2048, http="h1", hold_ms=0.0, h2_max_streams=250, max_keepalive_connections="default", request_timeout=10.0, severity="off"):
    """
    Compare a native Go client and ChatCompletions against a local fake HTTPS endpoint.

    Args:
        processes (int | None, optional):
            Positive number of spawned Python processes and native-client connection pools.
            Use `None` for one process per available logical CPU, respecting CPU affinity where supported.
            Defaults to `None`.
        threads_per_process (int, optional):
            Positive number of callers per process. Defaults to `10`.
        duration (float | int, optional):
            Positive measurement duration in seconds. Defaults to `10.0`.
        warmup (float | int, optional):
            Nonnegative warm-up duration in seconds. Zero sends one excluded probe per caller. Defaults to `2.0`.
        prompt_bytes (int, optional):
            Positive ASCII prompt size in bytes. Defaults to `2048`.
        response_bytes (int, optional):
            Positive ASCII completion size in bytes. Defaults to `2048`.
        http (str, optional):
            HTTPS protocol, "h1" or "h2". Defaults to "h1".
        hold_ms (float | int, optional):
            Nonnegative server delay in milliseconds. Defaults to `0.0`.
        h2_max_streams (int, optional):
            HTTP/2 stream limit from `1` through `4294967295`. Defaults to `250`.
        max_keepalive_connections (int | str | None, optional):
            Idle connection limit per process. Use "default" to preserve package settings,
            `None` for unlimited, or a nonnegative integer. Defaults to "default".
        request_timeout (float | int, optional):
            Positive connect, read, and completion timeout in seconds. Defaults to `10.0`.
        severity (str | int, optional):
            Underlying log severity: "debug", "info", "warn", "error", "fatal", or "off".
            Numeric levels `0`, `10`, `20`, `30`, `40`, and `50` are also accepted.
            Use "off" to mute underlying logs and benchmark error diagnostics.
            Benchmark statistics are always emitted. Defaults to "off".

    Raises:
        UnrecoverableError: If arguments are invalid or HTTP/2 dependencies are unavailable.

    Returns:
        int: The number of failed client runs, or `1` for setup failure or interruption.

    Notes:
        - Go 1.26+ on PATH is required. The benchmark is opt-in and is not run by `test()`.
        - Offered concurrency is 'processes' multiplied by 'threads_per_process'.
        - Go runs first, then NimbRo. Each has a separate warm-up and measurement.
        - Enabling underlying logs adds work to the measured workload and can reduce throughput.
        - Results overwrite "benchmark.txt" in the package test directory, including enabled error diagnostics.
        - Relative throughput compares successful measurement requests per second when both clients have successes, even if other requests fail.
        - Workers use the runner's HTTP pool settings with any benchmark override applied.
        - Certificates, binaries, and intermediate logs are temporary. Runner pool settings are restored after execution.
    """
    implementation = importlib.import_module(".benchmark", __name__).benchmark
    globals()['benchmark'] = implementation
    return implementation(processes=processes, threads_per_process=threads_per_process, duration=duration, warmup=warmup, prompt_bytes=prompt_bytes, response_bytes=response_bytes, http=http, hold_ms=hold_ms, h2_max_streams=h2_max_streams, max_keepalive_connections=max_keepalive_connections, request_timeout=request_timeout, severity=severity)

def test(provider=None, *, module=None, function="utilities", common=True, utilities=True, severity="off"):
    """
    Run common client checks, package utilities, and API-specific tests.

    Args:
        provider (str | None, optional):
            API provider to test. Use `None` for all providers. Defaults to `None`.
        module (str | None, optional):
            Case-sensitive substring filter for test filenames and common client class names.
            Use `None` for all modules. Defaults to `None`.
        function (str | None, optional):
            Case-sensitive substring filter for test functions starting with "test_".
            Use `None` to include live tests requiring services and credentials.
            Defaults to "utilities", which selects offline API utility tests.
        common (bool, optional):
            Run common client checks. Defaults to `True`.
        utilities (bool, optional):
            Run package utility checks. Defaults to `True`.
        severity (str | int, optional):
            Underlying log severity: "debug", "info", "warn", "error", "fatal", or "off".
            Numeric levels `0`, `10`, `20`, `30`, `40`, and `50` are also accepted.
            Use "off" to mute underlying logs. Test reports are always emitted. Defaults to "off".

    Raises:
        UnrecoverableError: If arguments are invalid or test preparation fails.

    Returns:
        int: The number of test failures.

    Notes:
        - Results overwrite "test.txt" and "test_pretty.txt" in the package test directory.
        - Test discovery uses each API provider's "test" directory.
        - Logging and credential cache settings changed by the runner are restored after execution.
    """
    implementation = importlib.import_module(".test", __name__).test
    globals()['test'] = implementation
    return implementation(provider, module=module, function=function, common=common, utilities=utilities, severity=severity)
