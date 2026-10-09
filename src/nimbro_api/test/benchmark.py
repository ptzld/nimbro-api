#!/usr/bin/env python3

"""
Compare ChatCompletions dispatch with a native Go client against a local fake endpoint.

Install Go 1.26+ on PATH, then install this checkout with Python 3.9+:
    pip install -e . 'httpx[http2]'

Run from the repository root:
    python src/nimbro_api/test/benchmark.py
    python src/nimbro_api/test/benchmark.py --http h2 --threads-per-process 100 --duration 60
    python src/nimbro_api/test/benchmark.py --processes 100 --threads-per-process 100 --hold-ms 1000
    python src/nimbro_api/test/benchmark.py --prompt-bytes 8192 --response-bytes 1048576 --warmup 5

Defaults: one process per available logical CPU, 10 callers per process, HTTP/1.1,
2 KiB prompt/completion text, immediate responses, and 2 seconds warm-up followed
by 10 seconds measurement for each client. See --help for all workload controls.

Go runs first, then NimbRo, using the same offered concurrency and payload sizes.
Go uses goroutines, pre-encoded JSON and generous connection reuse. It can open
additional HTTP/2 connections when streams are exhausted. NimbRo uses its normal
non-streaming prompt path with retries, model validation, and correction disabled.
Underlying `nimbro_api` logs are muted by default. Use --severity to enable them.
Benchmark reports are always emitted, independently of underlying logging settings.
Compare server receipt rates, active and peak server requests, and completions.
Use --hold-ms to observe overlap. Immediate responses naturally have low overlap.
A strong Go result demonstrates endpoint capacity for that run. If both clients
are slow, that does not establish a NimbRo bottleneck.

Benchmark statistics and enabled error diagnostics are saved to benchmark.txt beside
this script, overwritten each run. Certificates, binaries and intermediate logs are temporary.
Bandwidth counts HTTP body bytes, not HTTP/TLS overhead. Latency percentiles use
approximately 2% histogram buckets. --warmup 0 still probes once per caller.
"""

import os
import sys
import json
import math
import time
import queue
import shutil
import signal
import argparse
import platform
import tempfile
import threading
import subprocess
import faulthandler
import importlib.util
import urllib.request
import multiprocessing
import importlib.metadata
from pathlib import Path
from inspect import signature
from multiprocessing.connection import wait

import nimbro_api

from nimbro_api.utility.logger import Logger
from nimbro_api.utility.misc import UnrecoverableError, assert_type_value, assert_log

HISTOGRAM_BASE = 1.02
REPORT_FILE = None
LOGGER = None

# reporting

def _report(text, error_logger=None):
    if error_logger is not None:
        if not error_logger.error(text):
            return
    elif LOGGER is not None:
        LOGGER.info(text)
    if REPORT_FILE is not None:
        REPORT_FILE.write(text + "\n")
        REPORT_FILE.flush()

def _summarize(snapshots):
    result = {
        'submitted': 0,
        'completed': 0,
        'succeeded': 0,
        'failed': 0,
        'outstanding': 0,
        'max_latency_seconds': 0.0,
        'latency_sum_seconds': 0.0,
        'histogram': {},
        'errors': {},
        'error_samples': []
    }
    for snapshot in snapshots:
        for key in ['submitted', 'completed', 'succeeded', 'failed', 'outstanding', 'latency_sum_seconds']:
            result[key] += snapshot[key]
        result['max_latency_seconds'] = max(result['max_latency_seconds'], snapshot['max_latency_seconds'])
        for field in ['histogram', 'errors']:
            for key, count in snapshot[field].items():
                result[field][key] = result[field].get(key, 0) + count
        for message in snapshot['error_samples']:
            if message not in result['error_samples'] and len(result['error_samples']) < 5:
                result['error_samples'].append(message)
    return result

def _percentiles(snapshot):
    if sum(snapshot['histogram'].values()) != snapshot['completed']:
        return {f"p{percentile}_seconds": None for percentile in [50, 95, 99]}
    result = {}
    for percentile in [50, 95, 99]:
        target = math.ceil(snapshot['completed'] * percentile / 100)
        count = 0
        result[f"p{percentile}_seconds"] = None
        for bucket, frequency in sorted(snapshot['histogram'].items(), key=lambda item: int(item[0])):
            count += frequency
            if count >= target:
                result[f"p{percentile}_seconds"] = min(HISTOGRAM_BASE ** (int(bucket) + 1) / 1e6, snapshot['max_latency_seconds'])
                break
    return result

# Python callers and process entry point

class _Statistics:
    def __init__(self):
        self.lock = threading.Lock()
        self.data = _summarize([])

    def start(self):
        with self.lock:
            self.data['submitted'] += 1
            self.data['outstanding'] += 1

    def finish(self, duration, success, category, message):
        bucket = min(2047, max(0, int(math.log(max(duration * 1e6, 1), HISTOGRAM_BASE))))
        with self.lock:
            data = self.data
            data['completed'] += 1
            data['outstanding'] -= 1
            data['succeeded' if success else 'failed'] += 1
            data['max_latency_seconds'] = max(data['max_latency_seconds'], duration)
            data['latency_sum_seconds'] += duration
            data['histogram'][bucket] = data['histogram'].get(bucket, 0) + 1
            if not success:
                data['errors'][category] = data['errors'].get(category, 0) + 1
                if len(data['error_samples']) < 3 and message not in data['error_samples']:
                    data['error_samples'].append(message[:1000])

    def snapshot(self, histogram=True):
        with self.lock:
            result = dict(self.data)
            result['histogram'] = dict(self.data['histogram']) if histogram else {}
            result['errors'] = dict(self.data['errors'])
            result['error_samples'] = list(self.data['error_samples'])
            return result

def _failure_category(message):
    for name in ["ConnectTimeout", "ReadTimeout", "WriteTimeout", "PoolTimeout", "ConnectError", "ReadError", "WriteError", "RemoteProtocolError", "LocalProtocolError"]:
        if name in message:
            return name
    if "Timeout" in message or "timeout" in message:
        return "CompletionTimeout"
    return "CompletionFailure"

def _caller(client, statistics, config, phase, start, gate, stop):
    prompt = "x" * config['prompt_bytes']
    expected = "x" * config['response_bytes']
    gate.wait()
    deadline = start.value + phase['duration']
    attempted = 0
    while not stop.is_set():
        stamp = time.monotonic()
        if (phase['probe'] and attempted == 1) or (not phase['probe'] and stamp >= deadline):
            break
        statistics.start()
        attempted += 1
        try:
            success, message, response = client.prompt(text=prompt, reset_context=True, response_type="text")
            if success and (not isinstance(response, dict) or response.get('text') != expected):
                success, message = False, "Completion text does not match the fake endpoint response."
            category = _failure_category(message) if not success else ""
        except Exception as exc:
            success, category, message = False, type(exc).__name__, repr(exc)
        statistics.finish(time.monotonic() - stamp, success, category, message)

def _worker_main(index, connection, config, gate, start, stop):
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    os.environ['SSL_CERT_FILE'] = config['ca_file']
    for name in list(os.environ):
        if name.lower() in ["http_proxy", "https_proxy", "all_proxy"]:
            del os.environ[name]
    os.environ['NO_PROXY'] = "*"
    stack_path = os.path.join(config['run_directory'], f"worker-{index}-stacks.txt")
    with open(stack_path, "w", encoding="utf-8") as stacks:
        faulthandler.enable(file=stacks)
        try:
            settings = {
                'logger_mute': config['severity'] == "off",
                'logger_severity': config['severity'],
                'http_use_http2': config['http'] == "h2"
            }
            settings.update(config['pool_settings'])
            success, message = nimbro_api.set_settings(settings)
            if not success:
                raise UnrecoverableError(message)
            clients = [nimbro_api.openai.ChatCompletions(
                endpoint={
                    'name': "bandwidth",
                    'api_flavor': "openai",
                    'api_url': config['endpoint'],
                    'key_type': "plain",
                    'key_value': ""
                },
                model="fake-model", stream=False, validate_model=False, correction=False,
                retry=0, parser=[],
                timeout_connect=config['request_timeout'], timeout_read=config['request_timeout'],
                timeout_completion=config['request_timeout']
            ) for _ in range(config['threads_per_process'])]
            connection.send({
                'kind': "ready",
                'worker': index,
                'pid': os.getpid()
            })
            while True:
                phase = connection.recv()
                if phase['kind'] == "shutdown":
                    break
                local_gate = threading.Event()
                local_stop = threading.Event()
                statistics = [_Statistics() for _ in clients]
                threads = [threading.Thread(target=_caller, args=(client, stat, config, phase, start, local_gate, local_stop), daemon=True) for client, stat in zip(clients, statistics)]
                for thread in threads:
                    thread.start()
                connection.send({'kind': "prepared", 'worker': index})
                gate.wait()
                if stop.is_set():
                    local_stop.set()
                cpu_start = time.process_time()
                local_gate.set()
                last_report, interrupted = 0.0, False
                while any(thread.is_alive() for thread in threads):
                    if not interrupted and stop.is_set():
                        local_stop.set()
                        interrupted = True
                        faulthandler.dump_traceback(file=stacks, all_threads=True)
                        for client in clients:
                            threading.Thread(target=client.interrupt, daemon=True).start()
                    if time.monotonic() - last_report >= config['report_interval']:
                        connection.send({
                            'kind': "progress",
                            'worker': index,
                            'statistics': _summarize([stat.snapshot(histogram=False) for stat in statistics])
                        })
                        last_report = time.monotonic()
                    time.sleep(0.02)
                connection.send({
                    'kind': "done",
                    'worker': index,
                    'statistics': _summarize([stat.snapshot() for stat in statistics]),
                    'cpu_seconds': time.process_time() - cpu_start
                })
        except (EOFError, BrokenPipeError):
            pass
        except Exception as exc:
            connection.send({'kind': "error", 'worker': index, 'message': repr(exc)})
        finally:
            faulthandler.disable()
            connection.close()

# command line

def _positive_int(value):
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError(f"Expected a positive integer but got '{value}'.")
    return number

def _nonnegative_float(value):
    number = float(value)
    if not math.isfinite(number) or number < 0:
        raise argparse.ArgumentTypeError(f"Expected a finite nonnegative number but got '{value}'.")
    return number

def _positive_float(value):
    number = _nonnegative_float(value)
    if number == 0:
        raise argparse.ArgumentTypeError(f"Expected a positive number but got '{value}'.")
    return number

def _optional_int(value):
    if value == "default":
        return value
    if value.lower() == "none":
        return None
    number = int(value)
    if number < 0:
        raise argparse.ArgumentTypeError(f"Expected a nonnegative integer or 'none' but got '{value}'.")
    return number

def _default_processes():
    if hasattr(os, "process_cpu_count"):
        return os.process_cpu_count() or 1
    if hasattr(os, "sched_getaffinity"):
        return len(os.sched_getaffinity(0)) or 1
    return os.cpu_count() or 1

def _severity(value):
    return int(value) if value.isdigit() else value

def _arguments():
    parser = argparse.ArgumentParser(description="Compare a native Go client and ChatCompletions against a local fake HTTPS endpoint.", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--processes", type=_positive_int, help="Spawned Python workers. If omitted, use one per available logical CPU.")
    parser.add_argument("--threads-per-process", type=_positive_int, help="Persistent clients/callers per process.")
    parser.add_argument("--duration", type=_positive_float, help="Measured seconds per client.")
    parser.add_argument("--warmup", type=_nonnegative_float, help="Warm-up duration in seconds per client. Zero still probes once per caller.")
    parser.add_argument("--prompt-bytes", type=_positive_int, help="ASCII prompt content bytes.")
    parser.add_argument("--response-bytes", type=_positive_int, help="ASCII completion content bytes.")
    parser.add_argument("--http", choices=["h1", "h2"], help="Required negotiated protocol. Both use TLS.")
    parser.add_argument("--hold-ms", type=_nonnegative_float, help="Server delay after receiving the body.")
    parser.add_argument("--h2-max-streams", type=_positive_int, help="Server stream limit per HTTP/2 connection.")
    parser.add_argument("--max-keepalive-connections", type=_optional_int, help="NimbRo idle connections per process. Use 'none' for unlimited connections or 'default' to preserve package settings.")
    parser.add_argument("--request-timeout", type=_positive_float, help="Connect/read/completion timeout seconds.")
    parser.add_argument("--severity", type=_severity, choices=["debug", "info", "warn", "error", "fatal", "off", 0, 10, 20, 30, 40, 50], metavar="{debug,info,warn,error,fatal,off}", help="Logger severity for underlying nimbro_api logs and benchmark error diagnostics. Benchmark statistics are always emitted.")
    parser.set_defaults(**{name: parameter.default for name, parameter in signature(benchmark).parameters.items()})
    return vars(parser.parse_args())

def _admin_request(url, method="GET"):
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    request = urllib.request.Request(url, data=b"" if method == "POST" else None, method=method)
    with opener.open(request, timeout=3) as response:
        return json.load(response)

def _drain_server(config):
    deadline = time.monotonic() + config['drain_timeout']
    while True:
        metrics = _admin_request(config['admin_url'] + "/metrics")
        if not metrics['active']:
            return metrics
        if time.monotonic() >= deadline:
            raise TimeoutError("Server did not drain after the client stopped submitting.")
        time.sleep(0.02)

class _Server:
    def __init__(self, config):
        self.config = config
        self.process = None
        self.log = None

    def start(self):
        config = self.config
        directory = Path(config['run_directory'])
        source = Path(__file__).parent / "benchmark_server"
        binary = directory / ("server.exe" if os.name == "nt" else "server")
        go = shutil.which("go")
        if go is None:
            raise RuntimeError("Install Go 1.26+ and put its bin directory on PATH.")
        _report("Building server and reference client.")
        with open(directory / "build.log", "w", encoding="utf-8") as build_log:
            subprocess.run([go, "build", "-o", str(binary), "."], cwd=source, check=True, stdout=build_log, stderr=subprocess.STDOUT, timeout=config['startup_timeout'])
        config['binary'] = str(binary.resolve())
        readiness = directory / "server.json"
        self.log = open(directory / "server.log", "w", encoding="utf-8")
        with open(readiness, "w", encoding="utf-8") as output:
            self.process = subprocess.Popen([
                str(Path(binary).resolve()), "--cert-dir", str(directory / "tls"),
                "--response-bytes", str(config['response_bytes']), "--hold-ms", str(config['hold_ms']),
                "--h2-max-streams", str(config['h2_max_streams'])
            ], stdout=output, stderr=self.log)
        deadline = time.monotonic() + config['startup_timeout']
        while time.monotonic() < deadline:
            if self.process.poll() is not None:
                raise RuntimeError("Fake endpoint exited. See server diagnostics below.")
            try:
                info = json.loads(readiness.read_text(encoding="utf-8"))
            except (ValueError, OSError):
                time.sleep(0.05)
                continue
            for key in ['endpoint', 'admin_url', 'ca_file']:
                config[key] = info[key]
            config['response_body_bytes'] = info['config']['response_body_bytes']
            _admin_request(config['admin_url'] + "/metrics")
            return
        raise TimeoutError("Fake endpoint did not become ready.")

    def close(self):
        if self.process is not None:
            if self.process.poll() is None:
                try:
                    _admin_request(self.config['admin_url'] + "/shutdown", "POST")
                    self.process.wait(timeout=7)
                except (OSError, ValueError, TypeError, subprocess.TimeoutExpired):
                    self.process.kill()
                    self.process.wait()
        if self.log is not None:
            self.log.close()

class _Workers:
    def __init__(self, config):
        self.config = config
        self.context = multiprocessing.get_context("spawn")
        self.gate = self.context.Event()
        self.start_time = self.context.Value("d", 0.0)
        self.stop = self.context.Event()
        self.processes = []
        self.connections = []
        self.last_result = None
        self.units = config['processes']
        self.cpu_source = "process CPU seconds"

    def start(self):
        for index in range(self.config['processes']):
            parent, child = self.context.Pipe()
            process = self.context.Process(target=_worker_main, args=(index, child, self.config, self.gate, self.start_time, self.stop))
            process.start()
            child.close()
            self.connections.append(parent)
            self.processes.append(process)
        ready = self.await_kind("ready")
        self.config['worker_pids'] = [item['pid'] for item in ready]

    def receive(self, timeout):
        messages = []
        for connection in wait(self.connections, timeout=timeout):
            try:
                message = connection.recv()
            except EOFError as exc:
                raise RuntimeError("Worker exited before finishing. See diagnostics below.") from exc
            if message['kind'] == "error":
                raise RuntimeError(f"Worker '{message['worker']}': {message['message']}")
            messages.append(message)
        for process in self.processes:
            if process.exitcode is not None and process.exitcode != 0:
                raise RuntimeError(f"Worker PID '{process.pid}' exited with code '{process.exitcode}'.")
        return messages

    def await_kind(self, kind):
        received = {}
        deadline = time.monotonic() + self.config['startup_timeout']
        while len(received) < len(self.processes):
            if time.monotonic() > deadline:
                raise TimeoutError(f"Workers did not reach '{kind}'.")
            for message in self.receive(0.1):
                if message['kind'] == kind:
                    received[message['worker']] = message
        return [received[i] for i in range(len(self.processes))]

    def prepare(self, duration):
        self.gate.clear()
        self.stop.clear()
        for connection in self.connections:
            connection.send({'kind': "phase", 'duration': duration, 'probe': duration == 0})
        self.await_kind("prepared")

    def begin(self, start):
        self.start_time.value = start
        self.gate.set()

    def close(self):
        self.stop.set()
        self.gate.set()
        for connection in self.connections:
            try:
                connection.send({'kind': "shutdown"})
            except (BrokenPipeError, OSError):
                pass
        deadline = time.monotonic() + 7
        while any(process.is_alive() for process in self.processes) and time.monotonic() < deadline:
            for connection in wait(self.connections, timeout=0.1) if self.connections else []:
                try:
                    connection.recv()
                except (EOFError, OSError):
                    pass
        for process in self.processes:
            if process.is_alive():
                process.terminate()
        for process in self.processes:
            process.join(timeout=2)
            if process.is_alive():
                process.kill()
                process.join(timeout=2)
        for connection in self.connections:
            connection.close()

class _NativeClient:
    def __init__(self, config):
        self.config = config
        self.process = None
        self.log = None
        self.messages = queue.Queue()
        self.last_result = None
        self.units = 1
        self.cpu_source = "Go runtime CPU seconds estimate"

    def send(self, message):
        self.process.stdin.write(json.dumps(message) + "\n")
        self.process.stdin.flush()

    def start(self):
        self.log = open(Path(self.config['run_directory']) / "reference-client.log", "w", encoding="utf-8")
        self.process = subprocess.Popen([self.config['binary'], "client"], stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=self.log, text=True)
        threading.Thread(target=self.read_messages, daemon=True).start()
        self.send(self.config)
        self.await_kind("ready")

    def read_messages(self):
        try:
            for line in self.process.stdout:
                self.messages.put(json.loads(line))
        except (OSError, ValueError) as exc:
            self.messages.put({'kind': "error", 'message': repr(exc)})
        finally:
            self.messages.put({'kind': "error", 'message': "Reference client exited. See diagnostics below."})

    def receive(self, timeout):
        try:
            message = self.messages.get(timeout=timeout)
        except queue.Empty:
            return []
        if message['kind'] == "error":
            raise RuntimeError(message['message'])
        return [message]

    def await_kind(self, kind):
        deadline = time.monotonic() + self.config['startup_timeout']
        while time.monotonic() < deadline:
            for message in self.receive(0.1):
                if message['kind'] == kind:
                    return
        raise TimeoutError(f"Native client did not reach '{kind}'.")

    def prepare(self, duration):
        self.send({'kind': "phase", 'duration': duration, 'probe': duration == 0})
        self.await_kind("prepared")

    def begin(self, _start):
        self.send({'kind': "go"})

    def close(self):
        if self.process is not None:
            try:
                if self.process.poll() is None:
                    self.send({'kind': "shutdown"})
                    self.process.wait(timeout=2)
            except (OSError, subprocess.TimeoutExpired):
                self.process.kill()
                self.process.wait()
            self.process.stdin.close()
            self.process.stdout.close()
        if self.log is not None:
            self.log.close()

def _measure(workload, config, name, duration):
    workload.last_result = {'phase': name, 'status': "preparing"}
    workload.prepare(duration)
    _drain_server(config)
    baseline = _admin_request(config['admin_url'] + "/reset", "POST")
    start = time.monotonic()
    workload.begin(start)
    latest, finished, cpu_seconds = {}, set(), 0.0
    last_report, last_received = start, 0
    deadline = start + duration + config['drain_timeout']
    result = {'phase': name, 'status': "running", 'server_before': baseline}
    workload.last_result = result
    try:
        while len(finished) < workload.units:
            if time.monotonic() > deadline:
                raise TimeoutError(f"{name} exceeded its run/drain deadline.")
            for message in workload.receive(0.1):
                if message['kind'] in ["progress", "done"]:
                    latest[message['worker']] = message['statistics']
                if message['kind'] == "done":
                    finished.add(message['worker'])
                    cpu_seconds += message['cpu_seconds']
            now = time.monotonic()
            if now - last_report >= config['report_interval']:
                server = _admin_request(config['admin_url'] + "/metrics")
                result['server'] = server
                result['server_snapshot_seconds'] = now - start
                receipt_rate = (server['received'] - last_received) / (now - last_report)
                label = name.replace("/", " ").replace("warmup", "warm-up").capitalize()
                _report(f"{label}: '{now - start:.3f}s', '{server['active']}' active requests, '{receipt_rate:.0f}' received req/s")
                last_report, last_received = now, server['received']
    finally:
        elapsed = time.monotonic() - start
        statistics = _summarize(latest.values())
        result.update({
            'elapsed_seconds': elapsed,
            'statistics': statistics,
            'client_cpu_seconds': cpu_seconds,
            'client_cpu_source': workload.cpu_source,
            'finished_workers': len(finished),
            'latency': _percentiles(statistics),
            'successful_completions_per_second': statistics['succeeded'] / elapsed,
            'submission_seconds': min(duration, elapsed) if duration else elapsed,
            'drain_seconds': max(0.0, elapsed - duration) if duration else 0.0
        })
    server = _drain_server(config)
    result['server'] = server
    result['server_snapshot_seconds'] = time.monotonic() - start
    result['server_requests_per_second'] = server['received'] / elapsed
    result['server_runtime_cpu_seconds_estimate'] = max(0.0, server['runtime_cpu_seconds_estimate'] - baseline['runtime_cpu_seconds_estimate'])
    result['request_body_mib_per_second'] = server['request_bytes'] / elapsed / (1024 * 1024)
    result['response_body_mib_per_second'] = server['response_bytes'] / elapsed / (1024 * 1024)
    wrong = server['http1_requests' if config['http'] == "h2" else 'http2_requests']
    if wrong:
        raise RuntimeError(f"Protocol verification failed: requested '{config['http']}', server recorded h1='{server['http1_requests']}', h2='{server['http2_requests']}'.")
    if server['received'] == 0:
        raise RuntimeError("No complete request bodies reached the fake endpoint. See client failures and server counters.")
    if statistics['failed'] or server['received'] != statistics['submitted'] or server['completed'] != statistics['succeeded']:
        raise RuntimeError("Request failures or client/server accounting mismatch. See report.")
    result['status'] = "complete"
    return result

def _versions():
    result = {
        'python': platform.python_version(),
        'platform': platform.platform(),
        'cpu_count': os.cpu_count()
    }
    for name in ['nimbro-api', 'httpx', 'httpcore', 'h2']:
        try:
            result[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            result[name] = None
    return result

def _run_workload(name, workload, config, results):
    run = {'status': "starting"}
    results[name] = run
    try:
        workload.start()
        run['warmup'] = _measure(workload, config, f"{name}/warmup", config['warmup'])
        run['measurement'] = _measure(workload, config, f"{name}/measurement", config['duration'])
        run['status'] = "complete"
        _report(f"Completed {name} measurements.")
    except KeyboardInterrupt:
        run['status'] = "interrupted"
        raise
    except Exception as exc:
        run['status'], run['error'] = "failed", repr(exc)
    finally:
        if workload.last_result is not None:
            if run['status'] != "complete":
                workload.last_result['status'] = run['status']
            run[workload.last_result['phase'].split("/")[-1]] = workload.last_result
        workload.close()
        if isinstance(workload, _Workers):
            run['worker_exit_codes'] = [process.exitcode for process in workload.processes]
        elif workload.process is not None:
            run['exit_code'] = workload.process.returncode
        try:
            run['server_final'] = _admin_request(config['admin_url'] + "/metrics")
        except (OSError, ValueError) as exc:
            run['server_metrics_error'] = repr(exc)

def _report_results(name, result, logger):
    _report(f"{name.capitalize()}:")
    rate = None
    if 'error' in result:
        _report(f"{name.capitalize()}: {result['error']}", error_logger=logger)
    for phase_name in ['warmup', 'measurement']:
        phase = result.get(phase_name)
        if phase is None:
            continue
        if phase_name == "warmup" and phase['status'] == "complete":
            continue
        if phase_name == "warmup":
            _report("  Phase: warm-up")
        stats = phase.get('statistics')
        if stats:
            rate = phase['successful_completions_per_second']
            _report(f"  Requests: '{stats['submitted']}' submitted, '{stats['completed']}' completed")
            _report(f"  Successes: '{stats['succeeded']}'")
            _report(f"  Failures: '{stats['failed']}'")
            _report(f"  Outstanding: '{stats['outstanding']}'")
            success_rate = f"'{100 * stats['succeeded'] / stats['submitted']:.2f}%'" if stats['submitted'] else "unavailable"
            _report(f"  Success rate: {success_rate}")
            _report(f"  Elapsed: '{phase['elapsed_seconds']:.3f}s', drain: '{phase['drain_seconds']:.3f}s'")
            latencies = [f"{key.removesuffix('_seconds')} '{value:.3f}s'" if value is not None else f"{key.removesuffix('_seconds')} unavailable" for key, value in phase['latency'].items()]
            _report(f"  Latency: {', '.join(latencies)}, max '{stats['max_latency_seconds']:.3f}s'")
            _report(f"  Client CPU: '{phase['client_cpu_seconds']:.3f}s' ({phase['client_cpu_source']})")
        server = phase.get('server')
        if server:
            _report(f"  Peak server requests: '{server['peak_active']}'")
            if phase['status'] != "complete":
                _report(f"  Server requests: '{server['received']}' received, '{server['completed']}' completed")
            if server['read_errors'] or server['write_errors'] or server['canceled']:
                _report(f"{name.capitalize()} server errors: '{server['read_errors']}' read, '{server['write_errors']}' write, '{server['canceled']}' canceled", error_logger=logger)
        if 'response_body_mib_per_second' in phase:
            _report(f"  Body bandwidth: '{phase['request_body_mib_per_second']:.3f}' request MiB/s, '{phase['response_body_mib_per_second']:.3f}' response MiB/s")
    if 'worker_exit_codes' in result and any(code != 0 for code in result['worker_exit_codes']):
        _report(f"{name.capitalize()} worker exit codes: {result['worker_exit_codes']}", error_logger=logger)
    if 'exit_code' in result and result['exit_code'] != 0:
        _report(f"{name.capitalize()} client exit code: '{result['exit_code']}'", error_logger=logger)
    if 'server_metrics_error' in result:
        _report(f"{name.capitalize()} final server metrics unavailable: {result['server_metrics_error']}", error_logger=logger)
    _report(f"  Throughput: '{rate:.2f}' req/s" if rate is not None else "  Throughput: unavailable req/s")

def benchmark(*, processes=None, threads_per_process=10, duration=10.0, warmup=2.0, prompt_bytes=2048, response_bytes=2048, http="h1", hold_ms=0.0, h2_max_streams=250, max_keepalive_connections="default", request_timeout=10.0, severity="off"):
    """
    Compare a native Go client and ChatCompletions against a local fake HTTPS endpoint.

    Args:
        processes (int | None, optional):
            Number of spawned Python processes and native-client connection pools.
            Use `None` for one process per available logical CPU, respecting CPU affinity where supported.
            Defaults to `None`.
        threads_per_process (int, optional):
            Positive number of persistent ChatCompletions callers per process. Defaults to `10`.
        duration (float | int, optional):
            Positive measured submission duration in seconds for each client. Defaults to `10.0`.
        warmup (float | int, optional):
            Nonnegative warm-up duration in seconds per client. Zero sends one excluded probe per caller.
            Defaults to `2.0`.
        prompt_bytes (int, optional):
            Positive ASCII prompt content size in bytes. Defaults to `2048`.
        response_bytes (int, optional):
            Positive ASCII completion content size in bytes. Defaults to `2048`.
        http (str, optional):
            Required negotiated protocol, "h1" or "h2". Both use HTTPS. Defaults to "h1".
        hold_ms (float | int, optional):
            Nonnegative server delay after receiving a request body, in milliseconds. Defaults to `0.0`.
        h2_max_streams (int, optional):
            Server HTTP/2 stream limit per connection from `1` through `4294967295`. Defaults to `250`.
        max_keepalive_connections (int | str | None, optional):
            NimbRo idle connection limit per process. Use "default" to preserve package settings,
            `None` for unlimited idle connections, or a nonnegative integer. Defaults to "default".
        request_timeout (float | int, optional):
            Positive connect, read, and completion timeout in seconds. Defaults to `10.0`.
        severity (str, optional):
            Logger severity in ["debug", "info", "warn", "error", "fatal", "off"] for underlying `nimbro_api` logs.
            Use "off" to mute all underlying logs and benchmark error diagnostics.
            Benchmark statistics are always emitted. Defaults to "off".

    Raises:
        UnrecoverableError: If arguments are invalid or HTTP/2 dependencies are unavailable.

    Returns:
        int: The number of failed client runs, or 1 for setup failure or interruption.

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
    # parse arguments
    assert_type_value(obj=processes, type_or_value=[int, None], name="argument 'processes'")
    if processes is None:
        processes = _default_processes()
    config = {
        'processes': processes,
        'threads_per_process': threads_per_process,
        'duration': duration,
        'warmup': warmup,
        'prompt_bytes': prompt_bytes,
        'response_bytes': response_bytes,
        'http': http,
        'hold_ms': hold_ms,
        'h2_max_streams': h2_max_streams,
        'request_timeout': request_timeout,
        'pool_settings': {},
        'startup_timeout': 300.0,
        'drain_timeout': 0.0,
        'report_interval': 1.0,
        'admin_url': None
    }
    for name in ['processes', 'threads_per_process', 'prompt_bytes', 'response_bytes', 'h2_max_streams']:
        value = config[name]
        assert_type_value(obj=value, type_or_value=int, name=f"argument '{name}'")
        assert_log(expression=value > 0, message=f"Expected argument '{name}' to be positive but got '{value}'.")
    assert_log(expression=h2_max_streams <= 4294967295, message=f"Expected argument 'h2_max_streams' to be at most '4294967295' but got '{h2_max_streams}'.")
    for name in ['duration', 'warmup', 'hold_ms', 'request_timeout']:
        value = config[name]
        assert_type_value(obj=value, type_or_value=[float, int], name=f"argument '{name}'")
        assert_log(expression=math.isfinite(value) and value >= 0, message=f"Expected argument '{name}' to be finite and nonnegative but got '{value}'.")
        if name in ['duration', 'request_timeout']:
            assert_log(expression=value > 0, message=f"Expected argument '{name}' to be positive but got '{value}'.")
    config['drain_timeout'] = request_timeout + 5.0
    assert_type_value(obj=http, type_or_value=["h1", "h2"], name="argument 'http'")
    assert_type_value(obj=severity, type_or_value=[str, int], name="argument 'severity'")
    config['severity'] = Logger(severity=severity).get_settings(name='severity')
    assert_type_value(obj=max_keepalive_connections, type_or_value=[int, None, "default"], name="argument 'max_keepalive_connections'")
    if isinstance(max_keepalive_connections, int):
        assert_log(expression=max_keepalive_connections >= 0, message=f"Expected argument 'max_keepalive_connections' to be nonnegative but got '{max_keepalive_connections}'.")
    if max_keepalive_connections != "default":
        config['pool_settings']['http_max_keepalive_connections'] = max_keepalive_connections
    assert_log(expression=http != "h2" or importlib.util.find_spec("h2") is not None, message="HTTP/2 requires h2: install 'httpx[http2]'.")

    # set logger
    global REPORT_FILE, LOGGER
    logger_settings = nimbro_api.get_settings()
    pool_names = ['http_max_connections', 'http_max_keepalive_connections', 'http_keepalive_expiry']
    original_pool_settings = {name: logger_settings[name] for name in pool_names}
    config['pool_settings'] = {
        **original_pool_settings,
        **config['pool_settings']
    }
    logger_settings['logger_mute'] = False
    LOGGER = Logger(severity="info", name="Benchmark", core_settings=logger_settings)
    error_logger = Logger(severity=config['severity'], name="Benchmark", core_settings=logger_settings)
    output = Path(nimbro_api.__path__[0]) / "test" / "benchmark.txt"
    with open(output, "w", encoding="utf-8") as results_file, tempfile.TemporaryDirectory(prefix="nimbro-bandwidth-") as temporary:
        REPORT_FILE = results_file
        concurrency = config['processes'] * config['threads_per_process']
        protocol = "HTTP/2" if config['http'] == "h2" else "HTTP/1.1"
        server_details = f"Server: {protocol}, '{config['hold_ms'] / 1000:.3f}s' delay"
        if config['http'] == "h2":
            server_details += f", '{config['h2_max_streams']}' streams per connection"
        config['run_directory'] = temporary
        results = {}
        server = _Server(config)
        exit_code = 1
        try:
            success, message = nimbro_api.set_settings(settings=config['pool_settings'], mute=True)
            assert_log(expression=success, message=message)
            config['pool_settings'] = {name: nimbro_api.get_settings(name=name) for name in pool_names}
            server.start()
            server_config = _admin_request(config['admin_url'] + "/metrics")['config']
            _report("Environment:")
            _report(f"  Concurrency: '{config['processes']}' process{'' if config['processes'] == 1 else 'es'}, '{config['threads_per_process']}' caller{'' if config['threads_per_process'] == 1 else 's'} per process, '{concurrency}' total")
            _report(f"  Timing: '{config['warmup']:.3f}s' warm-up, '{config['duration']:.3f}s' measurement, '{config['request_timeout']:.3f}s' timeout")
            _report(f"  Payload: '{config['prompt_bytes']}' prompt bytes, '{config['response_bytes']}' completion bytes")
            _report(f"  {server_details}")
            versions = _versions()
            _report(f"  Python '{versions['python']}', '{versions['cpu_count']}' logical CPUs, {versions['platform']}")
            _report(f"  nimbro-api '{versions['nimbro-api']}', httpx '{versions['httpx']}', httpcore '{versions['httpcore']}', h2 '{versions['h2']}'")
            _report(f"  Go '{server_config['go_version']}', GOMAXPROCS '{server_config['gomaxprocs']}'")
            pool = config['pool_settings']
            _report(f"  HTTP settings: http_max_connections={pool['http_max_connections']}, http_max_keepalive_connections={pool['http_max_keepalive_connections']}, http_keepalive_expiry={pool['http_keepalive_expiry']}")
            _run_workload("reference", _NativeClient(config), config, results)
            if server.process.poll() is not None:
                raise RuntimeError("Server exited during the reference run.")
            _run_workload("package", _Workers(config), config, results)
            if all(results[name]['status'] == "complete" for name in ['reference', 'package']):
                exit_code = 0
        except KeyboardInterrupt:
            exit_code = 130
            _report("Interrupted by user.")
        except Exception as exc:
            _report(f"Benchmark failed: {exc}", error_logger=error_logger)
        finally:
            server.close()
            success, message = nimbro_api.set_settings(settings=original_pool_settings, mute=True)
            assert_log(expression=success, message=message)
            for name, result in results.items():
                _report_results(name, result, error_logger)
            if all(results.get(name, {}).get('measurement', {}).get('statistics', {}).get('succeeded', 0) > 0 for name in ['reference', 'package']):
                percentage = 100 * results['package']['measurement']['successful_completions_per_second'] / results['reference']['measurement']['successful_completions_per_second']
                _report(f"Relative package throughput: '{percentage:.2f}%'")
            for path in sorted(Path(temporary).iterdir()):
                if path.suffix in [".log", ".txt"] and path.stat().st_size:
                    _report(f"Diagnostics ({path.name}):\n{path.read_text(encoding='utf-8', errors='replace').rstrip()}", error_logger=error_logger)
            REPORT_FILE = None
    LOGGER = None
    if exit_code == 0:
        return 0
    return max(1, sum(result['status'] != "complete" for result in results.values()))

if __name__ == "__main__":
    sys.exit(benchmark(**_arguments()))
