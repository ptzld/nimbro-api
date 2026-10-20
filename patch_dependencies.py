"""
Apply or reverse measured httpcore 1.0.9 performance optimizations.

Run with the Python interpreter of the environment to change:

    python patch_dependencies.py apply
    python patch_dependencies.py apply --http2
    python patch_dependencies.py status
    python patch_dependencies.py restore

Use --http2 consistently for apply, status and restore of that patch selection.
Backups and a hash manifest are stored outside the checkout, under
~/.cache/nimbro-api/dependency-patches, unless --backup-dir is supplied.
Start from stock httpcore; restore any separately applied experiments first.
Stop running clients before applying or restoring, then start fresh processes.
"""

import os
import sys
import json
import stat
import hashlib
import argparse
import tempfile
import importlib
from pathlib import Path
from importlib.metadata import version

def _digest(data):
    return hashlib.sha256(data).hexdigest()

_HTTP2_LIMITER = b'''class _NimbroStreamLimiter:
    def __init__(self, bound):
        self._limit = bound
        self._active = 0
        self._reading = False
        self._changed = AsyncEvent()

    async def acquire(self, receive=None):
        while self._active >= self._limit:
            # With no active streams, a queued request must read later SETTINGS.
            if self._limit == 0 and self._active == 0 and receive is not None and not self._reading:
                self._reading = True
                try:
                    await receive()
                finally:
                    self._reading = False
                    self._notify()
            else:
                await self._changed.wait()
        self._active += 1

    async def release(self):
        self._active -= 1
        self._notify()

    def set_limit(self, limit):
        self._limit = limit
        self._notify()

    def _notify(self):
        changed = self._changed
        self._changed = AsyncEvent()
        changed.set()


'''

def _specifications(http2=False):
    if version("httpcore") != "1.0.9":
        raise ValueError("Expected httpcore 1.0.9; no source changes made.")
    pool = importlib.import_module("httpcore._async.connection_pool")
    backend = importlib.import_module("httpcore._backends.anyio")
    specifications = [
        (
            Path(pool.__file__).resolve(),
            b"""                connection.is_idle()
                and len([connection.is_idle() for connection in self._connections])
                > self._max_keepalive_connections""",
            b"""                len(self._connections) > self._max_keepalive_connections
                and connection.is_idle()"""
        ),
        (
            Path(backend.__file__).resolve(),
            b"""            sock = self._stream.extra(anyio.abc.SocketAttribute.raw_socket, None)
            return is_socket_readable(sock)""",
            b"""            if not hasattr(self, "_nimbro_cached_socket"):
                self._nimbro_cached_socket = self._stream.extra(anyio.abc.SocketAttribute.raw_socket, None)
            return is_socket_readable(self._nimbro_cached_socket)"""
        )
    ]
    if http2:
        module = importlib.import_module("httpcore._async.http2")
        path = Path(module.__file__).resolve()
        specifications.extend([
            (path, b"from .._synchronization import AsyncLock, AsyncSemaphore, AsyncShieldCancellation",
             b"from .._synchronization import AsyncEvent, AsyncLock, AsyncShieldCancellation"),
            (path, b"class AsyncHTTP2Connection(AsyncConnectionInterface):",
             _HTTP2_LIMITER + b"class AsyncHTTP2Connection(AsyncConnectionInterface):"),
            (path, b'''                local_settings_max_streams = (
                    self._h2_state.local_settings.max_concurrent_streams
                )
                self._max_streams_semaphore = AsyncSemaphore(local_settings_max_streams)''',
             b'''                local_settings_max_streams = 1
                self._max_streams_semaphore = _NimbroStreamLimiter(local_settings_max_streams)'''),
            (path, b"\n        await self._max_streams_semaphore.acquire()\n",
             b"\n        await self._max_streams_semaphore.acquire(lambda: self._receive_events(request))\n"),
            (path, b'''        max_concurrent_streams = event.changed_settings.get(
            h2.settings.SettingCodes.MAX_CONCURRENT_STREAMS
        )
        if max_concurrent_streams:
            new_max_streams = min(
                max_concurrent_streams.new_value,
                self._h2_state.local_settings.max_concurrent_streams,
            )
            if new_max_streams and new_max_streams != self._max_streams:
                while new_max_streams > self._max_streams:
                    await self._max_streams_semaphore.release()
                    self._max_streams += 1
                while new_max_streams < self._max_streams:
                    await self._max_streams_semaphore.acquire()
                    self._max_streams -= 1''',
             b'''        new_max_streams = self._h2_state.remote_settings.max_concurrent_streams
        if new_max_streams != self._max_streams:
            self._max_streams = new_max_streams
            self._max_streams_semaphore.set_limit(self._max_streams)''')
        ])
    grouped = {}
    for path, old, new in specifications:
        grouped.setdefault(path, []).append((old, new))
    return list(grouped.items())

def _transform(original, changes):
    modified = original
    for old, new in changes:
        if modified.count(old) != 1 or new in modified:
            raise ValueError("Unexpected or already patched source. Restore separate experiments first.")
        modified = modified.replace(old, new, 1)
    return modified

def _write_source(path, data):
    mode = stat.S_IMODE(path.stat().st_mode)
    descriptor, name = tempfile.mkstemp(prefix=".nimbro-patch-", dir=str(path.parent))
    temporary = Path(name)
    try:
        with os.fdopen(descriptor, "wb") as file:
            os.chmod(temporary, mode)
            file.write(data)
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()

def _prepare(specifications, directory, action):
    manifest = directory / "manifest.json"
    if not manifest.exists():
        if action != "apply":
            raise ValueError("No patch backup exists for this environment. No changes made.")
        entries = []
        originals = []
        for index, (target, changes) in enumerate(specifications):
            original = target.read_bytes()
            modified = _transform(original, changes)
            compile(modified, str(target), "exec")
            entries.append({
                'target': str(target),
                'backup': f"{index}.original",
                'original_sha256': _digest(original),
                'patched_sha256': _digest(modified)
            })
            originals.append(original)
        directory.mkdir(parents=True, exist_ok=True, mode=0o700)
        if any(directory.iterdir()):
            raise ValueError(f"Backup directory is not empty: {directory}. No source changes made.")
        for entry, original in zip(entries, originals):
            with (directory / entry['backup']).open("xb") as file:
                file.write(original)
        with manifest.open("x", encoding="utf-8") as file:
            json.dump(entries, file, indent=4)
    entries = json.loads(manifest.read_text(encoding="utf-8"))
    if len(entries) != len(specifications):
        raise ValueError("Unexpected manifest. No changes made.")
    plan = []
    for index, (entry, (target, changes)) in enumerate(zip(entries, specifications)):
        if entry['target'] != str(target) or entry['backup'] != f"{index}.original":
            raise ValueError("Manifest belongs to different dependency files. No changes made.")
        original = (directory / entry['backup']).read_bytes()
        modified = _transform(original, changes)
        if _digest(original) != entry['original_sha256'] or _digest(modified) != entry['patched_sha256']:
            raise ValueError("Backup hash mismatch. No changes made.")
        current = target.read_bytes()
        if current not in [original, modified]:
            raise ValueError(f"Source changed outside this script: {target}. Refusing to overwrite it.")
        replacement = modified if action == "apply" else original
        compile(replacement, str(target), "exec")
        plan.append((target, current, replacement, modified))
    return plan

def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("action", choices=["apply", "restore", "status"])
    parser.add_argument("--http2", action="store_true", help="Also use the server-negotiated outgoing HTTP/2 stream limit; use the same flag for status/restore.")
    parser.add_argument("--backup-dir", type=Path, help="Backup directory; defaults to an environment-specific directory in ~/.cache/nimbro-api/dependency-patches.")
    arguments = parser.parse_args()
    try:
        specifications = _specifications(arguments.http2)
        identifier = _digest(str(specifications[0][0]).encode("utf-8"))[:16]
        if arguments.http2:
            identifier += "-http2"
        directory = arguments.backup_dir or Path.home() / ".cache/nimbro-api/dependency-patches" / identifier
        plan = _prepare(specifications, directory, arguments.action)
        for target, current, replacement, modified in plan:
            if arguments.action != "status" and current != replacement:
                if target.read_bytes() != current:
                    raise ValueError(f"Source changed during patching: {target}. Refusing to overwrite it.")
                _write_source(target, replacement)
                current = replacement
            state = "patched" if current == modified else "original"
            print(f"{state}: {target}")
        print(f"Backups: {directory.resolve()}")
        if arguments.action != "status":
            print("Start fresh Python processes to use the changed source files.")
    except (OSError, ValueError, KeyError, ImportError) as exc:
        parser.exit(1, f"{exc}\n")

if __name__ == "__main__":
    main()
