import contextlib
import re
import sys
import threading
from datetime import datetime, timezone
from pathlib import Path

VLLM_LOG = "vllm.log"
PROXY_LOG = "proxy.log"
TAIL_BYTES = 5 * 1024 * 1024
MAX_LOG_BYTES = 200 * 1024 * 1024
KEEP_LOG_BYTES = 150 * 1024 * 1024
TRIM_INTERVAL_S = 30
CHUNK = 8 * 1024 * 1024
MAX_DETAIL = 500
PROXY_MARK = "[proxy]"
MESSAGE = re.compile(r"maximum context length|\bERROR\b|\b\w*(?:Error|Exception)\b|out of memory", re.IGNORECASE)
STATUS = re.compile(r'" (?:4|5)\d\d\b')
SERVER_FAILURE = re.compile(r"Error code: 5\d\d|\b50[0-4]\b")


def read_tail(path, limit=TAIL_BYTES, start=0):
    path = Path(path)
    try:
        size = path.stat().st_size
        with path.open("rb") as handle:
            handle.seek(max(start, size - limit))
            return handle.read()
    except OSError:
        return b""


def copy_tail(source, target, limit=TAIL_BYTES):
    data = read_tail(source, limit)
    if data:
        Path(target).write_bytes(data)
    return bool(data)


def last_error_line(text):
    lines = [line for line in text.splitlines() if line.strip()]
    for line in reversed(lines):
        if not line.startswith("INFO") and MESSAGE.search(line):
            return line.strip()[:MAX_DETAIL]
    for line in reversed(lines):
        if STATUS.search(line):
            return line.strip()[:MAX_DETAIL]
    return None


class LogTrimmer:
    def __init__(self, paths, limit=MAX_LOG_BYTES, keep=KEEP_LOG_BYTES, interval=TRIM_INTERVAL_S):
        self._paths = [Path(path) for path in paths]
        self._limit = limit
        self._keep = keep
        self._interval = interval
        self._removed = {path: 0 for path in self._paths}
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread = None

    def removed(self, path):
        with self._lock:
            return self._removed.get(Path(path), 0)

    def trim_once(self):
        for path in self._paths:
            try:
                self._trim(path)
            except OSError:
                continue

    def _trim(self, path):
        size = path.stat().st_size
        if size <= self._limit:
            return
        with path.open("r+b") as handle:
            read_at = size - self._keep
            handle.seek(read_at)
            head = handle.read(CHUNK)
            newline = head.find(b"\n")
            skip = newline + 1 if newline >= 0 else 0
            read_at += skip
            handle.seek(read_at)
            write_at = 0
            while True:
                chunk = handle.read(CHUNK)
                if not chunk:
                    break
                read_at += len(chunk)
                handle.seek(write_at)
                handle.write(chunk)
                write_at += len(chunk)
                handle.seek(read_at)
            with self._lock:
                self._removed[path] += read_at - write_at
            handle.truncate(write_at)

    def _loop(self):
        while not self._stop.wait(self._interval):
            self.trim_once()

    def start(self):
        self.trim_once()
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop(self):
        self._stop.set()
        if self._thread is not None:
            self._thread.join()
        self.trim_once()


class UpstreamErrors:
    def __init__(self, path, trimmer=None):
        self._path = Path(path)
        self._trimmer = trimmer
        self._start = 0

    def _removed(self):
        return self._trimmer.removed(self._path) if self._trimmer is not None else 0

    def mark(self):
        try:
            self._start = self._path.stat().st_size + self._removed()
        except OSError:
            self._start = 0

    def last(self):
        data = read_tail(self._path, TAIL_BYTES, max(0, self._start - self._removed()))
        return last_error_line(data.decode("utf-8", errors="replace"))


def is_server_failure(text):
    return SERVER_FAILURE.search(text) is not None


class _Tee:
    def __init__(self, stream, handle):
        self._stream = stream
        self._handle = handle
        self._pending = ""
        self._lock = threading.Lock()

    def write(self, text):
        written = self._stream.write(text)
        with self._lock:
            *lines, self._pending = (self._pending + text).split("\n")
            kept = [line for line in lines if PROXY_MARK in line]
            if kept:
                stamp = datetime.now(timezone.utc).isoformat()
                self._handle.write("".join(f"{stamp} {line}\n" for line in kept))
                self._handle.flush()
        return written

    def __getattr__(self, name):
        return getattr(self._stream, name)


@contextlib.contextmanager
def capture_proxy_log(path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    previous = sys.stdout
    with path.open("a", encoding="utf-8") as handle:
        tee = _Tee(previous, handle)
        sys.stdout = tee
        try:
            yield path
        finally:
            if sys.stdout is tee:
                sys.stdout = previous
