"""Pure contract helpers: no torch import and no model answer substitution."""
import hashlib
import json
import os
from pathlib import Path
import socket
import struct
import tempfile

BASE_IMAGE = "rocm/pytorch:rocm10.0_ubuntu26.04_py3.14_pytorch_release_2.13.0"
BASE_DIGEST = "sha256:3174cb7061d94c427da96c0edef4adea28046fa3f3b2ff3948dc4e995665ff8c"
GIB = 1024 ** 3
MAX_CONTAINER_BYTES = 60 * GIB
MIN_VRAM_BYTES = GIB
MAX_VRAM_BYTES = int(48 * 1.01 * GIB)
STARTUP_SECONDS = 600
IMAGE_SECONDS = 30
TOTAL_SECONDS = 600
MAX_FRAME = 256 * 1024
SOCKET_PATH = os.environ.get("VISIONFORGE_SOCKET_PATH", "/run/visionforge/worker.sock")


def normalize_for_score(text):
    """Exactly the brief's comparison normalization, never an OCR correction."""
    return "".join(c for c in text.upper() if not c.isspace() and c not in "-.·_")


def observed_peak_bytes(memories, driver_samples):
    """Maximum of every actual observation; never confuse capacity with usage."""
    keys = ("allocated_bytes", "reserved_bytes", "driver_used_bytes", "max_allocated_bytes", "max_reserved_bytes")
    values = [m.get(k, 0) for m in memories for k in keys]
    values.extend(s.get("driver_used_bytes", 0) for s in driver_samples)
    return max((v for v in values if type(v) is int and v >= 0), default=0)


def sampler_coverage(samples, started, finished):
    """Require live three-second sampling across inference, with 1s scheduling slack."""
    times = [s.get("monotonic_seconds") for s in samples]
    if len(times) < 2 or not all(type(t) in (int, float) for t in times):
        return False
    gaps = [b-a for a,b in zip(times,times[1:])]
    return times[0] <= started and times[-1] >= finished-4 and all(0 < g <= 4 for g in gaps)


def transcription(text):
    if not isinstance(text, str) or len(text) > 16384:
        raise ValueError("Model transcription must be a bounded string")
    # Preserve recognized characters: no banner blacklist, O/0 or I/1 replacement.
    return " ".join(text.split())


def hash_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(2 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=".visionforge-", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(value, stream, ensure_ascii=False, allow_nan=False)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)


def recv_exact(connection, count):
    parts = []
    while count:
        part = connection.recv(count)
        if not part:
            raise ConnectionError("Worker disconnected before completing its response")
        parts.append(part)
        count -= len(part)
    return b"".join(parts)


def receive(connection):
    count = struct.unpack("!I", recv_exact(connection, 4))[0]
    if not 0 < count <= MAX_FRAME:
        raise ValueError("Invalid IPC frame size")
    value = json.loads(recv_exact(connection, count).decode("utf-8"))
    if not isinstance(value, dict):
        raise ValueError("IPC message must be a JSON object")
    return value


def send(connection, value):
    data = json.dumps(value, ensure_ascii=False, allow_nan=False).encode("utf-8")
    if not 0 < len(data) <= MAX_FRAME:
        raise ValueError("IPC response exceeds its bounded frame")
    connection.sendall(struct.pack("!I", len(data)) + data)


def rpc(request, timeout=27.0, socket_path=SOCKET_PATH):
    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as connection:
        connection.settimeout(timeout)
        connection.connect(str(socket_path))
        send(connection, request)
        response = receive(connection)
    if response.get("request_id") != request.get("request_id"):
        raise ValueError("Worker response ID does not match this request")
    if not response.get("ok"):
        raise RuntimeError(response.get("error", "GPU inference failed"))
    return response
