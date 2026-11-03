"""One resident ROCm model; per-image harness processes communicate via Unix IPC."""
import json
import os
from pathlib import Path
import signal
import socket
import sys
import time
from PIL import Image, ImageDraw, ImageFont
from competition.contract import SOCKET_PATH, atomic_json, hash_file, receive, send
from competition.engine import RocmOCR, load_image


def main():
    os.umask(0o077)
    root = Path(SOCKET_PATH).parent
    root.mkdir(parents=True, exist_ok=True)
    ready_file = root / "ready.json"
    ready_file.unlink(missing_ok=True)
    Path(SOCKET_PATH).unlink(missing_ok=True)
    started = time.perf_counter()
    engine = RocmOCR()
    warmup = Image.new("RGB", (720, 200), "white")
    ImageDraw.Draw(warmup).text((35, 60), "Warmup", font=ImageFont.load_default(size=48), fill="black")
    warmup_budget = 600 - (time.perf_counter() - started) - 3
    if warmup_budget <= 4:
        raise RuntimeError("Model loading exhausted the startup budget")
    # First-kernel overhead belongs to startup, not the 30-second image budget.
    engine.transcribe(warmup, budget_seconds=warmup_budget)
    startup = time.perf_counter() - started
    if startup >= 600:
        raise RuntimeError("Model initialization/warmup exceeded the startup budget")
    description = {**engine.describe(), "model_loads": 1, "startup_seconds": startup, "pid": os.getpid(),
                   "ready_monotonic_seconds": time.monotonic()}
    stopped = False
    def stop(*_):
        nonlocal stopped
        stopped = True
    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    evidence = Path(os.environ.get("VISIONFORGE_EVIDENCE_DIR", "/app/evidence"))
    evidence.mkdir(parents=True, exist_ok=True)
    try:
        with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as listener:
            listener.bind(SOCKET_PATH)
            os.chmod(SOCKET_PATH, 0o600)
            listener.listen(4)
            listener.settimeout(.5)
            atomic_json(ready_file, description)
            print(json.dumps({"event": "ready", **description}), flush=True)
            while not stopped:
                try:
                    connection, _ = listener.accept()
                except socket.timeout:
                    continue
                with connection:
                    connection.settimeout(28)
                    request_id = None
                    try:
                        request = receive(connection)
                        request_id = request.get("request_id")
                        if not isinstance(request_id, str) or len(request_id) > 64:
                            raise ValueError("Invalid request ID")
                        if request.get("operation") == "health":
                            send(connection, {"ok": True, "request_id": request_id, "ready": True,
                                              **description, "memory": engine.memory(), "checked_monotonic_seconds": time.monotonic()})
                            continue
                        if request.get("operation") != "ocr" or not isinstance(request.get("input_image"), str):
                            raise ValueError("Unsupported worker request")
                        path = Path(request["input_image"]).resolve(strict=True)
                        if not path.is_file():
                            raise ValueError("Input must be a regular image file")
                        inference_started = time.perf_counter()
                        image = load_image(path, engine.spec["max_pixels"])
                        text, metrics = engine.transcribe(image)
                        metrics.update({"worker_seconds": time.perf_counter() - inference_started,
                                        "input_sha256": hash_file(path), "request_id": request_id,
                                        "backend": "rocm", "model_loads": 1, "worker_pid": os.getpid()})
                        # Evidence has hashes/timing, no private image bytes or OCR text.
                        with (evidence / "inference.jsonl").open("a", encoding="utf-8") as stream:
                            stream.write(json.dumps(metrics, allow_nan=False) + "\n")
                        send(connection, {"ok": True, "request_id": request_id, "text": text, "metrics": metrics})
                    except Exception as exc:
                        try:
                            send(connection, {"ok": False, "request_id": request_id,
                                              "error": type(exc).__name__ + ": " + str(exc)})
                        except (OSError, ValueError):
                            pass
                        print(json.dumps({"event": "request_failed", "type": type(exc).__name__}), file=sys.stderr, flush=True)
    finally:
        ready_file.unlink(missing_ok=True)
        Path(SOCKET_PATH).unlink(missing_ok=True)


if __name__ == "__main__":
    main()
