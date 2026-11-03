"""Independent driver-memory sampler. Writes measurements, never allocates padding."""
import argparse
import json
import os
from pathlib import Path
import time
from competition.engine import require_rocm
from competition.contract import atomic_json


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("/app/evidence/vram-samples.jsonl"))
    parser.add_argument("--stop-file", type=Path, default=Path("/app/evidence/stop-sampling"))
    args = parser.parse_args()
    import torch
    require_rocm(torch)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    status_file = args.output.parent / "sampler-status.json"
    count = 0
    atomic_json(status_file, {"status": "RUNNING", "pid": os.getpid(), "interval_seconds": 3})
    with args.output.open("w", encoding="utf-8") as stream:
        while not args.stop_file.exists():
            free, total = torch.cuda.mem_get_info(0)
            stream.write(json.dumps({"monotonic_seconds": time.monotonic(),
                                     "driver_used_bytes": total - free, "device_total_bytes": total,
                                     "hip_version": torch.version.hip}) + "\n")
            stream.flush()
            count += 1
            (args.output.parent / "telemetry-ready").touch()
            time.sleep(3.0)
    atomic_json(status_file, {"status": "COMPLETE", "pid": os.getpid(), "samples": count})


if __name__ == "__main__":
    main()
