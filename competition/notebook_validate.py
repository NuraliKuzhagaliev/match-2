"""Real ROCm acceptance without Docker. Never passes container/submission gates."""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import uuid
from competition.contract import atomic_json, hash_file, normalize_for_score, observed_peak_bytes, sampler_coverage, rpc, MIN_VRAM_BYTES, MAX_VRAM_BYTES
from competition.validate import load_cases


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", required=True, type=Path)
    parser.add_argument("--cases", type=Path, default=Path("data/competition/manifest.json"))
    parser.add_argument("--report", type=Path, default=Path("evidence/competition/notebook-acceptance.json"))
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    run = root / "runs" / uuid.uuid4().hex[:10]
    run.mkdir(parents=True)
    evidence = run / "evidence"
    evidence.mkdir()
    outputs = run / "output"
    socket = run / "worker.sock"
    env = {**os.environ, "PYTHONPATH": str(root), "PYTHONUNBUFFERED": "1",
           "VISIONFORGE_MODEL_DIR": str(args.model_dir.resolve()),
           "VISIONFORGE_SOCKET_PATH": str(socket), "VISIONFORGE_EVIDENCE_DIR": str(evidence),
           "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1", "HF_HUB_DISABLE_TELEMETRY": "1",
           "HIP_VISIBLE_DEVICES": "0"}
    report = {"checked_at": datetime.now(timezone.utc).isoformat(), "kind": "direct_notebook",
              "decision": "not_ready", "submission_ready": False, "run_directory": str(run),
              "source_sha256": {p.name: hash_file(p) for p in (root / "competition").glob("*.py")},
              "cases": [], "gates": [{"id": "docker_build_layers_size", "status": "NOT_RUN"}]}
    worker = sampler = None
    def health():
        return rpc({"operation": "health", "request_id": uuid.uuid4().hex}, timeout=3, socket_path=str(socket))
    def gate(id, ok, detail):
        report["gates"].append({"id": id, "status": "PASS" if ok else "FAIL", "detail": detail})
    try:
        cases = load_cases(args.cases)
        with (run / "worker.log").open("w") as worker_log, (run / "sampler.log").open("w") as sampler_log:
            started = time.monotonic()
            sampler = subprocess.Popen([sys.executable, "-m", "competition.telemetry", "--output", str(evidence / "vram-samples.jsonl"),
                                        "--stop-file", str(evidence / "stop-sampling")], cwd=root, env=env, stdout=sampler_log, stderr=subprocess.STDOUT)
            worker = subprocess.Popen([sys.executable, "-m", "competition.worker"], cwd=root, env=env, stdout=worker_log, stderr=subprocess.STDOUT)
            initial = None
            while time.monotonic() - started < 600:
                if worker.poll() is not None:
                    raise RuntimeError("Worker exited; inspect " + str(run / "worker.log"))
                try:
                    initial = health()
                    break
                except (OSError, ValueError, RuntimeError, ConnectionError):
                    time.sleep(1)
            report["startup_seconds"] = time.monotonic() - started
            gate("startup", bool(initial) and report["startup_seconds"] <= 600, report["startup_seconds"])
            if not initial:
                raise RuntimeError("No warmed model within 600 seconds")
            report["runtime"] = initial
            gate("rocm_runtime", initial.get("backend") == "rocm" and bool(initial.get("hip_version")), initial)
            sampler_alive = sampler.poll() is None and (evidence / "telemetry-ready").exists()
            gate("independent_sampler_started", sampler_alive, "Separate ROCm sampler must be ready and alive")
            if not sampler_alive:
                raise RuntimeError("Independent sampler failed before inference")
            total_started = time.monotonic()
            for case in cases:
                if sampler.poll() is not None:
                    raise RuntimeError("Independent sampler stopped during inference")
                row = {"file": case["file"], "expected": case["expected"], "input_sha256": hash_file(args.cases.parent / case["file"])}
                call_started = time.monotonic()
                try:
                    remaining = 600 - (time.monotonic() - total_started)
                    if remaining <= 0:
                        raise RuntimeError("Total budget exhausted")
                    call = subprocess.run([sys.executable, "-m", "competition.app", "--input-image", str((args.cases.parent / case["file"]).resolve()),
                                           "--output-dir", str(outputs)], cwd=root, env=env, capture_output=True, text=True, timeout=min(30, remaining))
                    row["seconds"] = time.monotonic() - call_started
                    data = json.loads((outputs / (Path(case["file"]).stem + "_output.json")).read_text()) if call.returncode == 0 else None
                    valid = isinstance(data, dict) and set(data) == {"text"} and isinstance(data["text"], str)
                    row.update({"exit_code": call.returncode, "output": data, "contract_pass": bool(valid),
                                "time_pass": row["seconds"] <= 30, "text_match": bool(valid) and normalize_for_score(data["text"]) == normalize_for_score(case["expected"]),
                                "stderr": call.stderr[-2000:]})
                except (OSError, ValueError, subprocess.TimeoutExpired, RuntimeError) as exc:
                    row.update({"seconds": time.monotonic()-call_started,"contract_pass": False, "time_pass": False,
                                "text_match": False, "error": str(exc)})
                report["cases"].append(row)
                print(json.dumps({"event": "case", **row}, ensure_ascii=True), flush=True)
            report["total_seconds"] = time.monotonic() - total_started
            if sampler.poll() is not None:
                raise RuntimeError("Independent sampler stopped before inference completed")
            final = health()
            gate("single_model_load", initial["pid"] == final["pid"] and final["model_loads"] == 1, final["pid"])
            gate("output_contract", len(report["cases"]) == 10 and all(x["contract_pass"] for x in report["cases"]), "Ten CLI calls with actual model outputs")
            gate("per_image_runtime", all(x["time_pass"] for x in report["cases"]), "Whole CLI <=30 seconds")
            gate("total_runtime", report["total_seconds"] <= 600, report["total_seconds"])
            gate("local_exact_match", all(x["text_match"] for x in report["cases"]), "Synthetic local regression; not hidden contest accuracy")
            (evidence / "stop-sampling").touch()
            sampler.wait(timeout=8)
            report["vram_samples"] = [json.loads(x) for x in (evidence / "vram-samples.jsonl").read_text().splitlines() if x]
            log = evidence / "inference.jsonl"
            report["inference_evidence"] = [json.loads(x) for x in log.read_text().splitlines() if x] if log.exists() else []
            status = json.loads((evidence / "sampler-status.json").read_text())
            coverage = sampler.returncode == 0 and status.get("status") == "COMPLETE" and sampler_coverage(
                report["vram_samples"], initial["ready_monotonic_seconds"], final["checked_monotonic_seconds"])
            gate("independent_sampler_coverage", coverage, {"exit_code": sampler.returncode, "status": status})
            peak = observed_peak_bytes([initial["memory"], final["memory"]] +
                                      [x["memory"] for x in report["inference_evidence"]], report["vram_samples"])
            report["peak_vram_bytes"] = peak
            gate("vram", bool(report["vram_samples"]) and bool(report["inference_evidence"]) and MIN_VRAM_BYTES <= peak <= MAX_VRAM_BYTES, peak)
            if all(x["status"] == "PASS" for x in report["gates"] if x["id"] != "docker_build_layers_size"):
                report["decision"] = "direct_runtime_passed_container_unverified"
    except Exception as exc:
        report["error"] = type(exc).__name__ + ": " + str(exc)
        print(report["error"], file=sys.stderr, flush=True)
    finally:
        for process in (worker, sampler):
            if process is not None and process.poll() is None:
                process.terminate()
                try: process.wait(timeout=8)
                except subprocess.TimeoutExpired: process.kill(); process.wait()
        for filename in ("worker.log", "sampler.log"):
            path = run / filename
            report[filename] = path.read_text(errors="replace")[-12000:] if path.exists() else ""
        atomic_json(args.report, report)
    print(json.dumps({"decision": report["decision"], "gates": report["gates"], "report": str(args.report)}, indent=2), flush=True)
    return 0 if report["decision"] == "direct_runtime_passed_container_unverified" else 2


if __name__ == "__main__":
    raise SystemExit(main())
