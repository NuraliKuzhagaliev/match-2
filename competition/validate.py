"""Host-side Docker/AMD acceptance harness. Missing hardware never becomes PASS."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import time
import uuid
from competition.contract import (BASE_IMAGE, BASE_DIGEST, MAX_CONTAINER_BYTES,
    MIN_VRAM_BYTES, MAX_VRAM_BYTES, STARTUP_SECONDS, IMAGE_SECONDS, TOTAL_SECONDS,
    atomic_json, normalize_for_score, observed_peak_bytes, sampler_coverage)


def command(args, timeout=20, check=True):
    result = subprocess.run(args, capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=timeout)
    if check and result.returncode:
        raise RuntimeError("Command failed: " + " ".join(args[:3]) + "\n" + result.stderr[-4000:])
    return result


def inspect(image):
    return json.loads(command(["docker", "image", "inspect", image]).stdout)[0]


def image_gates(base, candidate):
    required = base.get("RootFS", {}).get("Layers", [])
    actual = candidate.get("RootFS", {}).get("Layers", [])
    base_layers = bool(required) and actual[:len(required)] == required
    size = candidate.get("Size")
    within_size = isinstance(size, int) and 0 < size <= MAX_CONTAINER_BYTES
    return base_layers, within_size


def load_cases(path):
    from PIL import Image
    cases = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(cases, list) or len(cases) != 10:
        raise ValueError("Full acceptance requires exactly ten local validation images")
    seen = set()
    formats = set()
    for case in cases:
        name = case.get("file")
        if not isinstance(name, str) or Path(name).name != name or name in seen:
            raise ValueError("Case filenames must be unique, flat local names")
        if not isinstance(case.get("expected"), str):
            raise ValueError("Each case needs an exact ground-truth string")
        if not (Path(path).parent / name).is_file():
            raise ValueError("Missing case image: " + name)
        with Image.open(Path(path).parent / name) as source:
            formats.add(source.format)
        seen.add(name)
    if not {"PNG", "JPEG", "TIFF"}.issubset(formats):
        raise ValueError("Validation cases must cover actual PNG, JPEG and TIFF content")
    return cases


def validate(image, cases_path, report_path, keep=False):
    report = {"checked_at": datetime.now(timezone.utc).isoformat(), "target_image": image,
              "required_base": BASE_IMAGE + "@" + BASE_DIGEST,
              "decision": "not_ready", "gates": [], "cases": []}
    required_gates = ("case_manifest", "docker_daemon", "container_images", "base_layer_identity",
                      "uncompressed_size", "amd_devices", "startup", "rocm_inference_runtime",
                      "independent_sampler_started", "independent_sampler_coverage", "vram", "single_model_load", "output_contract",
                      "per_image_runtime", "total_runtime", "local_exact_match")
    report["gates"] = [{"id": id, "mandatory": True, "status": "NOT_RUN",
                        "detail": "Requires the preceding prerequisites to pass"} for id in required_gates]
    def gate(id, status, detail):
        item = {"id": id, "mandatory": True, "status": status, "detail": detail}
        for index, previous in enumerate(report["gates"]):
            if previous["id"] == id:
                report["gates"][index] = item
                return
        report["gates"].append(item)
    name = "visionforge-check-" + uuid.uuid4().hex[:10]
    created = False
    try:
        cases = load_cases(cases_path)
        gate("case_manifest", "PASS", "Ten exact-ground-truth cases; synthetic validation is not hidden scoring")
        daemon = command(["docker", "info", "--format", "{{.ServerVersion}}"], timeout=12, check=False)
        if daemon.returncode:
            gate("docker_daemon", "BLOCKED", daemon.stderr[-4000:])
            return report
        gate("docker_daemon", "PASS", "Docker daemon answered")
        base, candidate = inspect(BASE_IMAGE + "@" + BASE_DIGEST), inspect(image)
        gate("container_images", "PASS", "Both the pinned base and candidate image are locally inspectable")
        layers, size = image_gates(base, candidate)
        report["image_id"] = candidate["Id"]
        report["uncompressed_bytes"] = candidate.get("Size")
        report["base_layer_diff_ids"] = base.get("RootFS", {}).get("Layers", [])
        report["candidate_layer_diff_ids"] = candidate.get("RootFS", {}).get("Layers", [])
        gate("base_layer_identity", "PASS" if layers else "FAIL", "Compared actual RootFS diff-ID prefix, not the image name/label")
        gate("uncompressed_size", "PASS" if size else "FAIL", {"bytes": candidate.get("Size"), "maximum": MAX_CONTAINER_BYTES})
        if not layers or not size:
            return report
        if not Path("/dev/kfd").exists() or not Path("/dev/dri").exists():
            gate("amd_devices", "BLOCKED", "Run this validation on the Linux AMD host with /dev/kfd and /dev/dri")
            return report
        gate("amd_devices", "PASS", "AMD device paths exist; actual ROCm use is checked after loading")
        startup_started = time.monotonic()
        command(["docker", "run", "-d", "--name", name, "--network", "none",
                 "--device", "/dev/kfd", "--device", "/dev/dri", "--group-add", "video",
                 "--shm-size", "2g", "-e", "HIP_VISIBLE_DEVICES=0",
                 "--mount", f"type=bind,source={Path(cases_path).resolve().parent},target=/app/input,readonly", image], timeout=30)
        created = True
        # Start before model readiness so startup is sampled as well as the ten calls.
        command(["docker", "exec", "-d", name, "python3", "-m", "competition.telemetry"])
        initial = None
        while time.monotonic() - startup_started <= STARTUP_SECONDS:
            if command(["docker", "inspect", "--format", "{{.State.Running}}", name]).stdout.strip() != "true":
                raise RuntimeError("Container exited during startup: " + command(["docker", "logs", "--tail", "30", name], check=False).stderr[-4000:])
            health = command(["docker", "exec", name, "python3", "-m", "competition.health"], timeout=6, check=False)
            if health.returncode == 0:
                initial = json.loads(health.stdout)
                break
            time.sleep(1)
        startup_seconds = time.monotonic() - startup_started
        report["startup_seconds"] = startup_seconds
        if not initial:
            gate("startup", "FAIL", "No ready, warmed model within 600 seconds")
            return report
        gate("startup", "PASS" if startup_seconds <= STARTUP_SECONDS else "FAIL", startup_seconds)
        runtime_ok = initial.get("backend") == "rocm" and bool(initial.get("hip_version")) and initial.get("model_loads") == 1
        gate("rocm_inference_runtime", "PASS" if runtime_ok else "FAIL", initial)
        if not runtime_ok:
            return report
        total_started = time.monotonic()
        telemetry_ready = False
        telemetry_started = time.monotonic()
        while time.monotonic() - telemetry_started < 20:
            if command(["docker", "exec", name, "test", "-f", "/app/evidence/telemetry-ready"], check=False).returncode == 0:
                telemetry_ready = True
                break
            time.sleep(.5)
        gate("independent_sampler_started", "PASS" if telemetry_ready else "FAIL", "Driver memory sampled by a separate ROCm process every three seconds")
        for case in cases:
            elapsed_total = time.monotonic() - total_started
            if elapsed_total >= TOTAL_SECONDS:
                gate("total_runtime", "FAIL", "Ten-image run exhausted its 600-second budget")
                break
            start = time.monotonic()
            row = {"file": case["file"], "expected": case["expected"]}
            try:
                call = command(["docker", "exec", name, "python3", "/app/app.py", "--input-image", "/app/input/" + case["file"]],
                               timeout=min(IMAGE_SECONDS, TOTAL_SECONDS - elapsed_total), check=False)
                row["seconds"] = time.monotonic() - start
                output_name = Path(case["file"]).stem + "_output.json"
                data = json.loads(command(["docker", "exec", name, "cat", "/app/output/" + output_name]).stdout) if call.returncode == 0 else None
                valid = isinstance(data, dict) and set(data) <= {"text", "confidence"} and isinstance(data.get("text"), str)
                if valid and "confidence" in data:
                    valid = type(data["confidence"]) in (int, float) and 0 <= data["confidence"] <= 1
                row.update({"exit_code": call.returncode, "output": data,
                            "contract_pass": bool(valid), "time_pass": row["seconds"] <= IMAGE_SECONDS,
                            "text_match": bool(valid) and normalize_for_score(data["text"]) == normalize_for_score(case["expected"])})
            except (subprocess.TimeoutExpired, RuntimeError, ValueError) as exc:
                row.update({"seconds": time.monotonic() - start, "contract_pass": False, "time_pass": False,
                            "text_match": False, "error": type(exc).__name__ + ": " + str(exc)})
            report["cases"].append(row)
        report["total_seconds"] = time.monotonic() - total_started
        final = json.loads(command(["docker", "exec", name, "python3", "-m", "competition.health"]).stdout)
        command(["docker", "exec", name, "python3", "-c", "from pathlib import Path; Path('/app/evidence/stop-sampling').touch()"], check=False)
        samples = command(["docker", "exec", name, "cat", "/app/evidence/vram-samples.jsonl"], check=False)
        measurements = [json.loads(line) for line in samples.stdout.splitlines() if line.strip()]
        inference = command(["docker", "exec", name, "cat", "/app/evidence/inference.jsonl"], check=False)
        metrics = [json.loads(line) for line in inference.stdout.splitlines() if line.strip()]
        report["vram_samples"] = measurements
        report["inference_evidence"] = metrics
        sampler_status = {}
        wait_started = time.monotonic()
        while time.monotonic() - wait_started < 8:
            result = command(["docker", "exec", name, "cat", "/app/evidence/sampler-status.json"], check=False)
            if result.returncode == 0:
                sampler_status = json.loads(result.stdout)
                if sampler_status.get("status") == "COMPLETE":
                    break
            time.sleep(.5)
        report["sampler_status"] = sampler_status
        gate("independent_sampler_coverage", "PASS" if sampler_status.get("status") == "COMPLETE" and sampler_coverage(
             measurements, initial["ready_monotonic_seconds"], final["checked_monotonic_seconds"]) else "FAIL",
             "Normal sampler completion and coverage of inference period required")
        peak = observed_peak_bytes([initial["memory"], final["memory"]] + [x["memory"] for x in metrics], measurements)
        report["peak_vram_bytes"] = peak
        gate("vram", "PASS" if measurements and metrics and MIN_VRAM_BYTES <= peak <= MAX_VRAM_BYTES else "FAIL",
             {"peak_bytes": peak, "minimum": MIN_VRAM_BYTES, "maximum_with_tolerance": MAX_VRAM_BYTES})
        gate("single_model_load", "PASS" if initial["pid"] == final["pid"] and final["model_loads"] == 1 else "FAIL", {"initial_pid": initial["pid"], "final_pid": final["pid"]})
        all_cases = len(report["cases"]) == 10
        gate("output_contract", "PASS" if all_cases and all(x["contract_pass"] for x in report["cases"]) else "FAIL", "Exact filename and JSON text schema for each image")
        gate("per_image_runtime", "PASS" if all_cases and all(x["time_pass"] for x in report["cases"]) else "FAIL", "Whole python3 /app/app.py invocation timed, not just generate()")
        if not any(g["id"] == "total_runtime" and g["status"] == "FAIL" for g in report["gates"]):
            gate("total_runtime", "PASS" if all_cases and report["total_seconds"] <= TOTAL_SECONDS else "FAIL", report["total_seconds"])
        gate("local_exact_match", "PASS" if all_cases and all(x["text_match"] for x in report["cases"]) else "FAIL", "Local synthetic exact-match only; hidden accuracy remains unknown")
        if all(g["status"] == "PASS" for g in report["gates"]):
            report["decision"] = "ready_for_submission_review"
    except (OSError, ValueError, KeyError, RuntimeError, subprocess.TimeoutExpired) as exc:
        gate("execution", "BLOCKED" if isinstance(exc, (OSError, subprocess.TimeoutExpired)) else "FAIL", type(exc).__name__ + ": " + str(exc))
    finally:
        if created:
            report["container_name"] = name
            if not keep:
                try:
                    command(["docker", "rm", "-f", name], check=False)
                except (OSError, subprocess.TimeoutExpired) as exc:
                    report["cleanup_error"] = str(exc)
        atomic_json(report_path, report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", required=True)
    parser.add_argument("--cases", type=Path, required=True)
    parser.add_argument("--report", type=Path, default=Path("evidence/competition/acceptance.json"))
    parser.add_argument("--keep-container", action="store_true")
    args = parser.parse_args()
    report = validate(args.image, args.cases, args.report, args.keep_container)
    print(json.dumps({"decision": report["decision"], "gates": report["gates"]}, indent=2))
    return 0 if report["decision"] == "ready_for_submission_review" else 2


if __name__ == "__main__":
    raise SystemExit(main())
