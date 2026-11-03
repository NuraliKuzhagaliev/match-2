"""Verify a built image without claiming unavailable AMD container execution."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from competition.contract import BASE_IMAGE, BASE_DIGEST, atomic_json
from competition.validate import image_gates


def run(args, timeout=180):
    return subprocess.run(["docker", *args], capture_output=True, text=True,
                          encoding="utf-8", errors="replace", timeout=timeout)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", default="visionforge-amd:local")
    parser.add_argument("--gpu-report", type=Path, required=True)
    parser.add_argument("--report", type=Path, default=ROOT / "evidence/competition/image-verification.json")
    args = parser.parse_args()
    report = {"checked_at": datetime.now(timezone.utc).isoformat(), "image": args.image,
              "kind": "image_without_gpu_execution", "submission_ready": False, "gates": []}
    def gate(name, ok, detail):
        report["gates"].append({"id": name, "status": "PASS" if ok else "FAIL", "detail": detail})
    try:
        info = run(["info", "--format", '{"version":{{json .ServerVersion}},"driver_status":{{json .DriverStatus}}}'])
        if info.returncode:
            raise RuntimeError(info.stderr)
        report["docker"] = json.loads(info.stdout)
        import re
        version = tuple(int(x) for x in re.findall(r"\d+", report["docker"]["version"])[:2])
        supported = version >= (29, 8)
        gate("image_size_accounting_version", supported, report["docker"])
        if not supported:
            raise RuntimeError("This runner requires Docker 29.8+; older containerd inspect Size may count packed data only")
        report["size_method"] = "Docker inspect Size; containerd content+unpacked total is a conservative upper bound for unpacked size"
        reference = BASE_IMAGE + "@" + BASE_DIGEST
        base_call, candidate_call = run(["image", "inspect", reference]), run(["image", "inspect", args.image])
        if base_call.returncode or candidate_call.returncode:
            raise RuntimeError(base_call.stderr + candidate_call.stderr)
        base, candidate = json.loads(base_call.stdout)[0], json.loads(candidate_call.stdout)[0]
        layers, size = image_gates(base, candidate)
        report.update(image_id=candidate["Id"], uncompressed_bytes=candidate["Size"],
                      base_diff_ids=base["RootFS"]["Layers"], candidate_diff_ids=candidate["RootFS"]["Layers"])
        gate("base_layer_identity", layers, reference)
        gate("uncompressed_size", size, candidate["Size"])
        gpu = json.loads(args.gpu_report.read_text(encoding="utf-8"))
        code = """import json,hashlib
from pathlib import Path
from competition.contract import hash_file
spec=json.loads(Path('/app/competition/model-spec.json').read_text())
model=Path('/models/qwen3-vl-2b')
manifest=json.loads((model/'visionforge-manifest.json').read_text())
print(json.dumps({'source_sha256':{p.name:hash_file(p) for p in Path('/app/competition').glob('*.py')},
 'model_spec_sha256':hash_file('/app/competition/model-spec.json'),
 'model_manifest_sha256':hashlib.sha256(json.dumps(manifest,sort_keys=True,separators=(',',':')).encode()).hexdigest(),
 'model_metadata_sha256':{p.relative_to(model).as_posix():hash_file(p) for p in model.rglob('*')
   if p.is_file() and '.cache' not in p.parts and p.name not in (spec['weight_file'],'visionforge-manifest.json')},
 'entrypoint_matches':hash_file('/app/app.py')==hash_file('/app/competition/app.py'),
 'weight_sha256':hash_file(model/spec['weight_file']),
 'expected_weight_sha256':spec['weight_sha256'],
 'processor_build_check':json.loads(Path('/app/processor-check.json').read_text())}))
"""
        check = run(["run", "--rm", "--network", "none", "--entrypoint", "python3", args.image, "-c", code])
        if check.returncode:
            raise RuntimeError(check.stderr[-4000:])
        actual = json.loads(check.stdout)
        report["container_checks"] = actual
        gate("gpu_tested_source_identity", actual["source_sha256"] == gpu["source_sha256"], str(args.gpu_report))
        gate("gpu_tested_model_configuration_identity", all(actual[k] == gpu["runtime"][k] for k in
             ("model_spec_sha256", "model_manifest_sha256", "model_metadata_sha256")),
             "Model spec, checkpoint metadata, tokenizer and chat template must match the real GPU run")
        gate("cli_entrypoint_identity", actual["entrypoint_matches"], "/app/app.py")
        gate("weight_identity", actual["weight_sha256"] == actual["expected_weight_sha256"] == gpu["runtime"]["weight_sha256"], actual["weight_sha256"])
        gate("processor_build_check", actual["processor_build_check"].get("status") == "PASS", actual["processor_build_check"])
        processor = run(["run", "--rm", "--network", "none", "--entrypoint", "python3", args.image,
                         "-m", "competition.processor", "/models/qwen3-vl-2b"])
        gate("offline_processor_runtime", processor.returncode == 0, {"stdout": processor.stdout, "stderr": processor.stderr[-3000:]})
        # No AMD devices are passed: the real ROCm worker must reject this run.
        guard = run(["run", "--rm", "--network", "none", "--entrypoint", "python3", args.image,
                     "-m", "competition.worker"], timeout=60)
        gate("no_gpu_fails_closed", guard.returncode != 0 and "ROCm GPU unavailable" in guard.stderr,
             {"exit_code": guard.returncode, "stderr": guard.stderr[-3000:]})
    except Exception as exc:
        gate("execution", False, type(exc).__name__ + ": " + str(exc))
    report["image_checks_passed"] = bool(report["gates"]) and all(x["status"] == "PASS" for x in report["gates"])
    report["gates"].append({"id": "amd_container_inference", "status": "NOT_RUN",
                            "detail": "Separate Linux AMD Docker host required; direct notebook inference does not pass this gate"})
    atomic_json(args.report, report)
    print(json.dumps(report, indent=2))
    return 0 if report["image_checks_passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
