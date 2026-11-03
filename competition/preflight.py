"""Read-only prerequisite/registry checks. No image build, model weights or GPU rental."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import platform
import subprocess
import urllib.request
from competition.contract import BASE_IMAGE, BASE_DIGEST, atomic_json


def probe(args, timeout=12, full_stdout=False):
    try:
        r = subprocess.run(args, capture_output=True, timeout=timeout)
        encoding = "utf-16-le" if b"\x00" in r.stdout else "utf-8"
        output = r.stdout.decode(encoding, errors="replace")
        return {"exit_code": r.returncode, "output": output if full_stdout else output[-6000:],
                "stderr": r.stderr.decode("utf-8", errors="replace")[-6000:]}
    except (OSError, subprocess.TimeoutExpired) as exc:
        return {"status": "BLOCKED", "reason": type(exc).__name__ + ": " + str(exc)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, default=Path("evidence/competition/preflight.json"))
    args = parser.parse_args()
    spec = json.loads((Path(__file__).parent / "model-spec.json").read_text(encoding="utf-8"))
    report = {"checked_at": datetime.now(timezone.utc).isoformat(), "host": platform.platform(),
              "base": BASE_IMAGE, "required_digest": BASE_DIGEST,
              "amd_device_paths_present": Path("/dev/kfd").exists() and Path("/dev/dri").exists(),
              "docker_daemon": probe(["docker", "info", "--format", "{{.ServerVersion}}"]),
              "base_registry": probe(["docker", "manifest", "inspect", "--verbose", BASE_IMAGE], timeout=45, full_stdout=True),
              "decision": "not_ready", "model_weights_downloaded": False}
    registry = report["base_registry"]
    if registry.get("exit_code") == 0:
        manifest = json.loads(registry.pop("output"))
        report["base_manifest"] = manifest
        report["base_digest_matches"] = manifest["Descriptor"]["digest"] == BASE_DIGEST
        layers = manifest.get("SchemaV2Manifest", {}).get("layers", [])
        report["base_compressed_layer_bytes"] = sum(x["size"] for x in layers)
    try:
        url = f"https://huggingface.co/api/models/{spec['id']}/revision/{spec['revision']}?blobs=true"
        data = json.load(urllib.request.urlopen(url, timeout=45))
        weight = next(x for x in data["siblings"] if x["rfilename"] == spec["weight_file"])
        report["model_registry"] = {"revision": data["sha"], "license": data.get("cardData", {}).get("license"),
                                    "weight": weight, "source": url}
        report["model_spec_matches"] = data["sha"] == spec["revision"] and weight["lfs"]["sha256"] == spec["weight_sha256"] and weight["size"] == spec["weight_bytes"]
    except (OSError, ValueError, KeyError, StopIteration) as exc:
        report["model_registry"] = {"status": "BLOCKED", "reason": str(exc)}
    atomic_json(args.report, report)
    print(json.dumps({k: v for k, v in report.items() if k not in ("base_manifest", "model_registry")}, indent=2))
    # Metadata success is deliberately not hardware/runtime readiness.
    return 0 if report.get("base_digest_matches") and report.get("model_spec_matches") else 2


if __name__ == "__main__":
    raise SystemExit(main())
