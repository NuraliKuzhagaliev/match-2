"""Produce a bounded, source-only handoff archive, never a Docker-image claim."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import zipfile

ROOT = Path(__file__).resolve().parents[1]


def main():
    paths = [ROOT / ".dockerignore", ROOT / ".gitignore", Path(__file__).resolve(), ROOT / "scripts/verify_competition_image.py",
             ROOT / "scripts/submission_upload.py", ROOT / "scripts/prepare_submission.py"]
    paths += sorted(p for p in (ROOT / "competition").iterdir()
                    if p.is_file() and (p.suffix in {".py", ".json", ".txt", ".md", ".sh"} or p.name == "Dockerfile"))
    cases = json.loads((ROOT / "data/competition/manifest.json").read_text(encoding="utf-8"))
    paths += [ROOT / "data/competition/manifest.json"]
    for case in cases:
        if Path(case["file"]).name != case["file"]:
            raise ValueError("Unsafe fixture path")
        path = ROOT / "data/competition" / case["file"]
        if hashlib.sha256(path.read_bytes()).hexdigest() != case["sha256"]:
            raise ValueError("Fixture hash mismatch")
        paths.append(path)
    paths += sorted((ROOT / "tests/competition").glob("*.py"))
    paths += sorted((ROOT / "tests/submission").glob("*.py"))
    paths += [ROOT / "evidence/competition" / name for name in
              ("preflight.json", "local-acceptance.json", "contract-tests.txt",
               "regression-tests.txt", "readiness.md", "review.md", "packet-check.txt")]
    for directory in ("gpu-baseline-20261030", "gpu-final-20261101", "gpu-setup-20261101", "gpu-release-20261101"):
        paths += sorted(p for p in (ROOT / "evidence/competition" / directory).rglob("*")
                        if p.is_file() and not p.is_symlink())
    for name in ("amd-environment-20261030.json", "amd-final-20261101.png", "amd-release-20261101.png",
                 "contract-tests-20261101.txt", "regression-tests-20261101.txt",
                 "packet-check-20261101.txt", "tested-source-20261101.json",
                 "packet-tested-core-20261101.json",
                 "release-status-20261101.json", "image-verification.json", "local-acceptance-20261101.json",
                 "docker-build-release-20261101.txt", "base-pull-final-20261101.txt",
                 "image-verification-20261101.log", "local-validation-20261101.log"):
        path = ROOT / "evidence/competition" / name
        if path.is_file():
            paths.append(path)
    for name in ("brief-contract-20261101.json", "upload-tests-20261101.txt", "upload-preparation-20261101.json", "anonymous-mechanism-20261101.json"):
        path = ROOT / "evidence/competition" / name
        if path.is_file():
            paths.append(path)
    payload = {p.relative_to(ROOT).as_posix(): p.read_bytes() for p in paths}
    payload["README.md"] = (
        "# VisionForge AI — AMD AI League source packet\n\n"
        "Source-only packet; Docker image and model weights are separate.\n"
        "Direct AMD GPU evidence is included; this does not prove AMD container execution or submission readiness.\n\n"
        "Read competition/README.md. For independent container GPU validation on a compatible Linux AMD Docker host run:\n\n"
        "```bash\nbash competition/check.sh\n```\n\n"
        "Current evidence: evidence/competition/readiness.md.\n").encode("utf-8")
    manifest = {"created_at": datetime.now(timezone.utc).isoformat(), "artifact_kind": "source_only",
                "submission_ready": False, "files": {name: {"bytes": len(data),
                "sha256": hashlib.sha256(data).hexdigest()} for name, data in sorted(payload.items())}}
    payload["SOURCE-MANIFEST.json"] = json.dumps(manifest, ensure_ascii=False, indent=2).encode("utf-8")
    target = ROOT / "output/visionforge-amd-source.zip"
    target.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(target, "w", zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        for name, data in sorted(payload.items()):
            info = zipfile.ZipInfo(name, date_time=(2026, 10, 28, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = (0o100755 if name.endswith(".sh") else 0o100644) << 16
            archive.writestr(info, data)
    with zipfile.ZipFile(target) as archive:
        if archive.testzip() is not None:
            raise RuntimeError("Corrupt archive")
        for name, item in manifest["files"].items():
            if hashlib.sha256(archive.read(name)).hexdigest() != item["sha256"]:
                raise RuntimeError("Archive content mismatch: " + name)
    digest = hashlib.sha256(target.read_bytes()).hexdigest()
    target.with_suffix(".zip.sha256").write_text(digest + "  " + target.name + "\n", encoding="utf-8")
    print(json.dumps({"archive": str(target), "bytes": target.stat().st_size,
                      "files": len(payload), "sha256": digest, "kind": "source_only"}, indent=2))


if __name__ == "__main__":
    main()
