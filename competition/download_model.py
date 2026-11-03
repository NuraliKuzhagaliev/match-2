"""Build-time public checkpoint download; no credentials or image uploads."""
import json
from pathlib import Path
import sys
from competition.contract import atomic_json, hash_file


def main():
    from huggingface_hub import snapshot_download
    spec = json.loads((Path(__file__).parent / "model-spec.json").read_text(encoding="utf-8"))
    target = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("/models/qwen3-vl-2b")
    snapshot_download(repo_id=spec["id"], revision=spec["revision"], local_dir=target,
                      allow_patterns=["*.json", "*.safetensors", "*.txt", "*.jinja", "LICENSE*"], token=False)
    weights = target / spec["weight_file"]
    if weights.stat().st_size != spec["weight_bytes"] or hash_file(weights) != spec["weight_sha256"]:
        raise RuntimeError("Pinned model weights failed size/SHA-256 verification")
    inventory = {str(p.relative_to(target)): hash_file(p) for p in target.rglob("*")
                 if p.is_file() and ".cache" not in p.parts and p.name != "visionforge-manifest.json"}
    atomic_json(target / "visionforge-manifest.json", {"model": spec, "files_sha256": inventory})
    print(json.dumps({"model": spec["id"], "revision": spec["revision"], "weight_sha256": spec["weight_sha256"]}))


if __name__ == "__main__":
    main()
