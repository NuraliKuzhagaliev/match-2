"""Isolate dependencies while preserving the provider's real ROCm installation."""
import argparse
import importlib.metadata as metadata
import json
import os
from pathlib import Path
import subprocess
import sys
from competition.contract import atomic_json


def main():
    from competition.engine import require_rocm
    import torch
    require_rocm(torch)
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--venv", type=Path, default=root / ".amd-venv")
    parser.add_argument("--model-dir", type=Path, default=root / "models/qwen3-vl-2b")
    args = parser.parse_args()
    venv = args.venv.resolve()
    if venv.exists() and not (venv / "visionforge-owned.json").is_file():
        raise RuntimeError("Refusing to alter an existing environment without the VisionForge ownership marker")
    before = {"torch": torch.__version__, "torch_file": torch.__file__, "hip": torch.version.hip,
              "gpu": torch.cuda.get_device_name(0), "base_site": str(metadata.distribution("torch").locate_file(""))}
    subprocess.run([sys.executable, "-m", "venv", "--system-site-packages", str(venv)], check=True)
    atomic_json(venv / "visionforge-owned.json", before)
    site = next((venv / "lib").glob("python*/site-packages"))
    # AMD notebook Python injects /opt/venv, outside the standard system site.
    # Make this base visible to pip as well as imports; otherwise pip installs CUDA torch.
    (site / "amd-base-runtime.pth").write_text(before["base_site"] + "\n", encoding="utf-8")
    constraint = venv / "preserve-torch.txt"
    constraint.write_text("torch==" + before["torch"] + "\n", encoding="utf-8")
    python = str(venv / "bin/python")
    env = {**os.environ, "HF_HOME": str(root / "cache"), "HF_HUB_DISABLE_IMPLICIT_TOKEN": "1",
           "HF_HUB_DISABLE_TELEMETRY": "1", "HF_HUB_DISABLE_XET": "1", "TOKENIZERS_PARALLELISM": "false"}
    check = "import torch; assert torch.version.hip; assert torch.__version__ == " + repr(before["torch"]) + "; print(torch.__version__, torch.__file__)"
    subprocess.run([python, "-c", check], check=True, env=env)
    subprocess.run([python, "-m", "pip", "install", "--index-url", "https://pypi.org/simple", "--no-cache-dir",
                    "-r", str(root / "competition/requirements.txt"), "-c", str(root / "competition/constraints.txt"),
                    "-c", str(constraint)], check=True, env=env, timeout=600)
    subprocess.run([python, "-c", check], check=True, env=env)
    frozen = subprocess.check_output([python, "-m", "pip", "freeze", "--local"], text=True)
    (venv / "installed-requirements.txt").write_text(frozen, encoding="utf-8")
    subprocess.run([python, "-m", "competition.download_model", str(args.model_dir.resolve())], cwd=root, env=env, check=True, timeout=1800)
    subprocess.run([python, "-m", "competition.processor", str(args.model_dir.resolve())], cwd=root, env=env, check=True, timeout=120)
    print(json.dumps({"status": "SETUP_COMPLETE", "base_runtime_preserved": before, "venv": str(venv)}, indent=2), flush=True)
    return subprocess.call([python, "-m", "competition.notebook_validate", "--model-dir", str(args.model_dir.resolve())], cwd=root, env=env)


if __name__ == "__main__":
    raise SystemExit(main())
