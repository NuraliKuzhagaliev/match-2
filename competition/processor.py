"""Shared processor construction and build-time, GPU-free compatibility check."""
import json
from pathlib import Path
import sys
from PIL import Image


def load_processor(model_dir, spec):
    from transformers import AutoProcessor
    # Qwen3-VL requires a video processor even for image-only inputs.
    import torchvision
    from torchvision.extension import _has_ops
    if torchvision.__version__ != "0.28.0+rocm10.0.0" or not _has_ops():
        raise RuntimeError("The matching ROCm torchvision build and native ops are required")
    return AutoProcessor.from_pretrained(
        model_dir, local_files_only=True, trust_remote_code=False, use_fast=False,
        min_pixels=spec["min_pixels"], max_pixels=spec["max_pixels"])


def main():
    import torch
    import torchvision
    import transformers
    spec = json.loads((Path(__file__).parent / "model-spec.json").read_text(encoding="utf-8"))
    processor = load_processor(sys.argv[1], spec)
    image = Image.new("RGB", (320, 160), "white")
    inputs = processor.apply_chat_template(
        [{"role": "user", "content": [{"type": "image", "image": image},
          {"type": "text", "text": "Read visible text."}]}],
        tokenize=True, add_generation_prompt=True, return_dict=True, return_tensors="pt")
    assert inputs["input_ids"].ndim == 2 and inputs["pixel_values"].numel() > 0
    assert inputs["image_grid_thw"].shape[-1] == 3
    print(json.dumps({"check": "processor_only", "status": "PASS",
                      "torch": torch.__version__, "torchvision": torchvision.__version__,
                      "transformers": transformers.__version__, "gpu_inference_tested": False}))


if __name__ == "__main__":
    main()
