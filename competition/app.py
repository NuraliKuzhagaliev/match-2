"""Harness entry: python3 /app/app.py --input-image PATH."""
import argparse
from pathlib import Path
import sys
import uuid
from competition.contract import atomic_json, rpc, transcription


def main(argv=None):
    parser = argparse.ArgumentParser(description="AMD AI League ROCm OCR client")
    parser.add_argument("--input-image", required=True, type=Path)
    parser.add_argument("--output-dir", type=Path, default=Path("/app/output"))
    args = parser.parse_args(argv)
    output = args.output_dir / (args.input_image.stem + "_output.json")
    # Failed repeats must not expose a previous answer to the grading harness.
    try:
        output.unlink(missing_ok=True)
        image = args.input_image.resolve(strict=True)
        if not image.is_file():
            raise ValueError("Input must be a regular image file")
        response = rpc({"operation": "ocr", "request_id": uuid.uuid4().hex, "input_image": str(image)})
        atomic_json(output, {"text": transcription(response["text"])})
        return 0
    except (OSError, ValueError, RuntimeError, ConnectionError, KeyError) as exc:
        print(f"Competition OCR failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
