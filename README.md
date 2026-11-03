# VisionForge AI — AMD AI League Match 2

Offline multilingual OCR for US and Chinese registration plates and road signs on AMD ROCm. The submission uses a resident Qwen3-VL-2B-Instruct worker and a thin Python CLI in the same container. Model weights are included in the published Docker image; runtime needs no external inference API or model download.

This is the [Match 2 source](competition/) for the shared lablab.ai entry. The separate [Match 3 RAG source](https://github.com/NuraliKuzhagaliev/amd-match3) is in its own repository. Image references are supplied in the organizer's private form, as instructed by the challenge brief, and are omitted here.

## Code and validation

- [Dockerfile](competition/Dockerfile), [runtime](competition/app.py), and [contract checks](tests/competition/)
- [Self-generated synthetic fixtures](data/competition/)
- [Independent AMD Docker check](competition/check.sh): `bash competition/check.sh`

Direct ROCm execution on an AMD Radeon Pro W7900D passed 10 main and 7 additional self-generated synthetic cases. Last measured startup was 16.01 s; ten CLI calls took 3.57 s; observed GPU memory peak was 4.790 GiB. The published image was anonymously pulled by digest using existing local layers. **GPU inference inside this exact Docker image and a cold pull on another computer remain unverified.** Synthetic cases do not establish hidden-set accuracy.

The pinned [Qwen3-VL-2B-Instruct](https://huggingface.co/Qwen/Qwen3-VL-2B-Instruct) model is Apache-2.0. The synthetic images in `data/competition` are CC0-1.0; no fonts are redistributed.
