#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
if [[ ! -e /dev/kfd || ! -d /dev/dri ]]; then
    echo 'AMD device paths missing: use a Linux AMD host with ROCm driver access.' >&2
    exit 2
fi
vf_docker_version="$(docker info --format '{{.ServerVersion}}')"
echo "$vf_docker_version"
# Older containerd stores reported only packed data in image inspect Size.
vf_docker_major="${vf_docker_version%%.*}"
vf_docker_remainder="${vf_docker_version#*.}"
vf_docker_minor="${vf_docker_remainder%%.*}"
if [[ ! "$vf_docker_major" =~ ^[0-9]+$ || ! "$vf_docker_minor" =~ ^[0-9]+$ ]] ||
   (( vf_docker_major < 29 || (vf_docker_major == 29 && vf_docker_minor < 8) )); then
    echo 'This acceptance runner requires Docker Engine 29.8+ for verified image-size accounting.' >&2
    exit 2
fi
mkdir -p evidence/competition
python3 -m venv .competition-venv
.competition-venv/bin/python -m pip install Pillow==12.1.1
.competition-venv/bin/python -m competition.preflight --report evidence/competition/amd-preflight.json
docker pull rocm/pytorch:rocm10.0_ubuntu26.04_py3.14_pytorch_release_2.13.0@sha256:3174cb7061d94c427da96c0edef4adea28046fa3f3b2ff3948dc4e995665ff8c
docker build --pull --progress=plain -f competition/Dockerfile -t visionforge-amd:local . 2>&1 | tee evidence/competition/amd-build.txt
.competition-venv/bin/python -m competition.validate --image visionforge-amd:local --cases data/competition/manifest.json --report evidence/competition/amd-acceptance.json 2>&1 | tee evidence/competition/amd-validation.txt
