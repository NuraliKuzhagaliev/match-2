# Final stage must retain these exact base layers. Never squash/flatten this image.
FROM rocm/pytorch:rocm10.0_ubuntu26.04_py3.14_pytorch_release_2.13.0@sha256:3174cb7061d94c427da96c0edef4adea28046fa3f3b2ff3948dc4e995665ff8c
WORKDIR /app
ENV PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 \
    HF_HOME=/models/cache HF_HUB_DISABLE_IMPLICIT_TOKEN=1 HF_HUB_DISABLE_TELEMETRY=1 \
    TOKENIZERS_PARALLELISM=false DO_NOT_TRACK=1
COPY competition/requirements.txt /app/requirements.txt
COPY competition/constraints.txt /app/constraints.txt
RUN python3 -c "import torch; assert torch.version.hip and torch.__version__.startswith('2.13.0'), 'Wrong base torch build'" \
    && python3 -m pip install --no-cache-dir --no-deps --index-url https://stable.repo.amd.com/rocm/whl-next/ 'torchvision==0.28.0+rocm10.0.0' \
    && python3 -m pip install --no-cache-dir -r /app/requirements.txt -c /app/constraints.txt \
    && python3 -m pip check \
    && python3 -c "import torch; assert torch.version.hip and torch.__version__.startswith('2.13.0'), 'ROCm torch was replaced'" \
    && python3 -m pip freeze > /app/installed-requirements.txt
# Keep the public checkpoint layer reusable when only inference code changes.
COPY competition/__init__.py competition/contract.py competition/download_model.py competition/processor.py competition/model-spec.json /app/competition/
RUN python3 -m competition.download_model /models/qwen3-vl-2b \
    && python3 -m competition.processor /models/qwen3-vl-2b > /app/processor-check.json \
    && mkdir -p /app/input /app/output /app/evidence /run/visionforge
COPY competition/*.py /app/competition/
COPY competition/app.py /app/app.py
ENV HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
HEALTHCHECK --interval=3s --timeout=5s --start-period=600s --retries=3 CMD python3 -m competition.health
STOPSIGNAL SIGTERM
ENTRYPOINT []
CMD ["python3", "-m", "competition.worker"]
