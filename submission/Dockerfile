# LifeMate Search-First Docker image (ZAC 2026)
# Build from the repository root:
#   docker build -f submission/Dockerfile -t zac2026-lifemate:v1 .
FROM nvidia/cuda:13.2.0-devel-ubuntu22.04

LABEL maintainer="LifeMate Team"
LABEL description="Search-First LifeMate offline inference (ZAC 2026)"

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV TOKENIZERS_PARALLELISM=false
ENV CUBLAS_WORKSPACE_CONFIG=:4096:8
ENV PYTHONIOENCODING=utf-8

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 python3-pip libgl1-mesa-glx libglib2.0-0 libsndfile1 ffmpeg \
    vim git curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /code

RUN pip3 install --no-cache-dir --upgrade pip && \
    pip3 install --no-cache-dir torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

COPY submission/requirements.txt /code/requirements.txt
RUN pip3 install --no-cache-dir -r /code/requirements.txt

COPY src/ /code/src/
COPY pipeline/ /code/pipeline/
COPY zaic_tools/ /code/zaic_tools/
COPY submission/predict.py /code/predict.py
COPY submission/preprocessing.py /code/preprocessing.py
COPY submission/predict.sh /code/predict.sh
COPY submission/start_jupyter.sh /code/start_jupyter.sh
COPY submission/README.md /code/README.md
COPY models/ /code/saved_models/

RUN chmod +x /code/predict.sh /code/preprocessing.py /code/predict.py /code/start_jupyter.sh

# Base model (optional if models/Qwen3.5-4B is provided).
RUN if [ ! -f /code/saved_models/Qwen3.5-4B/config.json ]; then \
        python3 -c "from huggingface_hub import snapshot_download; snapshot_download('Qwen/Qwen3.5-4B', local_dir='/code/saved_models/Qwen3.5-4B')" ; \
    fi

# Cache the E5 retriever used by sys.search_tools for fully offline inference.
# HF_HOME points at the cache root, so snapshot_download uses <HF_HOME>/hub and
# SentenceTransformer at runtime finds the same files.
ENV HF_HOME=/code/saved_models/hf_cache
RUN python3 -c "from huggingface_hub import snapshot_download; snapshot_download('intfloat/multilingual-e5-small', revision='614241f622f53c4eeff9890bdc4f31cfecc418b3')"

# The image must run without network access.
ENV HF_HUB_OFFLINE=1
ENV TRANSFORMERS_OFFLINE=1

ENV DATA_DIR=/data
ENV OUTPUT_FILE=/result/submission.jsonl

CMD ["/bin/bash", "/code/predict.sh"]
