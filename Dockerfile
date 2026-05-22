FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    git && \
    pip install --no-cache-dir \
        openvino>=2026.0.0 \
        "optimum-intel[openvino] @ git+https://github.com/huggingface/optimum-intel.git@main" \
        transformers==4.55.4 \
        fastapi uvicorn[standard] torch>=2.4.0 tokenizers>=0.21 sentencepiece && \
    apt-get remove -y git && \
    apt-get autoremove -y && \
    rm -rf /var/lib/apt/lists/*

# Pre-convert tomaarsen/Qwen3-Reranker-0.6B-seq-cls to OpenVINO INT8 during build
RUN optimum-cli export openvino \
    --model tomaarsen/Qwen3-Reranker-0.6B-seq-cls \
    --task text-classification \
    --weight-format int8 \
    /models_cache/aimighty-reranker-0.6b && \
    rm -rf /root/.cache/huggingface

COPY rerank-server.py /app/server.py
COPY entrypoint.sh /app/entrypoint.sh
RUN chmod +x /app/entrypoint.sh
RUN mkdir -p /models_cache
WORKDIR /app

ENV MALLOC_ARENA_MAX=1
ENV OV_CACHE_DIR=/tmp/ov_cache
ENV MODEL_CACHE_DIR=/models_cache

EXPOSE 30001

ENTRYPOINT ["/app/entrypoint.sh"]
