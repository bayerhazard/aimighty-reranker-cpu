FROM ubuntu:24.04

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
        ca-certificates wget gnupg python3 python3-pip \
    && rm -rf /var/lib/apt/lists/*

# Intel GPU compute runtime for Arrow Lake iGPU (OpenVINO GPU plugin / Level Zero)
RUN wget -qO - https://repositories.intel.com/gpu/intel-graphics.key | gpg --dearmor --output /usr/share/keyrings/intel-graphics.gpg \
    && echo "deb [arch=amd64 signed-by=/usr/share/keyrings/intel-graphics.gpg] https://repositories.intel.com/gpu/ubuntu noble client" > /etc/apt/sources.list.d/intel.gpu.list \
    && apt-get update \
    && apt-get install -y --no-install-recommends \
        intel-opencl-icd \
        intel-level-zero-gpu \
        level-zero \
        intel-igc-cm \
        intel-ocloc \
        ocl-icd-libopencl1 \
    && rm -rf /var/lib/apt/lists/*

RUN pip3 install --no-cache-dir --break-system-packages \
        openvino==2026.4.0 \
        optimum-intel[openvino]==2.2.0 \
        transformers==4.57.6 \
        fastapi "uvicorn[standard]" "torch>=2.4.0" "tokenizers>=0.21" sentencepiece

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
