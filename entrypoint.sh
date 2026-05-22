#!/bin/bash
set -e

echo "============================================"
echo "  Aimighty OpenVINO Reranker (CPU)"
echo "============================================"
echo ""

CACHE_DIR="${MODEL_CACHE_DIR:-/models_cache}"
MODEL_SUBDIR="${MODEL_NAME:-aimighty-reranker-0.6b}"
MODEL_PATH="${CACHE_DIR}/${MODEL_SUBDIR}"
export OV_DEVICE="${OV_DEVICE:-CPU}"

echo "[1/4] Checking model cache..."
echo "  Cache directory: ${CACHE_DIR}"
echo "  Model path:      ${MODEL_PATH}"
echo ""

if [ -f "${MODEL_PATH}/openvino_model.xml" ]; then
    echo "  [OK] OpenVINO model found in cache."
    echo "  Skipping download and conversion."
else
    echo "  [ERROR] Model not found in cache and not pre-converted in image."
    echo "  Please rebuild the Docker image with the model included."
    exit 1
fi

echo ""
echo "[3/4] Starting Aimighty Reranker Server..."
echo "  Model:  ${MODEL_PATH}"
echo "  Port:   ${PORT:-30001}"
echo "  Device: ${OV_DEVICE}"
echo "============================================"
echo ""

export MODEL_DIR="${MODEL_PATH}"

exec python3 /app/server.py
