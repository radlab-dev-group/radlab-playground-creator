#!/bin/bash

export CUDA_VISIBLE_DEVICES=1

export TOKENIZERS_PARALLELISM=true
export CROSS_ENCODER_PORT=${CROSS_ENCODER_PORT:-8085}
export CROSS_ENCODER_HOST=${CROSS_ENCODER_HOST:-0.0.0.0}
export CROSS_ENCODER_MODEL=${CROSS_ENCODER_MODEL:-radlab/polish-cross-encoder}
export CROSS_ENCODER_DEVICE=${CROSS_ENCODER_DEVICE:-cuda:0}

MAIN_PROJECT_DIR=$(cd "$(dirname "$0")" && pwd)
cd "${MAIN_PROJECT_DIR}" || return

python3 -m services.cross_encoder_service
