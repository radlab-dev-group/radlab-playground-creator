#!/bin/bash

export CUDA_VISIBLE_DEVICES=1

APP_FILE=sub_graphs_generalizer.py
MAIN_DIR=/mnt/data2/dev/develop/radlab-article-creator

MAIN_OUT_DIR="resources/info_graphs/generalized"

DATE_STR=$(date +%Y%m%d_%H%M%S)
OUT_DIR="${MAIN_OUT_DIR}/${DATE_STR}"

OUT_DIR="resources/info_graphs/generalized/20250629_185005/"

# =============================================================================

MAIN_PROJECT_DIR=${MAIN_DIR}/article_creator
PERIODIC_APPS_DIR=${MAIN_PROJECT_DIR}/apps_creator/admin

APP_FILE_PATH_PERIODIC_APPS_DIR="${PERIODIC_APPS_DIR}/${APP_FILE}"
APP_FILE_PATH_MAIN_PROJECT_DIR="${MAIN_PROJECT_DIR}/${APP_FILE}"
cp "${APP_FILE_PATH_PERIODIC_APPS_DIR}" ${APP_FILE_PATH_MAIN_PROJECT_DIR}

# =============================================================================

cd "${MAIN_PROJECT_DIR}" || return

python3 "${APP_FILE_PATH_MAIN_PROJECT_DIR}" \
  --out-dir "${OUT_DIR}" \
  --device=gpu \
  --sub-graphs-dir "resources/info_graphs/20250629_023701/NX/subgraphs/specific/min-sim-0.0__min-tg-1.0/" \
  --skip-outliers --load-from-out-dir

# =============================================================================

rm "${APP_FILE_PATH_MAIN_PROJECT_DIR}"
