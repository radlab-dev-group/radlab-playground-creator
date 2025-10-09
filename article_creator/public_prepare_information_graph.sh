#!/bin/bash

#export CUDA_VISIBLE_DEVICES=1

APP_FILE=build_information_graph.py
APP_FILE_2=similar_sub_graphs_extractor.py
MAIN_DIR=/mnt/local/dev/radlab-article-creator
APP_VISUALISER_GRAPHS="../../radlab-graph-visualiser/graph-visualiser/llms/outdir-plg-graphs/"

MAIN_OUT_DIR="resources/info_graphs"

DATE_STR=$(date +%Y%m%d_%H%M%S)
OUT_DIR="${MAIN_OUT_DIR}/${DATE_STR}"

MAIN_PROJECT_DIR=${MAIN_DIR}/article_creator
PERIODIC_APPS_DIR=${MAIN_PROJECT_DIR}/apps_creator/admin

APP_FILE_PATH_PERIODIC_APPS_DIR="${PERIODIC_APPS_DIR}/${APP_FILE}"
APP_FILE_PATH_MAIN_PROJECT_DIR="${MAIN_PROJECT_DIR}/${APP_FILE}"
cp "${APP_FILE_PATH_PERIODIC_APPS_DIR}" ${APP_FILE_PATH_MAIN_PROJECT_DIR}

APP_FILE_PATH_PERIODIC_APPS_DIR_2="${PERIODIC_APPS_DIR}/${APP_FILE_2}"
APP_FILE_PATH_MAIN_PROJECT_DIR_2="${MAIN_PROJECT_DIR}/${APP_FILE_2}"
cp "${APP_FILE_PATH_PERIODIC_APPS_DIR_2}" ${APP_FILE_PATH_MAIN_PROJECT_DIR_2}

cd "${MAIN_PROJECT_DIR}" || return

# =============================================================================
# Prepare base graph -- store to out dir
python3 "${APP_FILE_PATH_MAIN_PROJECT_DIR}" \
  --use-cache \
  --normalize-tangens \
  --normalize-minmax \
  --out-dir "${OUT_DIR}"

# =============================================================================
# Sub graphs extractor
# ----------------------------------------------------------------------
# run specific -- high similarity is required
# (for the last prepared information graph)
OUT_DIR_SUB_GRAPHS_S="subgraphs/specific"
python3 "${APP_FILE_PATH_MAIN_PROJECT_DIR_2}" \
  --out-dir "${OUT_DIR_SUB_GRAPHS_S}" \
  --min-sim-value 0.0 \
  --min-tg-value 1.0 \
  --type "specific" \
  --remove-previous-subgraphes

# ----------------------------------------------------------------------
# Remove old graphs from visualiser
rm -rf "${APP_VISUALISER_GRAPHS}"
mkdir -p "${APP_VISUALISER_GRAPHS}"

# ----------------------------------------------------------------------
# copy OUT_DIR_SUB_GRAPHS_S (specific) to visualiser
cp -R "${OUT_DIR}/NX/${OUT_DIR_SUB_GRAPHS_S}" "${APP_VISUALISER_GRAPHS}/"

# ----------------------------------------------------------------------
## run general -- low similarity is required
## (for the last prepared information graph)
#OUT_DIR_SUB_GRAPHS_G="subgraphs/general"
#python3 "${APP_FILE_PATH_MAIN_PROJECT_DIR_2}" \
#  --out-dir "${OUT_DIR_SUB_GRAPHS_G}" \
#  --min-sim-value 0.0 \
#  --min-tg-value 0.6 \
#  --type "general" \
#  --remove-previous-subgraphes
#
## copy general to visualiser
#cp -R "${OUT_DIR}/NX/${OUT_DIR_SUB_GRAPHS_G}" "${APP_VISUALISER_GRAPHS}/"

# =============================================================================

rm "${APP_FILE_PATH_MAIN_PROJECT_DIR}"
rm "${APP_FILE_PATH_MAIN_PROJECT_DIR_2}"
