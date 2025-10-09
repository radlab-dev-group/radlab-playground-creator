#!/bin/bash

export PLAYGROUND_PUBLIC_USER_NAME=XYZ
export PLAYGROUND_PUBLIC_USER_PASS=XYZ

APP_FILE=scrap_news_for_public_stream.py
MAIN_DIR=/mnt/data2/dev/develop/radlab-article-creator

MAIN_PROJECT_DIR=${MAIN_DIR}/article_creator
CONFIG_FILE_PATH=${MAIN_PROJECT_DIR}/configs/categories-and-main-urls.json
PERIODIC_APPS_DIR=${MAIN_PROJECT_DIR}/apps_creator/periodic

APP_FILE_PATH_PERIODIC_APPS_DIR="${PERIODIC_APPS_DIR}/${APP_FILE}"
APP_FILE_PATH_MAIN_PROJECT_DIR="${MAIN_PROJECT_DIR}/${APP_FILE}"

cp "${APP_FILE_PATH_PERIODIC_APPS_DIR}" ${APP_FILE_PATH_MAIN_PROJECT_DIR}

cd "${MAIN_PROJECT_DIR}" || return

if [ $# -eq 0 ]
then
  python3 "${APP_FILE_PATH_MAIN_PROJECT_DIR}" --json-config ${CONFIG_FILE_PATH}
else
  python3 "${APP_FILE_PATH_MAIN_PROJECT_DIR}" --json-config ${CONFIG_FILE_PATH} --debug
fi

rm "${APP_FILE_PATH_MAIN_PROJECT_DIR}"
