#!/usr/bin/env sh
set -eu

DEFAULT_MODEL_DIR="/models/layout"

if [ -z "${GLMOCR_LAYOUT_MODEL_DIR:-}" ]; then
  export GLMOCR_LAYOUT_MODEL_DIR="$DEFAULT_MODEL_DIR"
fi

mkdir -p "${GLMOCR_LAYOUT_MODEL_DIR}"

if [ -n "${GLMOCR_LAYOUT_MODEL_SOURCE:-}" ]; then
  if [ ! -e "${GLMOCR_LAYOUT_MODEL_DIR}/config.json" ]; then
    echo "Preparing layout model from ${GLMOCR_LAYOUT_MODEL_SOURCE} -> ${GLMOCR_LAYOUT_MODEL_DIR}"
    case "${GLMOCR_LAYOUT_MODEL_SOURCE}" in
      /*)
        cp -R "${GLMOCR_LAYOUT_MODEL_SOURCE}/." "${GLMOCR_LAYOUT_MODEL_DIR}/"
        ;;
      *)
        echo "Unsupported GLMOCR_LAYOUT_MODEL_SOURCE: ${GLMOCR_LAYOUT_MODEL_SOURCE}" >&2
        echo "Use a local absolute directory path, for example a PAI-mounted OSS path." >&2
        exit 1
        ;;
    esac
  fi
fi

exec python -m glmocr.server "$@"
