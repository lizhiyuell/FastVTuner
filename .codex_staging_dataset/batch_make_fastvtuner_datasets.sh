#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
PY_SCRIPT="${SCRIPT_DIR}/make_fastvtuner_dataset.py"
SEED="${SEED:-42}"
BATCH_SIZE="${BATCH_SIZE:-256}"

resolve_repo_root() {
  local candidate
  local candidates=()

  if [[ -n "${FASTVTUNER_ROOT:-}" ]]; then
    candidates+=("${FASTVTUNER_ROOT}")
  fi

  candidates+=(
    "${PWD}"
    "$(dirname "${PWD}")"
    "${SCRIPT_DIR}"
    "$(dirname "${SCRIPT_DIR}")"
  )

  for candidate in "${candidates[@]}"; do
    if [[ -d "${candidate}/vector-db-benchmark/datasets" ]]; then
      printf '%s\n' "${candidate}"
      return 0
    fi
  done

  echo "Failed to locate the FastVTuner repo root. Set FASTVTUNER_ROOT to the repo path." >&2
  return 1
}

REPO_ROOT="$(resolve_repo_root)"
SOURCE_ROOT="${REPO_ROOT}/vector-db-benchmark/datasets"
OUTPUT_ROOT="${REPO_ROOT}/dataset"

# 这里只维护需要处理的 hdf5 文件列表。
HDF5_FILES=(
  "${SOURCE_ROOT}/glove-100-angular/glove-100-angular.hdf5"
  "${SOURCE_ROOT}/glove-100-angular-p-10/glove-100-angular-p-10.hdf5"
  "${SOURCE_ROOT}/glove-100-angular-p-20/glove-100-angular-p-20.hdf5"
  "${SOURCE_ROOT}/glove-100-angular-p-30/glove-100-angular-p-30.hdf5"
  "${SOURCE_ROOT}/glove-100-angular-p-40/glove-100-angular-p-40.hdf5"
  "${SOURCE_ROOT}/glove-100-angular-p-50/glove-100-angular-p-50.hdf5"
  "${SOURCE_ROOT}/glove-100-angular-p-60/glove-100-angular-p-60.hdf5"
  "${SOURCE_ROOT}/glove-100-angular-p-70/glove-100-angular-p-70.hdf5"
  "${SOURCE_ROOT}/glove-100-angular-p-80/glove-100-angular-p-80.hdf5"
  "${SOURCE_ROOT}/glove-100-angular-p-90/glove-100-angular-p-90.hdf5"
  "${SOURCE_ROOT}/glove-100-angular-p-100/glove-100-angular-p-100.hdf5"
  "${SOURCE_ROOT}/gist-960-euclidean/gist-960-euclidean.hdf5"
  "${SOURCE_ROOT}/gist-960-euclidean-p-10/gist-960-euclidean-p-10.hdf5"
  "${SOURCE_ROOT}/gist-960-euclidean-p-20/gist-960-euclidean-p-20.hdf5"
  "${SOURCE_ROOT}/gist-960-euclidean-p-30/gist-960-euclidean-p-30.hdf5"
  "${SOURCE_ROOT}/gist-960-euclidean-p-40/gist-960-euclidean-p-40.hdf5"
  "${SOURCE_ROOT}/gist-960-euclidean-p-50/gist-960-euclidean-p-50.hdf5"
  "${SOURCE_ROOT}/gist-960-euclidean-p-60/gist-960-euclidean-p-60.hdf5"
  "${SOURCE_ROOT}/gist-960-euclidean-p-70/gist-960-euclidean-p-70.hdf5"
  "${SOURCE_ROOT}/gist-960-euclidean-p-80/gist-960-euclidean-p-80.hdf5"
  "${SOURCE_ROOT}/gist-960-euclidean-p-90/gist-960-euclidean-p-90.hdf5"
  "${SOURCE_ROOT}/gist-960-euclidean-p-100/gist-960-euclidean-p-100.hdf5"
)

if [[ ! -f "${PY_SCRIPT}" ]]; then
  echo "Python script not found: ${PY_SCRIPT}" >&2
  exit 1
fi

if [[ ${#HDF5_FILES[@]} -eq 0 ]]; then
  echo "HDF5_FILES is empty. Edit this script and add at least one file." >&2
  exit 1
fi

derive_params() {
  local input_hdf5="$1"
  local basename stem family dimension metric ratio ratio_value output_file

  basename=$(basename "${input_hdf5}")
  stem="${basename%.hdf5}"
  IFS='-' read -r family dimension metric ratio ratio_value <<< "${stem}"

  if [[ -z "${family}" || -z "${dimension}" || -z "${metric}" ]]; then
    echo "Failed to parse dataset name: ${stem}" >&2
    return 1
  fi

  if [[ -n "${ratio:-}" ]]; then
    output_file="${OUTPUT_ROOT}/${family}-${ratio}-${ratio_value}.npz"
  else
    output_file="${OUTPUT_ROOT}/${family}.npz"
  fi

  printf '%s|%s|%s|%s\n' "${output_file}" "${dimension}" "${metric}" "${stem}"
}

for input_hdf5 in "${HDF5_FILES[@]}"; do
  if [[ ! -f "${input_hdf5}" ]]; then
    echo "Input HDF5 not found: ${input_hdf5}" >&2
    exit 1
  fi

  IFS='|' read -r output_file dimension distance_metric dataset_stem < <(derive_params "${input_hdf5}")

  echo "[run] dataset=${dataset_stem} output=${output_file} dim=${dimension} metric=${distance_metric}"
  python "${PY_SCRIPT}" \
    --input-hdf5 "${input_hdf5}" \
    --output-file "${output_file}" \
    --dimension "${dimension}" \
    --distance-metric "${distance_metric}" \
    --seed "${SEED}" \
    --batch-size "${BATCH_SIZE}"
done
