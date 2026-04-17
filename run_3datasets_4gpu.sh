#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

CONDA_ENV="${CONDA_ENV:-gyf}"
SEED="${SEED:-42}"
GPU_IDS=(0 1 2 3)
FORCE_PREPROCESS="${FORCE_PREPROCESS:-0}"

# 当前按 3 个数据集运行：
# sdwpf / penmanshiel / norrekaer_enge
DATASETS=("sdwpf" "penmanshiel" "norrekaer_enge")
GRAPH_VARIANTS=("baseline" "local_upstream")
ALL_METHODS=("ScaleShift" "VariationalScaleShift" "EAC" "PatchTST")
WARMUP_METHODS=("ScaleShift" "VariationalScaleShift" "EAC")


activate_conda() {
  if ! command -v conda >/dev/null 2>&1; then
    echo "conda not found. Please activate the gyf environment manually before running this script."
    return 1
  fi
  # shellcheck disable=SC1091
  source "$(conda info --base)/etc/profile.d/conda.sh"
  conda activate "$CONDA_ENV"
}


get_cache_meta() {
  local dataset="$1"
  local graph_variant="$2"

  python - <<PY
from config.config import Config
from config.dataset_registry import DATASET_REGISTRY

dataset = "${dataset}"
graph_variant = "${graph_variant}"
seed = int("${SEED}")
cfg = Config(method="ScaleShift", seed=seed, dataset=dataset, graph_variant=graph_variant)
num_stages = len(DATASET_REGISTRY[dataset]["default_expansion_groups"]) + 1
print(cfg.save_data_path)
print(cfg.graph_path)
print(num_stages)
PY
}


cache_ready() {
  local dataset="$1"
  local graph_variant="$2"
  local unified_path graph_path stage_count stage_idx
  local -a meta_lines

  mapfile -t meta_lines < <(get_cache_meta "$dataset" "$graph_variant")
  if (( ${#meta_lines[@]} != 3 )); then
    echo "[WARN] failed to resolve cache metadata for dataset=${dataset} graph=${graph_variant}"
    return 1
  fi

  unified_path="${meta_lines[0]}/unified_data.npz"
  graph_path="${meta_lines[1]}"
  stage_count="${meta_lines[2]}"

  if [[ ! -f "$unified_path" ]]; then
    return 1
  fi

  for ((stage_idx = 0; stage_idx < stage_count; stage_idx++)); do
    if [[ ! -f "${graph_path}/stage_${stage_idx}_adj.npz" ]]; then
      return 1
    fi
  done

  return 0
}


graph_supported() {
  local dataset="$1"
  local graph_variant="$2"
  if [[ "$graph_variant" == "baseline" ]]; then
    return 0
  fi
  if [[ "$graph_variant" == "local_upstream" && ( "$dataset" == "sdwpf" || "$dataset" == "penmanshiel" || "$dataset" == "norrekaer_enge" ) ]]; then
    return 0
  fi
  return 1
}


cleanup() {
  local pid
  for pid in $(jobs -pr); do
    kill "$pid" >/dev/null 2>&1 || true
  done
}


run_preprocess() {
  local dataset="$1"
  local graph_variant="$2"
  local logname="preprocess_${dataset}_${graph_variant}"
  local cmd=(
    python main.py
    --dataset "$dataset"
    --graph_variant "$graph_variant"
    --method ScaleShift
    --data_process 1
    --train 0
    --seed "$SEED"
    --gpuid -1
    --logname "$logname"
  )

  if [[ "$FORCE_PREPROCESS" != "1" ]] && cache_ready "$dataset" "$graph_variant"; then
    echo "============================================================"
    echo "[PREPROCESS] SKIP dataset=${dataset} graph=${graph_variant} (cache already exists)"
    echo "============================================================"
    return 0
  fi

  echo "============================================================"
  if [[ "$FORCE_PREPROCESS" == "1" ]]; then
    echo "[PREPROCESS] REBUILD dataset=${dataset} graph=${graph_variant} (FORCE_PREPROCESS=1)"
  else
    echo "[PREPROCESS] BUILD dataset=${dataset} graph=${graph_variant} (cache missing or incomplete)"
  fi
  echo "============================================================"
  "${cmd[@]}"
}


launch_job() {
  local gpu="$1"
  local job_id="$2"
  local total_jobs="$3"
  local cmd="$4"

  echo "[$(date '+%F %T')] [GPU ${gpu}] START ${job_id}/${total_jobs}: ${cmd}"
  bash -lc "${cmd} --gpuid ${gpu}" &
  LAUNCHED_PID="$!"
}


run_phase() {
  local phase_name="$1"
  shift
  local jobs=("$@")
  local total_jobs="${#jobs[@]}"
  local num_gpus="${#GPU_IDS[@]}"
  local next_job_idx=0
  local running_jobs=0
  local sleep_seconds=5
  local gpu pid status
  local -a gpu_pids

  if (( total_jobs == 0 )); then
    echo "[${phase_name}] no jobs to run."
    return 0
  fi

  echo "============================================================"
  echo "[${phase_name}] total jobs: ${total_jobs}"
  echo "============================================================"

  while (( next_job_idx < total_jobs || running_jobs > 0 )); do
    for gpu in "${GPU_IDS[@]}"; do
      pid="${gpu_pids[$gpu]:-}"
      if [[ -n "${pid}" ]]; then
        if kill -0 "${pid}" >/dev/null 2>&1; then
          continue
        fi
        wait "${pid}"
        status=$?
        if (( status != 0 )); then
          echo "[${phase_name}] job on GPU ${gpu} failed with exit code ${status}."
          return "${status}"
        fi
        echo "[$(date '+%F %T')] [GPU ${gpu}] DONE"
        unset 'gpu_pids[$gpu]'
        ((running_jobs--))
      fi

      if (( next_job_idx >= total_jobs )); then
        continue
      fi

      launch_job "${gpu}" "$((next_job_idx + 1))" "${total_jobs}" "${jobs[$next_job_idx]}"
      pid="${LAUNCHED_PID}"
      gpu_pids[$gpu]="${pid}"
      next_job_idx=$((next_job_idx + 1))
      running_jobs=$((running_jobs + 1))
    done

    if (( running_jobs > 0 )); then
      sleep "${sleep_seconds}"
    fi
  done
}


build_phase1_jobs() {
  local dataset graph_variant method logname cmd
  PHASE1_JOBS=()
  for dataset in "${DATASETS[@]}"; do
    for graph_variant in "${GRAPH_VARIANTS[@]}"; do
      if ! graph_supported "$dataset" "$graph_variant"; then
        continue
      fi
      for method in "${ALL_METHODS[@]}"; do
        logname="${dataset}_${graph_variant}_${method}_warmup_on"
        cmd="python main.py --dataset ${dataset} --graph_variant ${graph_variant} --method ${method} --train 1 --data_process 0 --seed ${SEED} --logname ${logname}"
        PHASE1_JOBS+=("${cmd}")
      done
    done
  done
}


build_phase2_jobs() {
  local dataset graph_variant method logname cmd
  PHASE2_JOBS=()
  for dataset in "${DATASETS[@]}"; do
    for graph_variant in "${GRAPH_VARIANTS[@]}"; do
      if ! graph_supported "$dataset" "$graph_variant"; then
        continue
      fi
      for method in "${WARMUP_METHODS[@]}"; do
        logname="${dataset}_${graph_variant}_${method}_warmup_off"
        cmd="python main.py --dataset ${dataset} --graph_variant ${graph_variant} --method ${method} --train 0 --data_process 0 --seed ${SEED} --no_warmup 1 --logname ${logname}"
        PHASE2_JOBS+=("${cmd}")
      done
    done
  done
}


main() {
  trap cleanup INT TERM
  activate_conda

  echo "Root dir      : $ROOT_DIR"
  echo "Conda env     : $CONDA_ENV"
  echo "Seed          : $SEED"
  echo "Datasets      : ${DATASETS[*]}"
  echo "GPU ids       : ${GPU_IDS[*]}"
  echo "Force preprocess: $FORCE_PREPROCESS"

  local dataset graph_variant
  for dataset in "${DATASETS[@]}"; do
    for graph_variant in "${GRAPH_VARIANTS[@]}"; do
      if ! graph_supported "$dataset" "$graph_variant"; then
        echo "[SKIP] dataset=${dataset} graph=${graph_variant} is not supported by current code."
        continue
      fi
      run_preprocess "$dataset" "$graph_variant"
    done
  done

  build_phase1_jobs
  build_phase2_jobs

  echo "Phase 1 jobs  : ${#PHASE1_JOBS[@]}"
  echo "Phase 2 jobs  : ${#PHASE2_JOBS[@]}"
  echo "Total jobs    : $((${#PHASE1_JOBS[@]} + ${#PHASE2_JOBS[@]}))"

  run_phase "PHASE1_TRAIN_AND_TEST" "${PHASE1_JOBS[@]}"
  run_phase "PHASE2_NO_WARMUP_ABLATION" "${PHASE2_JOBS[@]}"

  echo "============================================================"
  echo "All scheduled experiments completed."
  echo "============================================================"
}


if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
  main "$@"
fi
