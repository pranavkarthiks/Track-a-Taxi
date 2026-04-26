#!/bin/bash
set -euo pipefail

cd "${SLURM_SUBMIT_DIR:-$(pwd)}"

if [[ "$#" -ne 0 ]]; then
  echo "Usage: $0" >&2
  exit 1
fi

LOG_FILES=(
  "log-job-mask-eval-3m.out"
  "log-job-mask-eval-3m-dist.out"
  "log-job-3m-base-cfg2.out"
  "log-job-3m-dist-fix.out"
)

find_log_root() {
  local roots=()
  local first_log
  local candidate_root
  local log_name
  local has_all

  while IFS= read -r first_log; do
    candidate_root="$(cd "$(dirname "$first_log")" && pwd)"
    has_all=true
    for log_name in "${LOG_FILES[@]}"; do
      if [[ ! -f "${candidate_root}/${log_name}" ]]; then
        has_all=false
        break
      fi
    done
    if [[ "$has_all" == true ]]; then
      roots+=("$candidate_root")
    fi
  done < <(find . -type f -name "${LOG_FILES[0]}" | sort)

  if [[ "${#roots[@]}" -ne 1 ]]; then
    echo "Expected exactly one log directory containing the four required logs, found ${#roots[@]}." >&2
    printf '  %s\n' "${roots[@]}" >&2
    exit 1
  fi

  printf '%s\n' "${roots[0]}"
}

LOG_ROOT="$(find_log_root)"
CHECKPOINT_ROOT="$LOG_ROOT"

if [[ -f "$HOME/initMamba.sh" ]]; then
  . "$HOME/initMamba.sh"
fi

conda activate "${CONDA_ENV:-valency}"

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-4}"

nvidia-smi || true

RUN_EXP_IDS=()
RUN_LOG_FILES=()
RUN_CHECKPOINTS=()

find_checkpoint_for_log() {
  local log_file
  local exp_id
  local saved_model
  local checkpoint
  local matches=()
  local candidate
  local run_dir

  log_file="$1"

  exp_id="$(sed -n 's/.*Begin pipeline,.*exp_id=\([0-9][0-9]*\).*/\1/p' "$log_file" | tail -n 1)"
  if [[ -z "$exp_id" ]]; then
    echo "Could not read exp_id from log: $log_file" >&2
    exit 1
  fi

  saved_model="$(sed -n 's/.*Saved model at \(.*PDFormer_NYCTLC\.m\).*/\1/p' "$log_file" | tail -n 1)"
  if [[ -z "$saved_model" ]]; then
    echo "Could not read completed saved model from log: $log_file" >&2
    exit 1
  fi

  while IFS= read -r candidate; do
    run_dir="$(basename "$(dirname "$(dirname "$candidate")")")"
    if [[ "$run_dir" == "$exp_id" ]]; then
      matches+=("$candidate")
    fi
  done < <(find "$CHECKPOINT_ROOT" -type f -path '*/model_cache/PDFormer_NYCTLC.m' | sort)

  case "${#matches[@]}" in
    1) checkpoint="${matches[0]}" ;;
    0)
      echo "No real checkpoint under $CHECKPOINT_ROOT matches exp_id=${exp_id} from log: $log_file" >&2
      exit 1
      ;;
    *)
      echo "Multiple checkpoints under $CHECKPOINT_ROOT match exp_id=${exp_id} from log: $log_file" >&2
      printf '  %s\n' "${matches[@]}" >&2
      exit 1
      ;;
  esac

  RUN_EXP_IDS+=("$exp_id")
  RUN_LOG_FILES+=("$log_file")
  RUN_CHECKPOINTS+=("$checkpoint")
}

run_ablation() {
  local exp_id="$1"
  local log_file="$2"
  local checkpoint="$3"

  log_file="$(cd "$(dirname "$log_file")" && pwd)/$(basename "$log_file")"
  checkpoint="$(cd "$(dirname "$checkpoint")" && pwd)/$(basename "$checkpoint")"

  echo "Running PDFormer head ablation for exp_id=${exp_id}"
  echo "  output=${log_file}"
  echo "  checkpoint=${checkpoint}"

  python nyctlc_pdformer/pdformer_head_ablation.py \
    --dataset NYCTLC \
    --exp-id "$exp_id" \
    --split val \
    --log-file "$log_file" \
    --checkpoint "$checkpoint"
}

for log_name in "${LOG_FILES[@]}"; do
  log_file="${LOG_ROOT}/${log_name}"
  if [[ ! -f "$log_file" ]]; then
    echo "Missing required log file: $log_file" >&2
    exit 1
  fi
  find_checkpoint_for_log "$log_file"
done

for idx in "${!RUN_EXP_IDS[@]}"; do
  run_ablation "${RUN_EXP_IDS[$idx]}" "${RUN_LOG_FILES[$idx]}" "${RUN_CHECKPOINTS[$idx]}"
done
