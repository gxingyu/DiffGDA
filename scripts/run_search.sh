#!/usr/bin/env bash
set -euo pipefail

# Multi-GPU hyperparameter search for DiffGDA.
#
# The script shards the default hyperparameter grid across the selected GPUs. It can run
# explicit transfer tasks, all directed transfers within one benchmark domain,
# or all 14 transfer tasks used in the paper.
#
# Examples:
#   bash scripts/run_search.sh
#   SCENARIOS="BRAZIL:EUROPE USA:BRAZIL" GPUS="0 1 2 4" bash scripts/run_search.sh
#   DOMAIN=airport GPUS="0 1 2 4 5 6 7" bash scripts/run_search.sh
#   ALL_SCENARIOS=1 GPUS="0 1 2 4 5 6 7" bash scripts/run_search.sh
#
# Common overrides:
#   OUT_DIR=logs_search/airport_grid bash scripts/run_search.sh
#   RESUME_KEY=airport_mmd_v1 bash scripts/run_search.sh
#   T_VALUES="20 40 60 80 100 120 150" bash scripts/run_search.sh
#   ALIGNMENT=adv bash scripts/run_search.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

if command -v conda >/dev/null 2>&1; then
  # shellcheck disable=SC1091
  source "$(conda info --base)/etc/profile.d/conda.sh"
  conda activate DiffGDA
fi

CITATION_SCENARIOS=(
  "ACMv9:Citationv1"
  "ACMv9:DBLPv7"
  "Citationv1:ACMv9"
  "Citationv1:DBLPv7"
  "DBLPv7:ACMv9"
  "DBLPv7:Citationv1"
)

AIRPORT_SCENARIOS=(
  "USA:BRAZIL"
  "USA:EUROPE"
  "BRAZIL:USA"
  "BRAZIL:EUROPE"
  "EUROPE:USA"
  "EUROPE:BRAZIL"
)

BLOG_SCENARIOS=(
  "Blog1:Blog2"
  "Blog2:Blog1"
)

DOMAIN="${DOMAIN:-}"
SCENARIOS_STR="${SCENARIOS:-}"
ALL_SCENARIOS="${ALL_SCENARIOS:-0}"
GPUS_STR="${GPUS:-0}"
OUT_DIR="${OUT_DIR:-logs_search/diffgda_search_$(date +%Y%m%d%H%M)}"
RESUME_KEY="${RESUME_KEY:-default}"

LR_VALUES_STR="${LR_VALUES:-0.0001 0.001 0.01}"
ALPHA_VALUES_STR="${ALPHA_VALUES:-0.1 0.2 0.3}"
ETA_VALUES_STR="${ETA_VALUES:-0.1 0.2 0.3 0.4 0.5}"
T_VALUES_STR="${T_VALUES:-20 40 60 80 100}"

ALIGNMENT="${ALIGNMENT:-mmd}"
DROPOUT="${DROPOUT:-0.2}"
EPOCH="${EPOCH:-150}"
NHID="${NHID:-64}"
NUM_LAYERS="${NUM_LAYERS:-3}"
WEIGHT_DECAY="${WEIGHT_DECAY:-0.0005}"
S_PNUMS="${S_PNUMS:-0}"
T_PNUMS="${T_PNUMS:-10}"
ROUND="${ROUND:-0}"
SEED="${SEED:-521070}"
DIFFUSION_EPOCHS="${DIFFUSION_EPOCHS:-}"

read -r -a GPUS <<< "${GPUS_STR}"
read -r -a LRS <<< "${LR_VALUES_STR}"
read -r -a ALPHAS <<< "${ALPHA_VALUES_STR}"
read -r -a ETAS <<< "${ETA_VALUES_STR}"
read -r -a STEPS <<< "${T_VALUES_STR}"

if [[ "${#GPUS[@]}" -eq 0 ]]; then
  echo "[error] no GPUs configured. Set GPUS=\"0 1 2\"." >&2
  exit 1
fi

select_scenarios() {
  if [[ -n "${SCENARIOS_STR}" ]]; then
    read -r -a SELECTED_SCENARIOS <<< "${SCENARIOS_STR}"
    return
  fi

  if [[ "${ALL_SCENARIOS}" == "1" ]]; then
    SELECTED_SCENARIOS=("${CITATION_SCENARIOS[@]}" "${AIRPORT_SCENARIOS[@]}" "${BLOG_SCENARIOS[@]}")
    return
  fi

  case "${DOMAIN}" in
    citation)
      SELECTED_SCENARIOS=("${CITATION_SCENARIOS[@]}")
      ;;
    airport)
      SELECTED_SCENARIOS=("${AIRPORT_SCENARIOS[@]}")
      ;;
    blog)
      SELECTED_SCENARIOS=("${BLOG_SCENARIOS[@]}")
      ;;
    "")
      SELECTED_SCENARIOS=("ACMv9:DBLPv7")
      ;;
    *)
      echo "[error] unknown DOMAIN=${DOMAIN}. Use citation, airport, or blog." >&2
      exit 1
      ;;
  esac
}

config_for_source() {
  local source="$1"
  echo "${source}"
}

write_meta() {
  cat > "${OUT_DIR}/run_meta.txt" <<EOF
created_at=$(date '+%Y-%m-%dT%H:%M:%S')
scenarios=${SELECTED_SCENARIOS[*]}
domain=${DOMAIN:-custom}
all_scenarios=${ALL_SCENARIOS}
gpus=${GPUS[*]}
resume_key=${RESUME_KEY}
lr=${LRS[*]}
alpha=${ALPHAS[*]}
eta=${ETAS[*]}
T=${STEPS[*]}
alignment=${ALIGNMENT}
dropout=${DROPOUT}
epoch=${EPOCH}
nhid=${NHID}
num_layers=${NUM_LAYERS}
weight_decay=${WEIGHT_DECAY}
s_pnums=${S_PNUMS}
t_pnums=${T_PNUMS}
round=${ROUND}
seed=${SEED}
diffusion_epochs=${DIFFUSION_EPOCHS:-config_default}
EOF
}

run_trial() {
  local gpu="$1"
  local trial_idx="$2"
  local scenario="$3"
  local lr="$4"
  local alpha="$5"
  local eta="$6"
  local steps="$7"
  local source="${scenario%%:*}"
  local target="${scenario##*:}"
  local config
  config="$(config_for_source "${source}")"
  local seed=$((SEED + trial_idx))
  local scenario_dir="${scenario/:/_to_}"
  local tag
  tag="trial$(printf '%05d' "${trial_idx}")_r${ROUND}_${scenario_dir}_lr${lr}_a${alpha}_eta${eta}_T${steps}_${ALIGNMENT}_${RESUME_KEY}"
  tag="${tag//./p}"
  local log_dir="${OUT_DIR}/trial_logs/${scenario_dir}"
  local log_path="${log_dir}/${tag}.log"
  mkdir -p "${log_dir}"

  if [[ -f "${log_path}" ]] && grep -q '^RETURN_CODE: 0$' "${log_path}"; then
    echo "[skip] cuda:${gpu} trial=${trial_idx} scenario=${scenario}" >&2
    return 0
  fi

  local cmd=(
    python search_diffgda.py
    --trial
    --source "${source}"
    --target "${target}"
    --config "${config}"
    --device "cuda:${gpu}"
    --nhid "${NHID}"
    --dropout "${DROPOUT}"
    --alignment "${ALIGNMENT}"
    --resume-key "${RESUME_KEY}"
    --epoch "${EPOCH}"
    --num-layers "${NUM_LAYERS}"
    --weight-decay "${WEIGHT_DECAY}"
    --s-pnums "${S_PNUMS}"
    --t-pnums "${T_PNUMS}"
    --lr "${lr}"
    --alpha "${alpha}"
    --eta "${eta}"
    --diffusion-steps "${steps}"
    --seed "${seed}"
    --round "${ROUND}"
  )

  if [[ -n "${DIFFUSION_EPOCHS}" ]]; then
    cmd+=(--diffusion-epochs "${DIFFUSION_EPOCHS}")
  fi

  {
    echo "COMMAND: ${cmd[*]}"
    echo "START: $(date '+%Y-%m-%dT%H:%M:%S')"
    echo "GPU: cuda:${gpu}"
    echo "SCENARIO: ${scenario}"
    echo "TRIAL_INDEX: ${trial_idx}"
    echo
  } > "${log_path}"

  set +e
  "${cmd[@]}" >> "${log_path}" 2>&1
  local rc="$?"
  set -e

  {
    echo
    echo "END: $(date '+%Y-%m-%dT%H:%M:%S')"
    echo "RETURN_CODE: ${rc}"
  } >> "${log_path}"

  local result_json
  result_json="$(grep '^RESULT_JSON ' "${log_path}" | tail -n 1 | sed 's/^RESULT_JSON //')"
  python - "$rc" "$trial_idx" "$gpu" "$scenario" "$lr" "$alpha" "$eta" "$steps" "$log_path" "$result_json" <<'PY'
import csv
import json
import sys

rc, trial, gpu, scenario, lr, alpha, eta, steps, log_path, result_json = sys.argv[1:]
row = {
    "return_code": int(rc),
    "trial": int(trial),
    "gpu": f"cuda:{gpu}",
    "scenario": scenario,
    "lr": float(lr),
    "alpha": float(alpha),
    "eta": float(eta),
    "diffusion_steps": int(steps),
    "micro_f1": "",
    "macro_f1": "",
    "best_micro_f1": "",
    "best_macro_f1": "",
    "best_diff_epoch": "",
    "log_path": log_path,
}
if result_json:
    try:
        data = json.loads(result_json)
        for key in ("micro_f1", "macro_f1", "best_micro_f1", "best_macro_f1", "best_diff_epoch"):
            row[key] = data.get(key, "")
    except json.JSONDecodeError:
        pass

writer = csv.DictWriter(sys.stdout, fieldnames=list(row.keys()))
writer.writerow(row)
PY

  return "${rc}"
}

worker() {
  local shard="$1"
  local gpu="${GPUS[$shard]}"
  local results_csv="${OUT_DIR}/results_cuda${gpu}.csv"
  local worker_log="${OUT_DIR}/worker_cuda${gpu}.log"
  local trial_idx=0

  echo "return_code,trial,gpu,scenario,lr,alpha,eta,diffusion_steps,micro_f1,macro_f1,best_micro_f1,best_macro_f1,best_diff_epoch,log_path" > "${results_csv}"

  {
    echo "[worker] cuda:${gpu} shard=${shard}/${#GPUS[@]} start=$(date '+%Y-%m-%dT%H:%M:%S')"
    for scenario in "${SELECTED_SCENARIOS[@]}"; do
      for lr in "${LRS[@]}"; do
        for alpha in "${ALPHAS[@]}"; do
          for eta in "${ETAS[@]}"; do
            for steps in "${STEPS[@]}"; do
              if (( trial_idx % ${#GPUS[@]} == shard )); then
                echo "[trial] cuda:${gpu} idx=${trial_idx} scenario=${scenario} lr=${lr} alpha=${alpha} eta=${eta} T=${steps}"
                if ! run_trial "${gpu}" "${trial_idx}" "${scenario}" "${lr}" "${alpha}" "${eta}" "${steps}" >> "${results_csv}"; then
                  echo "[warn] cuda:${gpu} trial=${trial_idx} scenario=${scenario} failed"
                fi
              fi
              trial_idx=$((trial_idx + 1))
            done
          done
        done
      done
    done
    echo "[worker] cuda:${gpu} done=$(date '+%Y-%m-%dT%H:%M:%S')"
  } > "${worker_log}" 2>&1
}

select_scenarios
mkdir -p "${OUT_DIR}"
write_meta

total_trials=$((${#SELECTED_SCENARIOS[@]} * ${#LRS[@]} * ${#ALPHAS[@]} * ${#ETAS[@]} * ${#STEPS[@]}))
echo "[run] out_dir=${OUT_DIR}"
echo "[run] scenarios=${SELECTED_SCENARIOS[*]}"
echo "[run] gpus=${GPUS[*]}"
echo "[run] total_trials=${total_trials}"
echo "[run] resume_key=${RESUME_KEY}"

for shard in "${!GPUS[@]}"; do
  worker "${shard}" &
done

wait

merged="${OUT_DIR}/results_all.csv"
first=1
for gpu in "${GPUS[@]}"; do
  csv="${OUT_DIR}/results_cuda${gpu}.csv"
  if [[ ! -f "${csv}" ]]; then
    continue
  fi
  if (( first )); then
    cat "${csv}" > "${merged}"
    first=0
  else
    tail -n +2 "${csv}" >> "${merged}"
  fi
done

python - "${merged}" "${OUT_DIR}/summary_best.csv" "${OUT_DIR}/summary_best.json" <<'PY'
import csv
import json
import sys

results_path, summary_csv_path, summary_json_path = sys.argv[1:]


def to_float(value):
    if value in (None, ""):
        return None
    try:
        return float(value)
    except ValueError:
        return None


rows = []
try:
    with open(results_path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
except FileNotFoundError:
    rows = []

best_by_scenario = {}
for row in rows:
    if str(row.get("return_code", "")) != "0":
        continue

    best_micro = to_float(row.get("best_micro_f1"))
    best_macro = to_float(row.get("best_macro_f1"))
    final_micro = to_float(row.get("micro_f1"))
    final_macro = to_float(row.get("macro_f1"))

    score_micro = best_micro if best_micro is not None else final_micro
    score_macro = best_macro if best_macro is not None else final_macro
    if score_micro is None or score_macro is None:
        continue

    score = score_micro + score_macro
    scenario = row.get("scenario", "")
    candidate = {
        "scenario": scenario,
        "selection_score": score,
        "selected_micro_f1": score_micro,
        "selected_macro_f1": score_macro,
        "best_micro_f1": best_micro,
        "best_macro_f1": best_macro,
        "best_diff_epoch": row.get("best_diff_epoch", ""),
        "final_micro_f1": final_micro,
        "final_macro_f1": final_macro,
        "lr": to_float(row.get("lr")),
        "alpha": to_float(row.get("alpha")),
        "eta": to_float(row.get("eta")),
        "diffusion_steps": int(float(row["diffusion_steps"])) if row.get("diffusion_steps") else None,
        "trial": int(float(row["trial"])) if row.get("trial") else None,
        "gpu": row.get("gpu", ""),
        "log_path": row.get("log_path", ""),
    }
    current = best_by_scenario.get(scenario)
    if current is None or candidate["selection_score"] > current["selection_score"]:
        best_by_scenario[scenario] = candidate

summary = [best_by_scenario[key] for key in sorted(best_by_scenario)]
fieldnames = [
    "scenario",
    "selection_score",
    "selected_micro_f1",
    "selected_macro_f1",
    "best_micro_f1",
    "best_macro_f1",
    "best_diff_epoch",
    "final_micro_f1",
    "final_macro_f1",
    "lr",
    "alpha",
    "eta",
    "diffusion_steps",
    "trial",
    "gpu",
    "log_path",
]

with open(summary_csv_path, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(summary)

with open(summary_json_path, "w", encoding="utf-8") as f:
    json.dump(summary, f, indent=2, ensure_ascii=False)

print(f"[summary] wrote {summary_csv_path}")
print(f"[summary] wrote {summary_json_path}")
PY

echo "[done] all workers finished"
echo "[done] merged results: ${merged}"
echo "[done] best summary: ${OUT_DIR}/summary_best.csv"
