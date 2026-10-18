#!/usr/bin/env bash
# Usage: scripts/run_pilot.sh /workspace/models/Qwen3-1.7B
# Env: PYTHON (default python), JUDGE_DIR (default /workspace/models/Qwen3Guard-Gen-4B), FILLER (default 1024)
set -u
cd "$(dirname "$0")/.."
export PYTHONPATH=.
MODEL_DIR=$1
PYTHON=${PYTHON:-python}
JUDGE_DIR=${JUDGE_DIR:-/workspace/models/Qwen3Guard-Gen-4B}
F=${FILLER:-1024}
M=$(basename "$MODEL_DIR")
R=results/$M
mkdir -p "$R/markers" "$R/logs"

# step <name> <required 1|0> <cmd...>: skip if .done; a failed required step stops the queue
step() {
  local name=$1 required=$2; shift 2
  if [ -f "$R/markers/$name.done" ]; then echo "skip $name"; return 0; fi
  echo "start $name $(date +%FT%T%z)"
  rm -f "$R/markers/$name.failed"
  if "$@" > "$R/logs/$name.log" 2>&1; then
    touch "$R/markers/$name.done"; echo "done $name"
  else
    touch "$R/markers/$name.failed"; echo "FAILED $name (see $R/logs/$name.log)"
    [ "$required" = 1 ] && exit 1
  fi
}

run() { "$PYTHON" -m kvsafety.pipeline.run --model-dir "$MODEL_DIR" --filler-tokens "$F" "$@"; }

step data 1 "$PYTHON" -m kvsafety.pipeline.data
# DENSE=0 skips the uncompressed arms on boxes that only run compressed arms
if [ "${DENSE:-1}" = 1 ]; then
  step dense_train 1 run --set train --method dense
  step dense_val 1 run --set val --method dense
  step dense_eval 1 run --set eval --method dense
fi
# ARMS: compressed arms to run on this box, e.g. "snapkv_2 snapkv_4" (default: all four)
ARMS=${ARMS:-snapkv_2 snapkv_4 jolt_2 jolt_4}
for arm in $ARMS; do
  step "${arm%_*}_eval_r${arm#*_}" 0 run --set eval --method "${arm%_*}" --ratio "${arm#*_}"
done

for name in dense_eval snapkv_eval_r2 snapkv_eval_r4 jolt_eval_r2 jolt_eval_r4; do
  case $name in
    dense_eval) run_file=eval_dense_r1_f$F ;;
    *) m=${name%%_*}; r=${name##*_r}; run_file=eval_${m}_r${r}_f$F ;;
  esac
  [ -f "$R/markers/$name.done" ] && step "judge_$name" 0 "$PYTHON" -m kvsafety.pipeline.judge --judge-dir "$JUDGE_DIR" "$R/$run_file.jsonl"
done

step analyze 0 "$PYTHON" -m kvsafety.pipeline.analyze --results-dir "$R" --filler-tokens "$F"
