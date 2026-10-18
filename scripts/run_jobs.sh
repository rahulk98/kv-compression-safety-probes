#!/usr/bin/env bash
# Usage: scripts/run_jobs.sh <model_name> <jobs_file>
# Each line of jobs_file: <step_name> <python -m module args...>. Steps are marker-gated; then every eval file is judged.
set -u
cd "$(dirname "$0")/.."
export PYTHONPATH=.
M=$1; JOBS=$2
P=${PYTHON:-/workspace/.venv/bin/python}
JUDGE_DIR=${JUDGE_DIR:-/workspace/models/Qwen3Guard-Gen-4B}
R=results/$M; mkdir -p "$R/markers" "$R/logs"
[ -f data/sets/eval.jsonl ] || "$P" -m kvsafety.pipeline.data > "$R/logs/data.log" 2>&1
md5sum data/sets/*.jsonl > "$R/logs/sets.md5"
while read -r name cmd; do
  [ -z "$name" ] && continue
  if [ -f "$R/markers/$name.done" ]; then echo "skip $name"; continue; fi
  echo "start $name $(date +%FT%T%z)"
  if eval "$P -m $cmd" > "$R/logs/$name.log" 2>&1; then touch "$R/markers/$name.done"; echo "done $name"
  else touch "$R/markers/$name.failed"; echo "FAILED $name"; fi
done < "$JOBS"
echo "start judge $(date +%FT%T%z)"
files=$(ls $R/eval_*_f1024.jsonl 2>/dev/null | grep -v judged)
"$P" -m kvsafety.pipeline.judge --judge-dir "$JUDGE_DIR" $files > "$R/logs/judge.log" 2>&1 && echo "done judge" || echo "FAILED judge"
echo ALL_DONE
