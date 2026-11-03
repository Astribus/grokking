#!/bin/bash
# Launch the structured-holdout experiment batch: 15 runs, 1 CPU thread each, all in parallel.
# Usage: bash scripts/run_batch.sh     (logs go to logs/<run>.log)
cd "$(dirname "$0")/.."
{
  for split in nowrap triangle holdout_answer holdout_a; do
    for seed in 0 1 2; do echo "--split $split --seed $seed --run ${split}_s$seed"; done
  done
  echo "--split random --seed 1 --run random_s1"
  echo "--split random --seed 2 --run random_s2"
  echo "--split random --seed 0 --wd 0 --run wd0"
} | xargs -P 15 -I{} sh -c 'uv run python -m grokking.train --steps 20000 --threads 1 {} > logs/$(echo "{}" | sed "s/.*--run //").log 2>&1'
echo ALL DONE
