#!/usr/bin/env bash
jid=$(date +%s)
for t in $(seq "$1" "$2"); do
  SLURM_JOB_ID=$jid SLURM_ARRAY_JOB_ID=$jid SLURM_ARRAY_TASK_ID=$t bash "$3"; echo "${jid}_$t"
done
