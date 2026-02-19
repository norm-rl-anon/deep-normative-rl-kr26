#!/usr/bin/env bash
#SBATCH --job-name=evl-fast
#SBATCH --array=0-19%20     # create 20 tasks 0-19, each
#SBATCH --cpus-per-task=1   # * running on 1 CPU core
#SBATCH --mem=4G            # * reserving 4 GB RAM
#SBATCH --time=23:59:59     # * having a timeout of one day (usually takes much less)

# tasks will not be further parallelized
export OPT_NUM_PROC=1
export OPT_NUM_TRIALS=1

# read parameters from environment variables
ALGO="${ALGO:-PPO}"              # the algoright (PPO or DQN)
STEPS="${STEPS:-2500000}"        # number of steps per trial
NORM_BASE="${NORM_BASE:-vegan}"  # norm base to use and monitor

export STUDY_NAME=hl_${NORM_BASE}_${ALGO}_${STEPS}
export EVAL_FINAL_EPISODES=10000

if [ $SLURM_ARRAY_TASK_ID -lt 20 ]; then sleep $(( 10 * SLURM_ARRAY_TASK_ID )); fi

python ../train_hyper.py --algo ${ALGO} --norm-base ${NORM_BASE} --seeds 1 --steps ${STEPS} --save-models --save-video --log-csv
