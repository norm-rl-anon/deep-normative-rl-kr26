Structure of this repository
============================

* Selected videos are located in `videos-best/`.
* Experiment scripts are located in `experiments/`.
* Results and evaluation scripts are located in `evaluation/`.
* The main Python file for running individual experiments is `train_hyper.py`.


Prerequisites
=============

To run the scripts, create and activate a conda environment with the required packages:

    conda env create -n pacman -f environment.yml
    conda activate pacman


Exploring the weight tuning results
===================================

To inspect the results of the norm-weight optimization via hyperparameter tuning, activate the conda environment, navigate to `evaluation/`, and run:

    optuna-dashboard sqlite:///hpo-experiments.db

This starts a local web server for interactively exploring the weight tuning results. Open http://127.0.0.1:8080/ in your browser.


Recreating the plots and evaluation
===================================

To generate the preliminary (learning budget) plots, navigate to `evaluation/1_preliminary/` and run:

    python plot.py


To generate the weight tuning plots and table, navigate to `evaluation/2_weights/` and run:

    python plot1.py   # norm bases with 1 weight
    python plot2.py   # norm bases with 2 weights
    python table3.py  # HungryPenaltyVegan


To generate the sensitivity analysis plots, navigate to `evaluation/3_sensitivity_analysis/` and run:

    python plot.py


To evaluate the policies, navigate to `evaluation/4_policies/` and run:

    python analyze.py --algo ALGO --data DATA

where `ALGO` is either `PPO` or `DQN`, and `DATA` is either `hl` (high-level features) or `images`.


Running the experiments from scratch
====================================

To run the experiments from scratch, you will realistically need a SLURM cluster with many CPU cores for the high-level features experiments and some GPUs for the image-based experiments.

Activate the conda environment and run the following commands from within the `experiments/` folder:

* `ALGO=xxx NORM_BASE=yyy STEPS=zzz sbatch pacman_highlevel_tune.slurm` to optimize the Pacman high-level norm weights.
* `ALGO=xxx NORM_BASE=yyy STEPS=zzz sbatch pacman_highlevel_evaluate.slurm` to generate and evaluate (high-level features) policies for the norm weights reported in the paper (specified manually in `normbases.py`).
* `ALGO=xxx NORM_BASE=yyy STEPS=zzz sbatch pacman_images_evaluate.slurm` to generate and evaluate (image-based) policies for the norm weights reported in the paper (specified manually in `normbases.py`).

In these commands, replace `xxx` with either `DQN` or `PPO`. Replace `yyy` with one of `vegan`, `vegetarian`, `earlybird` (for the *Hungry* norm base), `contradiction` (for *HungryVegan*), `solution` (for *HungryVegetarian*), or `penalty1` (for *HungryPenaltyVegan*). Replace ```zzz``` with the number of training steps.

If you do not have access to a SLURM cluster (i.e., you cannot run `sbatch`), you can run `./without_slurm.sh MIN MAX xxx.slurm` instead of `sbatch xxx.slurm`, where `xxx.slurm` is the SLURM batch file and `MIN`/`MAX` are the minimum and maximum job indices for the job array. These indices are listed in the header of the respective SLURM file (e.g., for `pacman_highlevel_tune.slurm` you would use `MIN`=0 and `MAX`=99 to obtain 100 trials). Note that this runs all trials sequentially and will take **a lot** of time.