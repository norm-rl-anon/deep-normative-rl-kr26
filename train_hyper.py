from copy import deepcopy
from stable_baselines3 import DQN, PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv, VecVideoRecorder
from stable_baselines3.common.evaluation import evaluate_policy
from os import getenv
from optuna.samplers import TPESampler
from dotenv import load_dotenv
from datetime import datetime
from pprint import pformat
from numpy import mean
import argparse
import optuna
import torch

from utils.images import ImageFeaturesExtractor, compute_dfa_nr_states
from utils.misc import pooled, valid_json_file
from utils.stats import PacmanEvaluationStats, MerchantEvaluationStats, EvaluationStats
from utils.envs import create_pacman_env, create_merchant_env, create_test_env
from utils.hyperparams import hyperparameters
from algorithms.qlearning import QLearning
from normbases import norm_eval, normbase

# some fixed parameters that can be changed only via environment variables
load_dotenv(".env")
OPTUNA_DB = getenv("OPTUNA_DB", None)
TIME_LIMIT = getenv("TIME_LIMIT", None)
TRAIN_NUM_ENVS = 1  # int(getenv("TRAIN_NUM_ENVS", 1)) TODO
EVAL_NUM_ENVS = 1  # int(getenv("EVAL_NUM_ENVS", 1)) TODO
EVAL_INTER_COUNT = int(getenv("EVAL_INTER_COUNT", 10))
EVAL_INTER_EPISODES = int(getenv("EVAL_INTER_EPISODES", 100))
EVAL_FINAL_EPISODES = int(getenv("EVAL_FINAL_EPISODES", 1000))
NORMS_USE_LOG_SAMPLING = bool(getenv("NORMS_USE_LOG_SAMPLING", True))
NORMS_USE_SINGLE_OBJECTIVE = bool(getenv("NORMS_USE_SINGLE_OBJECTIVE", True))
NORMS_ORDERING_BASE = float(getenv("NORMS_ORDERING_BASE", 10.0))
VERBOSE = int(getenv("VERBOSE", 0))

# these probably should be set if we are using a SLURM batch
STUDY_NAME = getenv("STUDY_NAME", None)  # auto generated name if not set
OPT_NUM_TRIALS = int(getenv("OPT_NUM_TRIALS", 1))  # if we want to run multiple trials in parallel with one call
OPT_NUM_PROC = int(getenv("OPT_NUM_PROC", 1))  # number of simultaneous trials with multiprocessing

# some connection settings for postgres
ENGINE_KW = {
    "pool_pre_ping": True,
    "pool_recycle": 600,  # refresh before typical NAT/firewall idles
    "pool_timeout": 60,  # wait for a good conn during hiccups
    "pool_size": 1,  # per-process pool (we use processes, not threads)
    "max_overflow": 1,
}


def norm_weights_hp(trial, norm_base, range, log):
    if "dfa" not in norm_base:
        return
    weights = {}
    for dfa in norm_base["dfa"]:
        if norm_base["dfa"][dfa] is None:
            norm_base["dfa"][dfa] = trial.suggest_float(f"dfa_{dfa}", range[0], range[1], log=log)
        weights[dfa] = norm_base["dfa"][dfa]
    for dfa in norm_base["dfa"]:
        if type(norm_base["dfa"][dfa]) is str:
            assert norm_base["dfa"][dfa] in weights
            norm_base["dfa"][dfa] = weights[norm_base["dfa"][dfa]]


def train(args, hyperparams, trial):
    # check if we are using the GPU
    if torch.cuda.is_available():
        print(f"Using GPU {torch.cuda.current_device()}", flush=True)
    else:
        print("Using CPU", flush=True)

    # setup training and evaluation environments
    if args.env_use_images:
        image_stack_size = hyperparams["policy_kwargs"]["features_extractor_kwargs"]["image_stack_size"]
        greyscale = hyperparams["policy_kwargs"]["features_extractor_kwargs"]["greyscale"]
        hyperparams["policy_kwargs"]["features_extractor_class"] = ImageFeaturesExtractor
    else:
        image_stack_size = 1
        greyscale = True

    make_env = {"pacman": create_pacman_env, "merchant": create_merchant_env, "test": create_test_env}[args.env]
    model_env = make_vec_env(
        make_env,
        n_envs=args.train_num_envs,
        env_kwargs=dict(
            use_images=args.env_use_images,
            norm_base=args.norm_base,
            layout=args.env_layout,
            time_limit=args.env_time_limit,
            image_stack_size=image_stack_size,
            greyscale=greyscale,
        ),
        vec_env_cls=SubprocVecEnv if args.train_num_envs > 1 else DummyVecEnv,
    )
    eval_env = make_vec_env(
        make_env,
        n_envs=args.eval_num_envs,
        env_kwargs=dict(
            use_images=args.env_use_images,
            norm_base=args.norm_base,
            layout=args.env_layout,
            time_limit=args.env_time_limit,
            image_stack_size=image_stack_size,
            greyscale=greyscale,
        ),
        vec_env_cls=SubprocVecEnv if args.eval_num_envs > 1 else DummyVecEnv,
    )
    if args.env_use_images:
        single_env = model_env.envs[0]
        if not isinstance(single_env.unwrapped.features, str):
            hyperparams["policy_kwargs"]["features_extractor_kwargs"]["add_disc_features"] = compute_dfa_nr_states(
                single_env.unwrapped.features.dfa_list
            )
    # we use this both for intermediate and final evaluation
    eval_class = {
        "pacman": PacmanEvaluationStats,
        "merchant": MerchantEvaluationStats,
        "test": EvaluationStats,
    }[args.env]
    evaluation_stats = eval_class(
        trial,
        eval_env,
        train_envs=model_env if args.train_algo == "Q" else None,
        n_eval_episodes=args.eval_final_episodes,
        monitor_names=args.norm_base.get("monitors", []),
        int_eval_episodes=args.eval_inter_episodes,
        csv_prefix=args.load_model if args.load_model is not None else args.study_name if args.log_csv else None,
        int_eval_frequency=args.train_steps // args.eval_inter_count,
    )

    results = []
    for i in range(args.seeds):  # TODO: use reproducible seeds instead of random seeds
        print(f"--- Seed {i} ---", flush=True)

        # and use trial hyperparameters for model (assumes high-level features for now)
        if args.train_algo == "DQN":
            model = DQN("MlpPolicy", model_env, **hyperparams, verbose=VERBOSE)
        elif args.train_algo == "PPO":
            model = PPO("MlpPolicy", model_env, **hyperparams, verbose=VERBOSE)  # TODO: **hyperparams
        elif args.train_algo == "Q":
            model = QLearning(model_env, **hyperparams, verbose=VERBOSE)
        else:
            raise Exception(f"no such algorithm: {args.train_algo}")

        # load the model from file if explicitly specified
        if args.load_model is not None:
            assert args.seeds == 1
            assert not args.save_models
            model.set_parameters(args.load_model)
            eval_env.reset()
        # otherwise train the policy
        else:
            evaluation_stats.init_training_step(model)
            model.learn(args.train_steps, callback=evaluation_stats.learning_curve_callback(model))

        # evaluate the policy
        evaluation_stats.init_eval_step(model)
        mean_reward, std_reward = evaluate_policy(
            model, eval_env, n_eval_episodes=args.eval_final_episodes, callback=evaluation_stats.eval_callback
        )

        filename_prefix = f"{args.study_name}_{trial.number:03}_{i:02}"

        # maybe save the policy
        if args.save_models:
            model.save(f"{filename_prefix}.model")

        # maybe make a video
        if args.save_video:
            render_env = DummyVecEnv(
                [
                    lambda: make_env(
                        use_images=args.env_use_images,
                        norm_base=args.norm_base,
                        layout=args.env_layout,
                        time_limit=args.env_time_limit,
                        image_stack_size=image_stack_size,
                        greyscale=True if args.env_use_images else False,
                        render=True,
                    )
                ]
            )
            evaluate_policy(model, render_env, n_eval_episodes=1)
            render_env = VecVideoRecorder(
                render_env,
                "./",
                record_video_trigger=lambda x: x == 0,
                video_length=2000,
                name_prefix=args.load_model if args.load_model is not None else filename_prefix,
            )
            render_env.frames_per_sec = render_env.frames_per_sec // 2
            evaluate_policy(model, render_env, n_eval_episodes=15)
            render_env.close()

        print(f"--> seed {i}: {mean_reward} +/- {std_reward}", flush=True)
        results.append(mean_reward)

    avg_reward = mean(results)
    max_reward = max(results)

    print(f"==> average reward: {avg_reward}", flush=True)
    print(f"==> maximal reward: {max_reward}", flush=True)

    multi_objective_score = evaluation_stats.annotate_trial(
        trial, args.norm_base["order"] if args.norms_tune is not None else None
    )

    if args.norms_tune is None:
        return avg_reward
    elif args.norms_use_single_objective:
        norm_violations = list(zip(args.norm_base["order"], multi_objective_score[:-1]))
        score = multi_objective_score[-1]
        result = norm_eval(args.env, norm_violations, score, args.norms_ordering_base)
        print(norm_violations, score, result)
        return result
    else:
        return multi_objective_score


def objective(args):
    def result(trial):
        if args.hp_file is None:
            args.hp_file = {}
        hyperparams = hyperparameters(trial, args)
        norm_weights_hp(trial, args.norm_base, args.norms_tune, args.norms_use_log_sampling)
        return train(args, hyperparams, trial)

    return result


def make_storage():
    if OPTUNA_DB is not None:
        if OPTUNA_DB == "DISCARD":
            return optuna.storages.InMemoryStorage()
        # Robust Postgres URL with keepalives + generous connect timeout
        DB_URL = (
            OPTUNA_DB
            + "?connect_timeout=60"
            + "&keepalives=1&keepalives_idle=60&keepalives_interval=20&keepalives_count=6"
        )
        return optuna.storages.RDBStorage(
            url=DB_URL,
            engine_kwargs=ENGINE_KW,
            heartbeat_interval=None,  # heartbeat disabled
            grace_period=None,
        )
    else:
        # Minimal SQLite fallback (simple file, no keepalive/engine knobs)
        return "sqlite:///experiment-results.db"


def run_experiments(args):
    if args.norms_tune is None or args.norms_use_single_objective:
        directions = ["maximize"]
    else:
        directions = ["minimize"] * len(args.norm_base["order"]) + ["maximize"]

    study = optuna.create_study(
        study_name=args.study_name,
        storage=make_storage(),
        directions=directions,
        sampler=TPESampler(n_startup_trials=10, multivariate=True, constant_liar=True),
        load_if_exists=True,
    )

    # set user attributes, check whether they match for existing studies
    if len(study.user_attrs) == 0:
        for k, v in sorted(args.info.items()):
            study.set_user_attr(k, v)
    else:
        assert study.user_attrs == args.info

    o = objective(args)
    study.optimize(o, 1)  # only one run, multiple runs are dealt with multiprocessing or slurm batches


def study_name(args):
    timestr = datetime.now().strftime("%d%m%y-%H%M%S")
    imagestr = "_images" if args.env_use_images else ""
    normtunestr = "_tune" if args.norms_tune is not None else ""
    hptunestr = "_tune" if args.hp_file is None and args.load_model is None else ""
    return f"{args.env}{imagestr}-{args.env_layout}-{args.norm_base_name}{normtunestr}-{args.train_algo}{hptunestr}-{args.train_steps}steps-{timestr}"


if __name__ == "__main__":
    # command line arguments are used to set study parameters
    parser = argparse.ArgumentParser(description="Training Policies with HPO")
    parser.add_argument("--env", choices=["merchant", "pacman", "test"], default="pacman", help="Gym Environment")
    parser.add_argument("--layout", dest="env_layout", default=None, help="Environment Layout")
    parser.add_argument(
        "--use-images", dest="env_use_images", action="store_true", help="Learn directly on image inputs"
    )
    parser.add_argument(
        "--norm-base",
        dest="norm_base_name",
        metavar="BASE",
        type=str,
        default="default",
        help="Norm base (default = no norms)",
    )
    parser.add_argument(
        "--norms-tune",
        nargs=2,
        metavar=("MIN", "MAX"),
        type=float,
        help="Use hyperparameter tuning for norms with given range",
    )
    parser.add_argument(
        "--algo", dest="train_algo", choices=["Q", "DQN", "PPO"], default="DQN", help="Learning Algorithm"
    )
    parser.add_argument("--load-model", type=str, help="Load a model from file instead of training")
    parser.add_argument("--save-models", action="store_true", help="Saves all models after training")
    parser.add_argument("--save-video", action="store_true", help="Saves a video models after training (or loading)")
    parser.add_argument("--steps", dest="train_steps", type=int, default=1_000_000, help="Number of training steps")
    parser.add_argument("--seeds", type=int, default=1, help="Average over this number of seeds")
    parser.add_argument("--hp-file", type=valid_json_file, help="Load the hyperparameters from file (otherwise tune)")
    parser.add_argument("--log-csv", action="store_true", help="Write CSV statistics for every evaluation episode")
    args = parser.parse_args()

    args.env_time_limit = TIME_LIMIT
    args.train_num_envs = TRAIN_NUM_ENVS
    args.eval_num_envs = EVAL_NUM_ENVS
    args.eval_inter_count = EVAL_INTER_COUNT
    args.eval_inter_episodes = EVAL_INTER_EPISODES
    args.eval_final_episodes = EVAL_FINAL_EPISODES
    args.norms_use_log_sampling = NORMS_USE_LOG_SAMPLING
    args.norms_use_single_objective = NORMS_USE_SINGLE_OBJECTIVE
    args.norms_ordering_base = NORMS_ORDERING_BASE

    # use default layout if not explicitly specified
    if args.env == "test":  # frozenlake for simple testing
        assert args.env_layout is None
        assert args.env_use_images is False
    if args.env == "pacman":
        if args.env_layout is None:
            args.env_layout = "smallClassic"
        if args.env_time_limit is None:
            args.env_time_limit = 300
    elif args.env == "merchant":
        if args.env_layout is None:
            args.env_layout = "basic"
        if args.env_time_limit is None:
            args.env_time_limit = 50
        assert args.env_use_images is False

    # use default normbase if not explicitly specified
    args.norm_base = normbase(args.env, args.norm_base_name, args.norms_tune)

    # we summarize the relevant parameters and create a nice name
    args.info = deepcopy(vars(args))
    args.study_name = STUDY_NAME if STUDY_NAME is not None else study_name(args)

    # we write all this together with the experiment results into a database
    print(f"Writing experiment results to:\n  {OPTUNA_DB}/{args.study_name}", flush=True)
    # we also print the relevant parameters onto the terminal
    print(f"Parameters:\n{pformat(args.info)}", flush=True)

    # we run the experiments
    pooled(OPT_NUM_TRIALS, OPT_NUM_PROC, run_experiments, args)
