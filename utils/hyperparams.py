from .images import ImageFeaturesExtractor


def hyperparameters(trial, args):
    if args.train_algo == "Q":
        hyperparams = tabq_hyperparams(trial, args)
    elif args.train_algo == "DQN":
        hyperparams = dqn_hyperparams(trial, args)
    elif args.train_algo == "PPO":
        hyperparams = ppo_hyperparams(trial, args)
    else:
        raise Exception(f"No hyperparameter search configuration for {args.train_algo}.")
    print(hyperparams)
    return hyperparams


def hp_select(fixed, trial):
    def hps(name, sampling_function, user_attr_name=None):
        if name in fixed:
            return fixed[name]
        else:
            value = sampling_function()
            if user_attr_name is not None:
                trial.set_user_attr(user_attr_name, value)
            return value

    return hps


def tabq_hyperparams(trial, args):
    return args.hp_file
    hp = hp_select(args.hp_file, trial)
    learning_rate = hp("learning_rate", lambda: trial.suggest_float("qlearning_lr", 0.05, 1.0))
    gamma = hp("gamma", lambda: trial.suggest_float("qlearning_gamma", 0.8, 0.995))
    exploration_initial_eps = hp(
        "exploration_initial_eps", lambda: trial.suggest_float("qlearning_eps_initial", 0.25, 1.0)
    )
    exploration_final_eps = hp("exploration_final_eps", lambda: trial.suggest_float("qlearning_eps_final", 0.0, 0.25))
    exploration_fraction = hp("exploration_fraction", lambda: trial.suggest_float("qlearning_eps_fraction", 0.0, 1.0))

    return dict(
        learning_rate=learning_rate,
        gamma=gamma,
        exploration_initial_eps=exploration_initial_eps,
        exploration_final_eps=exploration_final_eps,
        exploration_fraction=exploration_fraction,
    )


def dqn_hyperparams(trial, args):
    return args.hp_file
    images = args.env_use_images
    hp = hp_select(args.hp_file, trial)
    # Common, broad ranges for learning rate, discount factor, and batch size
    learning_rate = hp("learning_rate", lambda: trial.suggest_float("dqn_lr", 3e-5, 3e-4, log=True))
    gamma = hp("gamma", lambda: 1 - trial.suggest_float("dqn_one_minus_gamma", 5e-3, 5e-2, log=True), "dqn_gamma")
    batch_size = hp("batch_size", lambda: 2 ** trial.suggest_int("dqn_log_batch_size", 4, 6), "dqn_batch_size")
    # Use a wide but RAM-friendly buffer span
    buffer_size = hp("buffer_size", lambda: 2 ** trial.suggest_int("dqn_log_buffer_size", 14, 17), "dqn_buffer_size")
    # Warmup: keep range broad to cover highlevel features / pixels
    learning_starts = hp(
        "learning_starts", lambda: 2 ** trial.suggest_int("dqn_log_learning_starts", 11, 15, 2), "dqn_learning_starts"
    )
    # Update cadence: on low-FPS grids, frequent updates help
    train_freq = hp("train_freq", lambda: trial.suggest_categorical("dqn_train_freq", [4] if images else [1, 2, 4]))
    gradient_steps = hp("gradient_steps", lambda: trial.suggest_categorical("dqn_gradient_steps", [1, 2, 4]))
    # Stability / targets
    target_update_interval = hp(
        "target_update_interval",
        lambda: 2 ** trial.suggest_int("dqn_log_target_update_interval", 10, 14),
        "dqn_target_update_interval",
    )
    # n_steps = trial.suggest_int("dqn_n_steps", 1, 5, step=2)  # multi-step returns
    # ε-greedy schedule
    exploration_initial_eps = hp("exploration_initial_eps", lambda: trial.suggest_float("dqn_eps_initial", 0.1, 1.0))
    exploration_final_eps = hp("exploration_final_eps", lambda: trial.suggest_float("dqn_eps_final", 0.0, 0.1))
    exploration_fraction = hp("exploration_fraction", lambda: trial.suggest_float("dqn_eps_fraction", 0.0, 1.0))
    # Policy & extractor (most impactful knobs only)
    policy_kwargs = {}
    hp = hp_select(args.hp_file.get("policy_kwargs", {}), trial)
    if images:
        features_dims = hp(
            "features_dims",
            lambda: list(
                map(
                    int,
                    trial.suggest_categorical("dqn_features_dims", ["256-256", "512-256", "512-512"]).split("-"),
                )
            ),
        )
        greyscale = hp("greyscale", lambda: trial.suggest_categorical("dqn_greyscale", [False, True]))
        image_stack_size = hp(
            "image_stack_size", lambda: 2 ** trial.suggest_int("dqn_log_image_stack_size", 0, 1), "dqn_image_stack_size"
        )
        policy_kwargs["features_extractor_class"] = ImageFeaturesExtractor
        policy_kwargs["features_extractor_kwargs"] = {
            "features_dims": features_dims,
            "dfa_states": 0,  # if args.use_norms else 0,  # compute_dfa_nr_states(...), TODO
            "greyscale": greyscale,
            "image_stack_size": image_stack_size,
        }
    else:
        # Compact MLPs are usually enough; keep options broad
        net_arch = hp(
            "net_arch",
            lambda: list(
                map(
                    int,
                    trial.suggest_categorical("dqn_mlp_arch", ["256-256", "512-256", "512-512"]).split("-"),
                )
            ),
        )
        policy_kwargs = {"net_arch": net_arch}

    return dict(
        # policy=policy,
        learning_rate=learning_rate,
        gamma=gamma,
        batch_size=batch_size,
        buffer_size=buffer_size,
        learning_starts=learning_starts,
        train_freq=train_freq,
        gradient_steps=gradient_steps,
        target_update_interval=target_update_interval,
        exploration_initial_eps=exploration_initial_eps,
        exploration_final_eps=exploration_final_eps,
        exploration_fraction=exploration_fraction,
        tau=1.0,  # hard target update (standard DQN)
        # optimize_memory_usage=args.env_use_images,
        policy_kwargs=policy_kwargs,
    )


def ppo_hyperparams(trial, args):
    # TODO: implement reasonable hyperparameter ranges
    return args.hp_file


def _divisors(n, lo=64, hi=2048):
    return [d for d in [64, 96, 128, 192, 256, 384, 512, 768, 1024, 1536, 2048] if d <= hi and d >= lo and (n % d == 0)]
