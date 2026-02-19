from copy import deepcopy
from norm_rl_gym.automata.merchant_DFAs import make_dfa_list
from norm_rl_gym.envs.pacman.pacman_env import PacmanEnv
from norm_rl_gym.envs.merchant.merchant import MerchantEnv
from gymnasium.wrappers import (
    TimeLimit,
    TransformReward,
    FrameStackObservation,
    FlattenObservation,
)
from norm_rl_gym.wrappers.rbWrapper import SimpleMerchantRBWrapper, SimplePacmanRBWrapper
from .images import WarpFrame, compute_dfa_nr_states
import gymnasium as gym


def feature_extractor(use_images, norm_base):
    feature_extractor_key = "feature_extractor_image" if use_images else "feature_extractor"
    features = norm_base.get(feature_extractor_key)
    print(f"feature_extractor: {features}")
    return features


def create_pacman_env(
    use_images,
    norm_base,
    layout,
    image_stack_size,
    greyscale,
    time_limit,
    render=False,
):
    features = feature_extractor(use_images, norm_base)
    env = PacmanEnv(
        layout=layout,
        features=features,
        dfas=norm_base.get("dfa"),
        render_mode="rgb_array" if render else None,
    )
    dfa_list = deepcopy(env.features.dfa_list) if "dfa" in norm_base and hasattr(env.features, "dfa_list") else []
    env = SimplePacmanRBWrapper(env, dfa_list=dfa_list)
    if True:  # normalize_rewards:
        env = TransformReward(env, lambda r: r / 100)
    if use_images:
        dfa = compute_dfa_nr_states(dfa_list)
        env = WarpFrame(env, additional_discrete_features=dfa, greyscale=greyscale, factor=3)
        env = FrameStackObservation(env, image_stack_size)
    if time_limit is not None:
        env = TimeLimit(env, max_episode_steps=time_limit)
    return env


def create_merchant_env(
    use_images,
    norm_base,
    layout,
    time_limit,
    **kwargs,
):
    assert not use_images
    env = MerchantEnv(layout=layout, risk_fight=1.0, risk_death=0.0)
    dfa_list = make_dfa_list(norm_base.get("dfa", {}), env)
    env = SimpleMerchantRBWrapper(env, dfa_list=dfa_list)
    env = FlattenObservation(env)
    env = TimeLimit(env, max_episode_steps=time_limit)
    return env


def create_test_env(**kwargs):
    env = gym.make("FrozenLake-v1", is_slippery=True)
    env = FlattenObservation(env)
    return env
