import gymnasium as gym
from gymnasium.wrappers import TransformReward, FrameStackObservation

from utils.images import WarpFrame


def create_pacman_env(env_name, feature_extractor, level, render_mode, greyscale=True, image_stack_size=1, scale=False):
    env = gym.make(env_name, layout=level, features=feature_extractor, render_mode=render_mode)
    if scale:
        env = TransformReward(env, lambda r: r / 10)
    if "image" in feature_extractor:
        env = WarpFrame(env, additional_discrete_features=0, greyscale=greyscale)
        env = FrameStackObservation(env, image_stack_size)
    else:
        return env
    return env
