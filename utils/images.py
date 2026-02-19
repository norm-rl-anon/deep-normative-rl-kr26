import gymnasium as gym
import numpy as np
import torch
import cv2

from torch import nn
from gymnasium import spaces


from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

import torch.nn.functional as F


def compute_dfa_nr_states(dfa_list):
    nr_states = 0
    for dfa in dfa_list:
        nr_states += len(dfa.states)
    return nr_states


class WarpFrame(gym.ObservationWrapper[np.ndarray, int, np.ndarray]):
    def __init__(self, env: gym.Env, additional_discrete_features=0, factor=3, greyscale=True) -> None:
        super().__init__(env)
        self.width = int(env.observation_space.shape[0] / factor)
        self.height = int(env.observation_space.shape[1] / factor)
        assert isinstance(env.observation_space, spaces.Box), f"Expected Box space, got {env.observation_space}"

        self.channels = 1 if greyscale else 3
        if additional_discrete_features:
            self.channels += 1
        self.i = 0
        self.observation_space = spaces.Box(
            low=0,
            high=255,
            shape=(self.height, self.width, self.channels),
            dtype=env.observation_space.dtype,  # type: ignore[arg-type]
        )
        self.greyscale = greyscale

    def observation(self, frame: np.ndarray) -> np.ndarray:
        has_discrete = frame.shape[2] == 4
        if has_discrete:
            discrete_features = frame[:, :, -1]
            frame = frame[:, :, :-1]
        if self.greyscale:
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)

        frame = cv2.resize(frame, (self.width, self.height), interpolation=cv2.INTER_AREA)

        if self.greyscale:
            frame = frame[:, :, None]
        if has_discrete:
            discrete_features_resized = discrete_features[: self.height, : self.width]
            discrete_features_resized = np.expand_dims(discrete_features_resized, axis=2)
            frame = np.concatenate((frame, discrete_features_resized), axis=2)
        return frame


class ImageFeaturesExtractor(BaseFeaturesExtractor):
    # based on NatureCNN
    def __init__(
        self,
        observation_space: gym.Space,
        features_dims: list,
        greyscale: bool,
        image_stack_size: int,
        add_disc_features=0,
        disc_feature_index=1,
    ) -> None:
        super().__init__(
            observation_space,
            features_dims[-1] if disc_feature_index < len(features_dims) else features_dims[-1] + add_disc_features,
        )
        # We  work with CxHxW images (channels first)
        assert 0 <= disc_feature_index <= len(features_dims)
        print(observation_space)
        print(self.features_dim)
        self.greyscale = greyscale
        self.image_stack_size = image_stack_size
        color_channels = 1 if greyscale else 3
        self.n_input_channels = color_channels * self.image_stack_size
        self.normalize = True
        self.large_cnn = nn.Sequential(
            nn.Conv2d(self.n_input_channels, 32, kernel_size=8, stride=4, padding=0),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=0),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=0),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten(),
        )
        self.smaller_cnn = nn.Sequential(
            nn.Conv2d(self.n_input_channels, 32, kernel_size=8, stride=4, padding=0),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=0),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=0),
            nn.ReLU(),
            nn.Flatten(),
        )
        self.disc_feature_index = disc_feature_index
        # Compute shape by doing one forward pass
        with torch.no_grad():
            sample_observation_space = spaces.Box(
                low=0, high=255, shape=(self.n_input_channels, observation_space.shape[1], observation_space.shape[2])
            )
            try:
                n_flatten = self.large_cnn(torch.as_tensor(sample_observation_space.sample()[None]).float()).shape[1]
                self.cnn = self.large_cnn
                print(n_flatten)
            except:
                n_flatten = self.smaller_cnn(torch.as_tensor(sample_observation_space.sample()[None]).float()).shape[1]
                self.cnn = self.smaller_cnn
                print("Using smaller CNN: ", n_flatten)
        self.linear_layers = nn.ModuleList()
        previous_dim = n_flatten

        for index, feat_dim in enumerate(features_dims):
            if self.disc_feature_index == index:
                previous_dim += add_disc_features

            layer = nn.Linear(previous_dim, feat_dim)
            # if use_cuda:
            #     layer = layer.to("cuda")
            self.linear_layers.append(layer)
            previous_dim = feat_dim

        self.add_disc_features = add_disc_features
        # self.linear = nn.Sequential(*linear_layers)

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        if self.add_disc_features > 0:
            disc_features = observations[:, -1, :, :, -1]  # the most recent discrete features states
            batch_size = observations.shape[0]
            height = observations.shape[2]
            width = observations.shape[3]
            observations = observations[:, :, :, :, :-1]
            if self.normalize:
                observations /= 255
            observations = observations.permute(0, 1, 4, 2, 3).contiguous()

            observations = observations.view(batch_size, self.n_input_channels, height, width)
            # cnn_outputs = self.cnn(observations)
            # return self.linear(torch.concat((cnn_outputs,state_nrs),dim=1))
            state_nrs = disc_features[:, 0, : self.add_disc_features]

            features = self.cnn(observations)
            for index, layer in enumerate(self.linear_layers):
                if index == self.disc_feature_index:
                    features = F.relu(layer(torch.concat((features, state_nrs), dim=1)))
                else:
                    features = F.relu(layer(features))
            if self.disc_feature_index >= len(self.linear_layers):
                features = torch.concat((features, state_nrs), dim=1)
            return features
        else:
            batch_size = observations.shape[0]
            height = observations.shape[2]
            width = observations.shape[3]
            observations = observations.permute(0, 1, 4, 2, 3).contiguous()
            if self.normalize:
                observations /= 255

            observations = observations.view(batch_size, self.n_input_channels, height, width)
            features = self.cnn(observations)
            for layer in self.linear_layers:
                features = F.relu(layer(features))
            return features
