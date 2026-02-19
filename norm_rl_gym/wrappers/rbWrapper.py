from gymnasium import Wrapper
from gymnasium.spaces import Box
import numpy as np
from norm_rl_gym.envs.pacman.featureExtractors import features_dict_to_array, DeepRLLabelledCompleteExtractor
from norm_rl_gym.envs.pacman.pacman import ClassicGameRules
from norm_rl_gym.envs.pacman.pacmanAgents import OpenAIAgent
from norm_rl_gym.envs.pacman.ghostAgents import RandomGhost

PACMAN_ACTIONS = list(range(5))

PACMAN_DIRECTIONS = list(range(5))
MAX_GHOSTS = 5


class PacmanRBWrapper(Wrapper):
    def __init__(self, env, dfa_list=None):
        super().__init__(env)
        if dfa_list is None:
            self.dfa_list = []
        else:
            self.dfa_list = dfa_list
        self.observation_space = Box(
            self.observation_space.low[0],
            self.observation_space.high[0],
            (self.observation_space.shape[0] + len(self.dfa_list),),
        )
        self.env.observation_space = Box(
            self.env.observation_space.low[0],
            self.env.observation_space.high[0],
            (self.env.observation_space.shape[0] + len(self.dfa_list),),
        )

    def reset(self, layout=None, **kwargs):
        # get new layout
        # if self.layout is None:
        #    self.chooseLayout(randomLayout=True)

        self.env.step_counter = 0
        self.env.cum_reward = 0

        self.env.terminated = False
        self.env.truncated = False

        # we don't want super powerful ghosts
        # self.ghosts = [DirectionalGhost( i+1, prob_attack=0.2, prob_scaredFlee=0.2) for i in range(MAX_GHOSTS)]
        self.env.ghosts = [RandomGhost(i + 1) for i in range(MAX_GHOSTS)]

        # this agent is just a placeholder for graphics to work
        self.env.pacman = OpenAIAgent()

        self.env.rules = ClassicGameRules(300)
        self.env.rules.quiet = False

        self.env.game = self.env.rules.newGame(
            self.env.layout,
            self.env.pacman,
            self.env.ghosts,
            self.env.display,
            True,
            False,
            do_render=self.env.do_render,
        )

        self.env.game.init()

        if self.env.do_render:
            self.env.display.initialize(self.env.game.state.data)
            self.env.display.updateView()

        self.env.location = self.env.game.state.data.agentStates[0].getPosition()
        self.env.ghostLocations = [a.getPosition() for a in self.env.game.state.data.agentStates[1:]]
        self.env.ghostInFrame = any(
            [np.sum(np.abs(np.array(g) - np.array(self.env.location))) <= 2 for g in self.env.ghostLocations]
        )

        self.env.location_history = [self.env.location]
        self.env.orientation = PACMAN_DIRECTIONS.index(self.env.game.state.data.agentStates[0].getDirection())
        self.env.orientation_history = [self.env.orientation]
        self.env.illegal_move_counter = 0

        self.env.cum_reward = 0

        self.env.initial_info = {
            "step_counter": [[0]],
        }
        return self.obs(self.env.game.state, None), self.env.initial_info

    def step(self, action):
        rb = []
        fin = []
        for dfa in self.dfa_list:
            inpt = self.env.get_labels(action)
            state = dfa.transition(inpt)
            if state in dfa.final:
                rb.append(dfa.reward)
                fin.append(1.0)
                dfa.state = dfa.reset(dfa.state, state)
            else:
                dfa.state = state
                rb.append(0.0)
                fin.append(0.0)
        truncated = False
        if self.env.step_counter >= self.env.max_ep_len or self.env.terminated or self.env.truncated:
            self.env.step_counter += 1
            self.env.truncated = self.env.truncated or self.env.step_counter >= self.env.max_ep_len
            return (
                self.env.null_obs(),
                0.0,
                self.env.terminated,
                self.env.truncated,
                {
                    "step_counter": [[self.env.step_counter]],
                    "r": [self.env.cum_reward],
                    "l": [self.env.step_counter],
                    "episode": {"r": self.env.cum_reward, "l": self.env.step_counter},
                },
            )
        pacman_action = PACMAN_ACTIONS[action]
        legal_actions = self.env.game.state.getLegalPacmanActions()
        illegal_action = False
        if pacman_action not in legal_actions:
            self.env.illegal_move_counter += 1
            illegal_action = True
            pacman_action = 0  # Stop is always legal
        reward = self.env.game.step(pacman_action)
        self.env.cum_reward += reward

        terminated = self.env.game.state.isWin() or self.env.game.state.isLose()
        self.env.location = self.env.game.state.data.agentStates[0].getPosition()
        self.env.location_history.append(self.env.location)
        self.env.ghostLocations = [a.getPosition() for a in self.env.game.state.data.agentStates[1:]]

        self.env.orientation = PACMAN_DIRECTIONS.index(self.env.game.state.data.agentStates[0].getDirection())
        self.env.orientation_history.append(self.env.orientation)

        extent = (
            (self.env.location[0] - 1, self.env.location[1] - 1),
            (self.env.location[0] + 1, self.env.location[1] + 1),
        )
        self.env.ghostInFrame = any(
            [
                g[0] >= extent[0][0] and g[1] >= extent[0][1] and g[0] <= extent[1][0] and g[1] <= extent[1][1]
                for g in self.env.ghostLocations
            ]
        )
        self.env.step_counter += 1
        info = {
            "step_counter": [[self.env.step_counter]],
            "episode": None,
        }

        if self.env.step_counter >= self.env.max_ep_len:
            truncated = True
        self.env.terminated = terminated
        self.env.truncated = truncated
        if self.env.terminated or self.env.truncated:  # only if done, send 'episode' info
            info["episode"] = [{"r": self.env.cum_reward, "l": self.env.step_counter}]
        info["agent_eaten"] = self.env.game.state.data.agentEatenCnt

        observation = (self.obs(self.env.game.state, action, fin),)
        for r in rb:
            reward += r
        return observation, reward, terminated, truncated, info

    def obs(self, state, action, final=None):
        if final is None:
            final = list(np.zeros(len(self.dfa_list)))
        if self.env.features == "image":
            return self.env._get_image()
        else:
            if type(self.env.features) != DeepRLLabelledCompleteExtractor:
                raise TypeError
            feature_dict = self.env.features.getFeatures(state, action, final)
            print("FEATURE DICT", feature_dict)
            # print("LEN:", len(feature_dict.keys()))
            feature_array = features_dict_to_array(feature_dict)
            print(feature_array)
            return feature_array


class SimplePacmanRBWrapper(Wrapper):
    def __init__(self, env, dfa_list=None):
        super().__init__(env)
        if dfa_list is None:
            self.dfa_list = []
        else:
            self.dfa_list = dfa_list

    def reset(self, layout=None, **kwargs):
        obs, info = self.env.unwrapped.reset()
        inpt = self.env.unwrapped.get_labels()
        for dfa in self.dfa_list:
            dfa.state = 0
            state = dfa.transition(inpt)
            dfa.state = state
        return obs, info

    def step(self, action):
        observation, reward, terminated, truncated, info = self.env.unwrapped.step(action)
        inpt = self.env.unwrapped.get_labels()
        rb = []
        for dfa in self.dfa_list:
            state = dfa.transition(inpt)
            if state in dfa.final:
                rb.append(dfa.reward)
            else:
                dfa.state = state
                rb.append(0.0)
            dfa.state = dfa.reset(dfa.state, state)
        for r in rb:
            reward += r
        return observation, reward, terminated, truncated, info


class SimpleMerchantRBWrapper(Wrapper):
    def __init__(self, env, dfa_list=None):
        super().__init__(env)
        if dfa_list is None:
            self.dfa_list = []
        else:
            self.dfa_list = dfa_list

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        inpt = self.env.get_labels()
        for dfa in self.dfa_list:
            dfa.state = 0
            state = dfa.transition(inpt)
            dfa.state = state
        return obs, info

    def step(self, action):
        observation, reward, terminated, truncated, info = self.env.step(action)
        inpt = self.env.get_labels()
        rb = []
        for dfa in self.dfa_list:
            state = dfa.transition(inpt)
            if state in dfa.final:
                rb.append(dfa.reward)
            else:
                dfa.state = state
                rb.append(0.0)
            dfa.state = dfa.reset(dfa.state, state)
        for r in rb:
            reward += r
        return observation, reward, terminated, truncated, info
