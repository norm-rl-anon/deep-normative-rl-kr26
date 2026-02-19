import numpy as np
from pickle import dump, load
from typing import Any, Optional, Tuple
from gymnasium.spaces import Discrete
from collections import defaultdict
from stable_baselines3.common.logger import configure
from stable_baselines3.common.vec_env import DummyVecEnv


class QLearning:
    def __init__(
        self,
        vec_env,
        learning_rate: float = 0.2,
        gamma: float = 0.99,
        exploration_fraction: float = 0.1,
        exploration_initial_eps: float = 1.0,
        exploration_final_eps: float = 0.1,
        seed: Optional[int] = None,
        verbose: int = 0,
        tensorboard_log: Optional[str] = None,
    ):
        self.vec_env = vec_env
        self.num_envs = vec_env.num_envs
        self.lr = float(learning_rate)
        self.gamma = float(gamma)
        self.exploration_fraction = float(exploration_fraction)
        self.eps_start = float(exploration_initial_eps)
        self.eps_final = float(exploration_final_eps)
        self.verbose = verbose
        self.log_interval = 10000
        self.use_masking = isinstance(vec_env, DummyVecEnv) and hasattr(
            vec_env.envs[0].unwrapped, "exclActions"
        )  # TODO: improve API

        # for logging
        self._log_ep_rets = []
        self._log_ep_lens = []
        self._log_ep_ret = np.zeros(self.num_envs, dtype=np.float32)
        self._log_ep_len = np.zeros(self.num_envs, dtype=np.int32)
        self._log_td_running = []
        self._log_t_last = None

        # check parameters
        assert 0 <= self.lr <= 1
        assert 0 <= self.gamma <= 1
        assert 0 <= self.exploration_fraction <= 1
        assert 0 <= self.eps_start <= 1
        assert 0 <= self.eps_final <= 1

        # setup logging
        logging = []
        if self.verbose >= 1:
            logging.append("stdout")
        if self.verbose >= 2:
            self.log_interval = 1000
        if tensorboard_log is not None:
            logging += ["csv", "tensorboard"]
        self.logger = configure(tensorboard_log, logging)
        self._no_logging = len(logging) == 0

        # assumes action space is Discrete(n)
        if isinstance(vec_env.action_space, Discrete):
            self.nA = vec_env.action_space.n
        else:
            raise Exception("Action space must be Discrete(n).")
        # also assumes state space is hashable (no check)

        # Q table and episode tracking
        self._q = defaultdict(lambda: np.zeros(self.nA))
        self._rng = np.random.default_rng(seed)
        self.num_timesteps = 0
        self.num_episodes = 0
        self._last_obs = None

    # ---------- SB3-like API ----------
    def learn(
        self,
        total_timesteps: int,
        reset_num_timesteps: bool = True,
        callback: Optional[Any] = None,
    ) -> "QLearning":
        if reset_num_timesteps:
            self.num_timesteps = 0
            self.num_episodes = 0

        # learning loop
        obs = self.vec_env.reset()
        if callback is not None:
            callback.model = self
            callback.on_rollout_start()
        while self.num_timesteps < total_timesteps:
            self._epsilon = self._current_epsilon(total_timesteps)
            actions, _ = self.predict(obs, deterministic=False)
            new_obs, rewards, dones, infos = self.vec_env.step(actions)
            terminal = dones & ~np.array([x["TimeLimit.truncated"] for x in infos])
            # Q-learning update (vanilla, online).
            self._update(obs, actions, rewards, new_obs, terminal)
            self.num_timesteps += self.num_envs
            # Handle episode ends per env
            if np.any(dones):
                self.num_episodes += int(np.sum(dones))
            self._logging(rewards, dones, self._epsilon)
            obs = new_obs
            if callback is not None:
                callback.update_locals(locals())
                if not callback.on_step():
                    break
        if callback is not None:
            callback.on_rollout_end()
        return self

    def predict(
        self,
        observation,
        state: Optional[tuple[np.ndarray, ...]] = None,
        episode_start: Optional[np.ndarray] = None,
        deterministic: bool = False,
    ) -> Tuple[np.ndarray, Optional[Any]]:
        eps = 0.0 if deterministic else self._epsilon
        actions = self._epsilon_greedy(observation, eps)
        return actions.astype(int), state

    def get_vec_normalize_env(self):
        # neccessary for callbacks to work
        return None

    def register_vec_env(self, vec_env):
        self.vec_env = vec_env

    def save(self, filename):
        # convert defaultdict to plain dict for pickling
        q_dict = dict(self._q)
        with open(filename, "wb") as f:
            dump(q_dict, f)

    def set_parameters(self, filename):
        with open(filename, "rb") as f:
            q_dict = load(f)
        # restore as defaultdict to keep the zero‑init behaviour
        new_q = defaultdict(lambda: np.zeros(self.nA))
        for state, q_values in q_dict.items():
            new_q[state] = q_values
        self._q = new_q

    # ---------- Internals ----------
    def _current_epsilon(self, total_timesteps):
        if self.exploration_fraction == 0:
            return self.eps_final
        decay_steps = self.exploration_fraction * total_timesteps
        fraction = np.clip(self.num_timesteps / decay_steps, 0.0, 1.0)
        return self.eps_start + fraction * (self.eps_final - self.eps_start)

    def _q_np(self, obs, mask=None):
        matrix = [self._q[tuple(o)] for o in obs]
        if mask is not None:
            return np.ma.array(matrix, mask=mask)
        return np.array(matrix)

    def _epsilon_greedy(self, obs: np.ndarray, eps: float) -> np.ndarray:
        # obtain masked actions if possible TODO: improve
        mask = np.zeros((self.num_envs, self.nA), dtype=bool)
        if self.use_masking:
            for i, env in enumerate(self.vec_env.envs):
                excl = env.unwrapped.exclActions()
                assert len(excl) < self.nA
                mask[i, excl] = True
        masked_q = self._q_np(obs, mask)
        greedy = np.ma.argmax(masked_q, axis=1).astype(int)
        random_mask = self._rng.random(self.num_envs) < eps
        random_actions = np.array([self._rng.choice(np.flatnonzero(~row)) for row in mask])
        actions = np.where(random_mask, random_actions, greedy)
        if self.verbose > 2:
            print("---")
            print(f"obs: {obs}")
            print(f"mask: {mask}")
            print(f"masked_q: {masked_q}")
            print(f"greedy: {greedy}")
            print(f"randoms: {random_actions}")
            print(f"random_mask: {random_mask}")
            print(f"actions: {actions}")
        return actions.astype(int)

    def _update(self, s, a, r, s2, terminal):
        for i in range(self.num_envs):
            si, ai, ri, s2i, ti = tuple(s[i]), a[i], r[i], tuple(s2[i]), terminal[i]
            future_q_value = (not ti) * np.max(self._q[s2i])
            temporal_difference = ri + self.gamma * future_q_value - self._q[si][ai]
            self._q[si][ai] += self.lr * temporal_difference
            self._log_td_running.append(temporal_difference)

    def _logging(self, rewards, dones, eps):
        if self._no_logging:
            return
        # otherwise log everything
        self._log_ep_ret += rewards.astype(np.float32)
        self._log_ep_len += 1
        if np.any(dones):
            finished = np.where(dones)[0]
            self._log_ep_rets += tuple(self._log_ep_ret[finished])
            self._log_ep_lens += tuple(self._log_ep_len[finished])
            self._log_ep_ret[finished] = 0.0
            self._log_ep_len[finished] = 0
        if self.num_timesteps % self.log_interval == 0:
            self.logger.record("rollout/ep_rew_mean", float(np.mean(self._log_ep_rets)))
            self.logger.record("rollout/ep_len_mean", float(np.mean(self._log_ep_lens)))
            self.logger.record("time/total_timesteps", int(self.num_timesteps))
            self.logger.record("time/episodes", int(self.num_episodes))
            td_arr = np.asarray(self._log_td_running, dtype=np.float32)
            self.logger.record("train/explored_states", len(self._q))
            self.logger.record("train/td_error_mean", float(td_arr.mean()))
            self.logger.record("train/td_error_std", float(td_arr.std()))
            self.logger.record("train/td_error_abs_mean", float(np.abs(td_arr).mean()))
            self.logger.record("train/epsilon", float(eps))
            self.logger.dump(self.num_timesteps)
            self._log_td_running.clear()
            self._log_ep_rets.clear()
            self._log_ep_lens.clear()
