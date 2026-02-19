from norm_rl_gym.monitors.pacman_monitors import make_pacman_monitor
from norm_rl_gym.monitors.merchant_monitors import make_merchant_monitor
from norm_rl_gym.monitors.taxi_monitors import make_taxi_monitor
from stable_baselines3.common.callbacks import EvalCallback
from utils.csvwriter import StatsWriter
from sys import stdout
import numpy as np


class LearningCurveStats(EvalCallback):
    def __init__(
        self,
        trial,
        eval_env,
        n_eval_episodes,
        eval_freq,
        train_env=None,
    ):
        super().__init__(
            eval_env=eval_env,
            n_eval_episodes=n_eval_episodes,
            eval_freq=eval_freq,
            deterministic=True,
        )
        self.eval_env = eval_env
        self.train_env = train_env
        self.trial = trial
        self.results = []

    def set_model(self, model):
        self.model = model

    def next(self):
        self.results.append([])

    def annotate(self):
        if self.trial is None:
            return
        for i, x in enumerate(map(lambda x: sum(x) / len(self.results), zip(*self.results))):
            try:
                self.trial.report(x, i + 1)
            except NotImplementedError:
                self.trial.set_user_attr(f"return_{i + 1:02}", x)

    def _on_step(self):
        if self.eval_freq > 0 and self.n_calls % self.eval_freq == 0:
            if self.train_env is not None:
                self.model.register_vec_env(self.eval_env)
            super()._on_step()
            stdout.flush()
            if self.train_env is not None:
                self.model.register_vec_env(self.train_env)
            self.results[-1].append(self.last_mean_reward)
        return True


class EvaluationStats:
    def __init__(
        self,
        trial,
        eval_envs,
        n_eval_episodes,
        monitor_names,
        int_eval_episodes,
        int_eval_frequency,
        csv_prefix,
        train_envs,
    ):
        self.eval_envs = eval_envs
        self.train_envs = train_envs
        self.n_eval_episodes = n_eval_episodes
        self.n_evals = 0
        self.monitor_names = monitor_names
        self.lcs = LearningCurveStats(trial, eval_envs, int_eval_episodes, int_eval_frequency, train_env=train_envs)
        self.writer = (
            StatsWriter(csv_prefix, trial.number if trial is not None else 0) if csv_prefix is not None else None
        )
        self.monitored_stats = {}

    def get_stats(self, n_episodes):
        return {k: v / n_episodes for k, v in self.monitored_stats.items()}

    def learning_curve_callback(self, model):
        self.lcs.set_model(model)
        return self.lcs

    def init_training_step(self, model):
        if self.train_envs is not None:
            model.register_vec_env(self.train_envs)
        self.lcs.next()

    def init_eval_step(self, model):
        self.lcs.n_calls = 0
        self.lcs.num_timesteps = 0
        self.lcs.best_mean_reward = -np.inf
        self.lcs.last_mean_reward = -np.inf
        if self.train_envs is not None:
            model.register_vec_env(self.eval_envs)
        self.n_evals += 1

    def eval_callback(self, locals, gobals):
        pass

    def annotate_trial(self, trial, order=None) -> tuple[float] | None:
        self.lcs.annotate()


class PacmanEvaluationStats(EvaluationStats):
    def __init__(
        self,
        trial,
        eval_envs,
        n_eval_episodes,
        monitor_names,
        int_eval_episodes,
        int_eval_frequency,
        csv_prefix,
        train_envs=None,
    ):
        super().__init__(
            trial,
            eval_envs,
            n_eval_episodes,
            monitor_names,
            int_eval_episodes,
            int_eval_frequency,
            csv_prefix,
            train_envs,
        )
        self.pm_score = 0
        self.eaten_blue = 0
        self.eaten_red = 0
        self.left_food = 0
        self.lost = 0
        self.won = 0

    def init_eval_step(self, model):
        super().init_eval_step(model)
        self.monitors = [
            [make_pacman_monitor(mn, e.unwrapped) for mn in self.monitor_names] for e in self.eval_envs.envs
        ]
        first_run = self.monitored_stats == {}
        for i in range(len(self.eval_envs.envs)):
            for monitor in self.monitors[i]:
                if first_run and i == 0:
                    self.monitored_stats.update(monitor.export())
                monitor.reset()
        self.prev_games = [e.unwrapped.game for e in self.eval_envs.envs]

    def eval_callback(self, locals, gobals):
        i_env = locals["i"]
        env = locals["env"].envs[i_env].unwrapped
        for monitor in self.monitors[i_env]:
            monitor.detectViolation(
                self.prev_games[i_env].state,
                # locals["states"][i_env] would not work in a final state
                locals["actions"][i_env],
            )
        if locals["done"]:
            # have to use prev_game to not get fresh initial state
            final_state = self.prev_games[i_env].state
            # global stats
            eaten_blue, eaten_red = final_state.getGhostsEaten()
            self.pm_score += final_state.getScore()
            self.eaten_blue += eaten_blue
            self.eaten_red += eaten_red
            self.left_food += final_state.getNumFood()
            self.lost += final_state.isLose()
            self.won += final_state.isWin()
            # output to csv
            if self.writer is not None:
                self.writer.write_trial(
                    self.monitors[i_env],
                    {
                        "Seed": self.n_evals - 1,
                        "Score": final_state.getScore(),
                        "Blue Eaten": eaten_blue,
                        "Orange Eaten": eaten_red,
                        "Win/Lose": "win" if final_state.isWin() else ("lose" if final_state.isLose() else "timeout"),
                    },
                )
            # monitor stats
            for monitor in self.monitors[i_env]:
                monitor_stats = monitor.export()
                for k, v in monitor_stats.items():
                    self.monitored_stats[k] += v
                monitor.reset()
        self.prev_games[i_env] = env.game

    def eval_callback2(self, locals, globals):
        print(locals["env"].envs[0].unwrapped.state)

    def annotate_trial(self, trial, order=None) -> tuple[float] | None:
        super().annotate_trial(trial)
        if trial is None:
            return
        trial.set_user_attr("pm_score", self.pm_score / self.n_eval_episodes / self.n_evals)
        trial.set_user_attr("pm_eaten_blue", self.eaten_blue / self.n_eval_episodes / self.n_evals)
        trial.set_user_attr("pm_eaten_red", self.eaten_red / self.n_eval_episodes / self.n_evals)
        trial.set_user_attr("pm_left_food", self.left_food / self.n_eval_episodes / self.n_evals)
        trial.set_user_attr("pm_lost", self.lost / self.n_eval_episodes / self.n_evals)
        trial.set_user_attr("pm_won", self.won / self.n_eval_episodes / self.n_evals)
        for k, v in self.monitored_stats.items():
            trial.set_user_attr(f"norm_{k}", v / self.n_eval_episodes / self.n_evals)
        if order is not None:
            return tuple(self.monitored_stats[n] / self.n_eval_episodes / self.n_evals for n in order) + (
                self.pm_score / self.n_eval_episodes / self.n_evals,
            )


class MerchantEvaluationStats(EvaluationStats):
    def __init__(
        self,
        trial,
        eval_envs,
        n_eval_episodes,
        monitor_names,
        int_eval_episodes,
        int_eval_frequency,
        csv_prefix,
        train_envs,
    ):
        super().__init__(
            trial,
            eval_envs,
            n_eval_episodes,
            monitor_names,
            int_eval_episodes,
            int_eval_frequency,
            csv_prefix,
            train_envs,
        )
        self.m_score = 0

    def init_eval_step(self, model):
        super().init_eval_step(model)
        self.monitors = [
            [make_merchant_monitor(mn, e.unwrapped) for mn in self.monitor_names] for e in self.eval_envs.envs
        ]
        first_run = self.monitored_stats == {}
        for i in range(len(self.eval_envs.envs)):
            for monitor in self.monitors[i]:
                if first_run and i == 0:
                    self.monitored_stats.update(monitor.export())
                monitor.reset()

    def eval_callback(self, locals, gobals):
        i_env = locals["i"]
        env = locals["env"].envs[i_env].unwrapped
        self.m_score += env.unwrapped.reward
        for monitor in self.monitors[i_env]:
            monitor.detectViolation(
                env.state_or_final,
                env.action,
            )
        if locals["done"]:
            # monitor stats
            for monitor in self.monitors[i_env]:
                monitor_stats = monitor.export()
                for k, v in monitor_stats.items():
                    self.monitored_stats[k] += v
                monitor.reset()

    def annotate_trial(self, trial, order=None) -> tuple[float] | None:
        super().annotate_trial(trial)
        if trial is None:
            return
        trial.set_user_attr("m_score", self.m_score / self.n_eval_episodes / self.n_evals)
        for k, v in self.monitored_stats.items():
            trial.set_user_attr(f"norm_{k}", v / self.n_eval_episodes / self.n_evals)
        if order is not None:
            return tuple(self.monitored_stats[n] / self.n_eval_episodes / self.n_evals for n in order) + (
                self.m_score / self.n_eval_episodes / self.n_evals,
            )


class TaxiEvaluationStats(EvaluationStats):
    def __init__(
        self,
        trial,
        eval_envs,
        n_eval_episodes,
        monitor_names,
        int_eval_episodes,
        int_eval_frequency,
        csv_prefix,
        train_envs,
    ):
        super().__init__(
            trial,
            eval_envs,
            n_eval_episodes,
            monitor_names,
            int_eval_episodes,
            int_eval_frequency,
            csv_prefix,
            train_envs,
        )

    def init_eval_step(self, model):
        super().init_eval_step(model)
        self.monitors = [[make_taxi_monitor(mn, e.unwrapped) for mn in self.monitor_names] for e in self.eval_envs.envs]
        first_run = self.monitored_stats == {}
        for i in range(len(self.eval_envs.envs)):
            for monitor in self.monitors[i]:
                if first_run and i == 0:
                    self.monitored_stats.update(monitor.export())
                monitor.reset()

    def eval_callback(self, locals, gobals):
        i_env = locals["i"]
        env = locals["env"].envs[i_env].unwrapped
        for monitor in self.monitors[i_env]:
            monitor.detectViolation(
                env.s,
                env.lastaction,
            )
        if locals["done"]:
            # monitor stats
            for monitor in self.monitors[i_env]:
                monitor_stats = monitor.export()
                for k, v in monitor_stats.items():
                    self.monitored_stats[k] += v
                monitor.reset()

    def annotate_trial(self, trial, order=None) -> tuple[float] | None:
        super().annotate_trial(trial)
        if trial is None:
            return
        for k, v in self.monitored_stats.items():
            trial.set_user_attr(f"norm_{k}", v / self.n_eval_episodes / self.n_evals)
        if order is not None:
            return tuple(self.monitored_stats[n] / self.n_eval_episodes / self.n_evals for n in order)
