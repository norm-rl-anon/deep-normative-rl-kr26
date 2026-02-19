from norm_rl_gym.monitors.monitor import Monitor


def make_merchant_monitor(name, env):
    # all monitors defined in this module
    monitors = {name: obj for name, obj in globals().items() if isinstance(obj, type) and issubclass(obj, Monitor)}
    # now return the appropriate one
    if name is None:
        return Monitor(env)
    elif name in monitors:
        return monitors[name](env)
    # or
    else:
        print(f"ERROR. No such merchant monitor: {name}.\n  Available monitors:")
        print("  " + "\n  ".join(sorted(monitors)))
        raise SystemExit


class DangerMonitor(Monitor):
    def __init__(self, env, verbose=False):
        super().__init__(env, verbose)

    def detectViolation(self, state, action):
        if self.labels.atDanger(state, action):
            self.violations += 1
            if self.verbose:
                print("Danger Prohibition Violated!")
            return True
        else:
            return False


class PacifistMonitor(Monitor):
    def __init__(self, env, verbose=False):
        self.danger = DangerMonitor(env)
        self.ctdViolations = 0
        super().__init__(env, verbose)

    def detectViolation(self, state, action):
        dan = self.danger.detectViolation(state, action)
        if dan:
            self.violations += 1
            if action != 6:
                self.ctdViolations += 1
                self.violations += 1
            return True
        return False

    # TODO:
    def reset(self):
        self.danger.reset()
        self.ctdViolations = 0
        super().reset()

    def export(self):
        exp = {
            self.__class__.__name__[:-7] + "(total)": self.violations,
            # self.danger.__class__.__name__[:-7]: self.danger.violations,
            # "CTD": self.ctdViolations,
        }
        return exp


class DeliveryMonitor(Monitor):
    def __init__(self, env, verbose=False):
        self.oblInForce = False
        super().__init__(env, verbose)

    def detectViolation(self, state, action):
        if self.labels.atHome(state, action):
            self.oblInForce = True
        if self.oblInForce and self.labels.sundown(state, action):
            self.violations += 1
            if self.verbose:
                print("Failed to make delivery!")
            self.oblInForce = False
            return True
        if self.labels.atMarket(state, action):
            self.oblInForce = False
        return False

    def reset(self):
        self.oblInForce = False
        super().reset()


class EnvFriendlyMonitor(Monitor):
    def __init__(self, env, verbose=False):
        self.justAtTree = 0
        super().__init__(env, verbose)

    def detectViolation(self, state, action):
        if self.labels.hasWood(state, action) and self.labels.atTree(state, action):
            self.justAtTree = 1
        if self.justAtTree != 0 and self.justAtTree < 2 and action == "extract":
            self.violations += 1
            self.justAtTree = 0
            return True
        return False

    def reset(self):
        self.justAtTree = 0
        super().reset()


class ObstacleMonitor(Monitor):
    def __init__(self, env, verbose=False):
        self.danger = DangerMonitor(env)
        self.delivery = DeliveryMonitor(env)
        super().__init__(env, verbose)

    def detectViolation(self, state, action):
        dan = self.danger.detectViolation(state, action)
        deliv = self.delivery.detectViolation(state, action)
        self.violations = self.danger.violations + self.delivery.violations
        if dan or deliv:
            self.violations += 1
            return True
        return False

    def reset(self):
        self.danger.reset()
        self.delivery.reset()
        super().reset()

    def export(self):
        exp = {
            # self.danger.__class__.__name__[:-7]: self.danger.violations,
            # self.delivery.__class__.__name__[:-7]: self.delivery.violations,
            self.__class__.__name__[:-7] + "(total)": self.violations,
        }
        return exp


class EvolvingMonitor(Monitor):
    def __init__(self, env, verbose=False):
        self.counter = 0
        self.justAtTree = 0
        super().__init__(env, verbose)

    def detectViolation(self, state, action):
        self.counter += 1
        if self.counter >= 15:
            if self.labels.hasWood(state, action) and self.labels.atTree(state, action):
                self.justAtTree = 1
            if self.justAtTree != 0 and self.justAtTree < 2 and action == "extract":
                self.violations += 1
                self.justAtTree = 0
                return True
        else:
            if self.labels.atTree(state, action):
                self.justAtTree = 1
            if self.justAtTree != 0 and self.justAtTree < 2 and action == "extract":
                self.violations += 1
                self.justAtTree = 0
                return True
        return False

    def reset(self):
        self.justAtTree = 0
        self.counter = 0
        super().reset()
