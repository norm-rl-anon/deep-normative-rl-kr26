from norm_rl_gym.automata.automaton import DFA


def make_dfa_list(dfa_spec, env):
    # all dfas defined in this module
    dfas = {name: obj for name, obj in globals().items() if isinstance(obj, type) and issubclass(obj, DFA)}
    result = []
    for name, value in dfa_spec.items():
        result.append(dfas[name](env.labels, value))
    return result


class DangerDFA(DFA):
    def __init__(self, alphabet, reward=0):
        super().__init__(alphabet, -reward, reset="punctual", permission=False)
        self.states = [0, 1]
        self.final = [1]

    def transition(self, inpt, state=None):
        if state is None:
            state = self.state
        if state == 0:
            if "atDanger" in inpt:
                return 1
            else:
                return 0
        if state == 1:
            return 1


class PassiveDFA(DFA):
    def __init__(self, alphabet, reward=0):
        super().__init__(alphabet, -reward, reset="punctual", permission=False)
        self.states = [0, 1]
        self.final = [1]

    def transition(self, inpt, state=None):
        if state is None:
            state = self.state
        if state == 0:
            if "Fight" in inpt:
                return 1
            else:
                return 0
        if state == 1:
            return 1


class EnvFriendlyDFA(DFA):
    def __init__(self, alphabet, reward=0):
        super().__init__(alphabet, -reward, reset="punctual", permission=False)
        self.states = [0, 1]
        self.final = [1]

    def transition(self, inpt, state=None):
        if state is None:
            state = self.state
        if state == 0:
            if "atTree" in inpt and "Extract" in inpt:
                return 1
            else:
                return 0
        if state == 1:
            return 1


# O^A_sundown(atMarket | atHome)
# F(atHome & !atMarket U sundown)
class DeliveryDFA(DFA):
    def __init__(self, alphabet, reward=0):
        super().__init__(alphabet, -reward, reset="achievement", permission=False)
        self.states = [0, 1, 2]
        self.final = [2]

    def transition(self, inpt, state=None):
        if state is None:
            state = self.state
        if state == 0:
            if "atHome" in inpt and "sundown" in inpt:
                return 2
            elif "atHome" in inpt and "atMarket" not in inpt:
                return 1
            else:
                return 0
        if state == 1:
            if "sundown" in inpt:
                return 2
            elif "atMarket" not in inpt:
                return 1
            else:
                return 0
        if state == 2:
            return 2


# TODO: permission DFAs
class EnvPerm(DFA):
    def __init__(self, alphabet, reward=0):
        super().__init__(alphabet, reward, reset="punctual", permission=True)
        self.states = [0, 1]
        self.final = [1]

    def transition(self, inpt, state=None):
        if state is None:
            state = self.state
        if state == 0:
            if "atTree" in inpt and "extract" in inpt and "hasWood" not in inpt:
                return 1
            else:
                return 0
        if state == 1:
            return 1
