from norm_rl_gym.automata.automaton import DFA


def make_dfa_list(dfa_spec, env):
    # all dfas defined in this module
    dfas = {name: obj for name, obj in globals().items() if isinstance(obj, type) and issubclass(obj, DFA)}
    result = []
    for name, value in dfa_spec.items():
        result.append(dfas[name](env.labels, value))
    return result


# NORMS


# F(!rain & X(rain) & !X(warn))


class WarningDFA(DFA):
    def __init__(self, alph, reward=0):
        super().__init__(alph, -reward, reset="achievement")
        self.states = [0, 1, 2]
        self.final = [2]

    def transition(self, inpt, state=None):
        if state is None:
            state = self.state
        if state == 0:
            if "rain" in inpt:
                return 0
            else:
                return 2
        if state == 1:
            if "rain" not in inpt:
                return 1
            elif "warn" in inpt:
                return 0
            else:
                return 2
        if state == 2:
            return 2


# F()
class WarningDFA(DFA):
    def __init__(self, alph, reward=0):
        super().__init__(alph, -reward, reset="achievement")
        self.states = [0, 1, 2, 3, 4, 5]
        self.final = [5]

    def transition(self, inpt, state=None):
        if state is None:
            state = self.state
        if state == 0:
            if "rain" in inpt and "hurricane" not in inpt:
                return 1
            else:
                return 0
        if state == 1:
            if "hurricane" in inpt and "atShelter" not in inpt:
                return 2
            elif "rain" in inpt and "hurricane" not in inpt:
                return 0
            else:
                return 2
        if state == 2:
            if "atShelter" not in inpt:
                return 3
            elif "rain" in inpt and "hurricane" not in inpt:
                return 1
            else:
                return 0
        if state == 3:
            if "atShelter" not in inpt:
                return 4
            elif "rain" in inpt and "hurricane" not in inpt:
                return 1
            else:
                return 0
        if state == 4:
            if "atShelter" not in inpt:
                return 5
            elif "rain" in inpt and "hurricane" not in inpt:
                return 1
            else:
                return 0
        if state == 5:
            return 5


# F
def transition(self, inpt, state=None):
    if state is None:
        state = self.state
    if state == 0:
        if "rain" not in inpt and "hurricane" not in inpt:
            return 1
        else:
            return 0
    if state == 1:
        if "hurricane" in inpt and "atShelter" not in inpt:
            return 2
        elif "rain" not in inpt and "hurricane" not in inpt:
            return 0
        else:
            return 2
    if state == 2:
        if "atShelter" not in inpt:
            return 3
        elif "rain" not in inpt and "hurricane" not in inpt:
            return 1
        else:
            return 0
    if state == 3:
        if "atShelter" not in inpt:
            return 4
        elif "rain" not in inpt and "hurricane" not in inpt:
            return 1
        else:
            return 0
    if state == 4:
        if "atShelter" not in inpt:
            return 5
        elif "rain" not in inpt and "hurricane" not in inpt:
            return 1
        else:
            return 0
    if state == 5:
        if "atShelter" not in inpt:
            return 6
        elif "rain" not in inpt and "hurricane" not in inpt:
            return 1
        else:
            return 0
    if state == 6:
        if "atShelter" not in inpt:
            return 7
        elif "rain" not in inpt and "hurricane" not in inpt:
            return 1
        else:
            return 0
    if state == 7:
        if "atShelter" not in inpt:
            return 8
        elif "rain" not in inpt and "hurricane" not in inpt:
            return 1
        else:
            return 0
    if state == 8:
        if "atShelter" not in inpt:
            return 9
        elif "rain" not in inpt and "hurricane" not in inpt:
            return 1
        else:
            return 0
    if state == 9:
        return 9


# F(hurricane & !(shelter | (home & flood)) U !hurricane)
