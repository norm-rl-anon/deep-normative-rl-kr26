coords = {"red": (0, 4), "green": (4, 4), "yellow": (0, 0), "blue": (3, 0)}

colors = {0: "red", 1: "green", 2: "yellow", 3: "blue"}


def get_shelter(state):
    return state % 4


def get_floodrisk(state):
    state = state // 4
    return state % 2


def get_home(state):
    state = state // 4
    state = state // 2
    return state % 4


def get_hurricane(state):
    state = state // 4
    state = state // 2
    state = state // 4
    return state % 11


def get_rain(state):
    state = state // 4
    state = state // 2
    state = state // 4
    state = state // 11
    return state % 2


def get_dest(state):
    state = state // 4
    state = state // 2
    state = state // 4
    state = state // 11
    state = state // 2
    return state % 4


def get_passenger(state):
    state = state // 4
    state = state // 2
    state = state // 4
    state = state // 11
    state = state // 2
    state = state // 4
    return state % 5


def get_x(state):
    state = state // 4
    state = state // 2
    state = state // 4
    state = state // 11
    state = state // 2
    state = state // 4
    state = state // 5
    return state % 5


def get_y(state):
    state = state // 4
    state = state // 2
    state = state // 4
    state = state // 11
    state = state // 2
    state = state // 4
    state = state // 5
    state = state // 5
    return state


class Labels:
    def __init__(self, constitutive=False):
        self.c = constitutive
        self.labels = ["hurricane", "floodrisk", "rain", "atHome", "atShelter", "newHurricane"]

    # NEED TO CHANGE TAXI CODE

    def isEmergency(self, state, action):
        if self.c:
            pass
        else:
            return self.hurricane(state, action)

    def isRisk(self, state, action):
        if self.c:
            pass
        else:
            return self.rain(state, action)

    def warn(self, state, action):
        return action == 6

    def floodrisk(self, state, action):
        return get_floodrisk(state) == 1

    def atHome(self, state, action):
        return get_home(state) == get_passenger(state)

    def atSafety(self, state, action):
        return (self.atHome(state, action) and not self.floodrisk(state, action)) or self.atShelter(state, action)

    def atShelter(self, state, action):
        return get_shelter(state) == get_passenger(state)

    def atBlue(self, state, action):
        return (get_x(state), get_y(state)) == coords[colors[3]]

    def atGreen(self, state, action):
        return (get_x(state), get_y(state)) == coords[colors[1]]

    def atRed(self, state, action):
        return (get_x(state), get_y(state)) == coords[colors[0]]

    def atYellow(self, state, action):
        return (get_x(state), get_y(state)) == coords[colors[2]]

    def atDestination(self, state, action):
        return (get_x(state), get_y(state)) == coords[colors[get_dest(state)]]

    def hasPassenger(self, state, action):
        return get_passenger(state) == 4

    def hurricane(self, state, action):
        return get_hurricane(state) > 0

    def newHurricane(self, state, action):
        return get_hurricane(state) == 1

    def rain(self, state, action):
        return get_rain(state) == 1

    def getLabels(self, state, action):
        labels = [action]
        for l in self.labels:
            fn = getattr(self, l)
            if fn(state, action):
                labels.append(l)
        return labels
