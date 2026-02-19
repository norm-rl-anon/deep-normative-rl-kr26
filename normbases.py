from copy import deepcopy


def normbase(env, name, optimize=False):
    if env not in BASES:
        raise Exception(f"Unknown environment: {env}.")
    bases = BASES[env]
    if name not in bases:
        raise Exception(f"Unknown norm base: {name}. Available: {sorted(bases)}")
    base = deepcopy(bases[name])
    if optimize:
        for dfa in base.get("dfa", []):
            if type(base["dfa"][dfa]) is not str:
                base["dfa"][dfa] = None
    return base


def norm_eval(env_name, violations, score, factor):
    result = score / SCALING[env_name]["Score"]
    for i, (norm_name, violation_count) in enumerate(reversed(violations)):
        lexicographic_factor = factor ** (i + 1)
        max_violations = SCALING[env_name][norm_name]
        adherence = (max_violations - violation_count) / max_violations
        result += lexicographic_factor * adherence
    return result


SCALING = {
    "test": {},
    "merchant": {
        "Score": 750.0,  # TODO
        "Delivery": 1.0,  # TODO
        "Danger": 2.0,  # TODO
    },
    "pacman": {
        "Score": 1795,  # optimal score is 1795
        "Vegan": 4,  # max 4 violations
        "VegetarianOrange": 2,  # etc.
        "VegetarianBlue": 2,
        "EarlyBird": 1,
        "OblBlue": 5,
        "HighScore": 100,
        "CTD": 4,
        "Penalty(total)": 16,
    },
}


BASES = {
    "test": {
        "default": {},
    },
    "merchant": {
        "default": {},
        "control": {
            "dfa": {},
            "monitors": ["DangerMonitor", "DeliveryMonitor"],
        },
        "norms": {
            "dfa": {
                "DangerDFA": 10.059022786571928,
                "DeliveryDFA": 812.716227344329,  # Delivery > Danger
            },
            "monitors": ["DangerMonitor", "DeliveryMonitor"],
            "order": ["Delivery", "Danger"],
        },
    },
    "pacman": {
        # vanilla
        "default": {"feature_extractor": "complete", "feature_extractor_image": "image-full"},
        # control experiments
        "control": {
            "dfa": {},
            "feature_extractor": "complete",
            "feature_extractor_image": "image-full",
            "monitors": [
                "VegetarianBlueMonitor", # = VegetarianBlue
                "SolutionMonitor", # = VegetarianOrange, Earlybird
                "Penalty1Monitor", # = CTD
            ],
        },
        # simple norm base
        "vegan": { # best highlevel: DQN, 2.5 million steps
            "dfa": {"VegBlueDFA": 1636.632265720805, "VegOrangeDFA": "VegBlueDFA"},  # even priorities
            "feature_extractor": "dfa",
            "feature_extractor_image": "image-full+dfa",
            "monitors": ["VeganMonitor"],
            "order": ["Vegan"],
        },
        # with an exception
        "vegetarian": {
            "dfa": { # best highlevel: DQN, 10 million steps
                "VegBlueDFA": 3710.6783688206015,
                "VegOrangeDFA": "VegBlueDFA",
                "PermBlueDFA": "VegBlueDFA",
            },  # even priorities
            "feature_extractor": "complete-distinguish",
            "feature_extractor_image": "image-full+dfa",
            "monitors": ["VegetarianOrangeMonitor"],
            "order": ["VegetarianOrange"],
        },
        # with priorities
        "vegan-pref": {
            "dfa": {"VegBlueDFA": 1000.0, "VegOrangeDFA": 6000.0},  # VegBlue < VegOrange
            "feature_extractor": "complete-distinguish",
            "feature_extractor_image": "image-full+dfa",
            "monitors": ["VeganPreferenceMonitor"],
            "order": ["VegetarianOrange", "VegetarianBlue"],
        },
        # maintenance obligation
        "highscore": {
            "dfa": {"TrappedDFA": 100.0},
            "feature_extractor": "dfa",
            "feature_extractor_image": "image-full+dfa",
            "monitors": ["HighScoreMonitor"],
            "order": ["HighScore"],
        },
        # achievenemt obligation
        "earlybird": {  # best highlevel: PPO, 10 million steps
            "dfa": {"EarlyBirdDFA1": 4927.569082952659},
            "feature_extractor": "dfa",
            "feature_extractor_image": "image-full+dfa",
            "monitors": ["EarlyBirdMonitor1"],
            "order": ["EarlyBird"],
        },
        # conflicts
        "vegan-conflict": {
            "dfa": {
                "VegBlueDFA": 1000.0,
                "VegOrangeDFA": "VegBlueDFA",
                "OblBlueDFA": 3000.0,  # OblBlue > VegBlue, and VegOrange should be minimized, but shouldn't cause weak pareto optimality
            },
            "feature_extractor": "dfa",
            "feature_extractor_image": "image-full+dfa",
            "monitors": ["VeganConflictMonitor"],
            "order": ["VegetarianOrange", "OblBlue", "VegetarianBlue"],
        },
        "contradiction": {  # best highlevel: PPO, 20 million steps
            "dfa": {
                "EarlyBirdDFA1": 4719.7584507181455,
                "VegBlueDFA": 714.172186332458,
                "VegOrangeDFA": "VegBlueDFA",
            },
            "feature_extractor": "dfa",
            "feature_extractor_image": "image-full+dfa",
            "monitors": ["ContradictionMonitor", "VeganMonitor"],
            "order": ["EarlyBird", "Vegan"],
        },
        "solution": {
            "dfa": {
                "EarlyBirdDFA1": 1483.499729055288,
                "VegBlueDFA": 323.27418723646514,
                "VegOrangeDFA": "VegBlueDFA",
                "PermBlueDFA": "VegBlueDFA",
            },
            "feature_extractor": "complete-distinguish",
            "feature_extractor_image": "image-full+dfa",
            "monitors": ["SolutionMonitor"],
            "order": ["EarlyBird", "VegetarianOrange"],
        },
        "solution-altweights": {
            "dfa": {
                "EarlyBirdDFA1": 2176.2105313142497,
                "VegBlueDFA": 365.2269427724439,
                "VegOrangeDFA": "VegBlueDFA",
                "PermBlueDFA": "VegBlueDFA",
            },
            "feature_extractor": "complete-distinguish",
            "feature_extractor_image": "image-full+dfa",
            "monitors": ["SolutionMonitor"],
            "order": ["EarlyBird", "VegetarianOrange"],
        },
        # contrary to duty
        "penalty1": {
            "dfa": {
                "EarlyBirdDFA1": 3067.2038106118816,
                "VegBlueDFA": 426.1333673762767,
                "VegOrangeDFA": "VegBlueDFA",
                "CTDBlueDFA": 806.9335974146014,
                "CTDOrangeDFA": "CTDBlueDFA",
            },
            "feature_extractor": "dfa",
            "feature_extractor_image": "image-full+dfa",
            "monitors": ["ContradictionMonitor", "VeganMonitor", "Penalty1Monitor"],
            "order": ["EarlyBird", "CTD", "Vegan"],
        },
        "penalty3": {
            "dfa": {
                "EarlyBirdDFA1": 5000.0,
                "VegBlueDFA": 500.0,
                "VegOrangeDFA": "VegBlueDFA",
                "CTDBlueDFA3": 1000.0,
                "CTDOrangeDFA3": "CTDBlueDFA3",
            },
            "feature_extractor": "dfa",
            "feature_extractor_image": "image-full+dfa",
            "monitors": ["ContradictionMonitor", "VeganMonitor", "Penalty3Monitor"],
            "order": ["EarlyBird", "CTD", "Vegan"],
        },
    },
}
