from multiprocessing import Pool
from time import sleep
import argparse
import json
import os


def pooled(n_trials, n_proc, f, *args, **kwargs):
    if n_trials == 1:
        f(*args, **kwargs)
        return
    with Pool(n_proc) as p:
        tasks = []
        for _ in range(n_trials):
            sleep(10)
            tasks.append(p.apply_async(f, args, kwargs))
        for task in tasks:
            task.wait()


def valid_json_file(filepath):
    if not os.path.isfile(filepath):
        raise argparse.ArgumentTypeError(f"{filepath} is not a valid file.")
    try:
        with open(filepath, "r") as f:
            return json.load(f)  # Try to load the file as JSON
    except json.JSONDecodeError:
        raise argparse.ArgumentTypeError(f"{filepath} is not a valid JSON file.")
