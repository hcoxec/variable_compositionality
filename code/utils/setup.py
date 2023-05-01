import argparse
import torch
from typing import Any, Iterable, List, Optional
import torch
import sys
import re
import json
import _jsonnet
import attr
import random
import numpy as np
import datetime
import json
import os
import logging


from collections import defaultdict
from torch.serialization import default_restore_location
from .streamer import Streamer

common_opts = None


class Config(object):
    def __init__(self, config_dict):
        config_dict.update(config_dict['args'])
        self.config = config_dict
        self.__dict__ = config_dict

    def __repr__(self):
        return f"Experiment Config: {self.__dict__}"

@attr.s
class InferConfig:
    global_config = attr.ib()
    config = attr.ib()
    config_args = attr.ib()
    run_id = attr.ib()
    logdir = attr.ib()
    section = attr.ib()
    beam_size = attr.ib()
    output = attr.ib()
    step = attr.ib()
    debug = attr.ib(default=False)
    method = attr.ib(default="beam_search")
    mode = attr.ib(default="infer")
    limit = attr.ib(default=None)
    output_history = attr.ib(default=False)
    streamer= attr.ib(default=None)
    analyses = attr.ib(default=None)


@attr.s
class EvalConfig:
    global_config = attr.ib()
    logdir = attr.ib()
    section = attr.ib()
    inferred = attr.ib()
    output = attr.ib()
    etype = attr.ib(default="match")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "mode",
        choices=["mcd_split","preprocess", "train", "eval", "reval"],
        help="train or eval",
    )
    parser.add_argument("exp_config_file", help="jsonnet file for experiments")
    parser.add_argument("--config_args", help="exp configs")
    args = parser.parse_args()
    return args

def load_arg_file(args):
    # 1. exp config
    if args.config_args:
        exp_config = json.loads(
            _jsonnet.evaluate_file(
                args.exp_config_file, tla_codes={"command_line_args": args.config_args}
            )
        )
    else:
        # empty args make it compatible with non-function exp file
        exp_config = json.loads(_jsonnet.evaluate_file(args.exp_config_file))

    #If using emerge v 2.0+ then run and model config are separate

    if 'model_config' in exp_config.keys():
        model_config = json.loads(
            _jsonnet.evaluate_file(
                exp_config['model_config'], tla_codes={"args": json.dumps(
                    exp_config['model_config_args']
                )}
            )
        )

        exp_config.update(model_config)

    exp_config['mode'] = args.mode
    exp_config['device'] = "cuda" if torch.cuda.is_available() else "cpu"

    
    global common_opts

    exp_config = Config(exp_config)
    run_id_strings = [str(i) for i in exp_config.run_id]
    exp_config.run_id = '_'.join(run_id_strings)
    common_opts = exp_config

    return exp_config

def setup_logger(config):
    # logdir
    

    # Initialize the logger
    logfile_path = os.path.join(config.logdir, "log.txt")
    os.makedirs(os.path.dirname(logfile_path), exist_ok=True)
    logger = logging.getLogger("emerge")
    logger.setLevel(logging.INFO)
    fh = logging.FileHandler(logfile_path)
    formatter = logging.Formatter(
        "%(asctime)s - %(levelname)s - %(module)s - %(message)s"
    )
    fh.setFormatter(formatter)
    logger.addHandler(fh)
    logger.info("Logging to {}".format(config.logdir))
    
    # Save the config info
    with open(
        os.path.join(
            config.logdir,
            "config-{}.json".format(
                datetime.datetime.now().strftime("%Y%m%dT%H%M%S%Z")
            ),
        ),
        "w",
    ) as f:
        json.dump(config.__dict__, f, sort_keys=True, indent=4)


    
    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)

    return logger

def setup_streamer(config):
    return Streamer(config)

def setup_output(outdir):
    dir_name = os.path.dirname(outdir)
    if not os.path.exists(dir_name):
        os.mkdir(dir_name)
    if os.path.exists(outdir):
        print("WARNING Output file {} already exists".format(outdir))
        # sys.exit(1)

def set_seeds(config):
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)
    random.seed(config.seed)

    torch.use_deterministic_algorithms(True)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.seed)

    print(f"Base Seeds set to: {config.seed}")


def fetch_args() -> argparse.Namespace:
    """
    :return: command line options
    """
    global common_opts
    return common_opts

def _set_seed(seed) -> None:
    """
    Seeds the RNG in python.random, torch {cpu/cuda}, numpy.
    :param seed: Random seed to be used


    >>> _set_seed(10)
    >>> random.randint(0, 100), torch.randint(0, 100, (1,)).item(), np.random.randint(0, 100)
    (73, 37, 9)
    >>> _set_seed(10)
    >>> random.randint(0, 100), torch.randint(0, 100, (1,)).item(), np.random.randint(0, 100)
    (73, 37, 9)
    """
    print(f'seed set to:{seed}')

    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

