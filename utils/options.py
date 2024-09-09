import argparse
import importlib

import yaml


def args_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--alg', type=str)

    # ===== Basic Setting ======
    parser.add_argument('--suffix', type=str, help="Suffix for file")
    parser.add_argument('--device', type=int, help="Device to use")
    parser.add_argument('--dataset', type=str)
    parser.add_argument('--model', type=str)

    # ===== Federated Setting =====
    parser.add_argument('--total_num', type=int, help="Total clients num")
    parser.add_argument('--sr', type=float, help="Clients sample rate")
    parser.add_argument('--rnd', type=int, help="Communication rounds")
    parser.add_argument('--test_gap', type=int, help='Rounds between two test phases')

    # ===== Local Training Setting =====
    parser.add_argument('--bs', type=int, help="Batch size")
    parser.add_argument('--epoch', type=int, help="Epoch num")
    parser.add_argument('--lr', type=float, help="Learning rate")
    parser.add_argument('--gamma', type=float, help="Exponential decay of learning rate")

    # ===== System Heterogeneity Setting =====
    parser.add_argument('--lag_level', type=int, default=3, help="Lag level used to simulate latency of device")
    parser.add_argument('--lag_rate', type=float, default=0.3, help="Proportion of stale device")

    # ===== Other Setting =====
    # Asynchronous aggregation
    parser.add_argument('--alpha', type=float, default=0.3, help='Weight decay')

    # === read specific parameters from each method
    global_args = parser.parse_known_args()
    spec_alg = global_args.alg
    trainer_module = importlib.import_module(f'trainer.alg.{spec_alg}')
    spec_args = trainer_module.add_args(parser) if hasattr(trainer_module, 'add_args') else global_args

    # === read params from yaml ===
    # NOTE: Only overwrite when the value is None
    with open('config.yaml', 'r') as f:
        yaml_config = yaml.load(f.read(), Loader=yaml.Loader)
    for k, v in vars(spec_args).items():
        if v is None:
            setattr(spec_args, k, yaml_config[k])
    return spec_args
