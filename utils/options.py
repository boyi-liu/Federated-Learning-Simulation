import argparse
import importlib
import sys


def args_parser():
    parser = argparse.ArgumentParser()

    # ===== Method Setting ======
    parser.add_argument('alg', type=str, default='fedavg')
    parser.add_argument('--dataset', type=str, default='mnist')
    parser.add_argument('--model', type=str, default='mlp')

    # ===== Training Setting =====
    parser.add_argument('--total_num', type=int, default=5, help="Total clients num")
    parser.add_argument('--sr', type=float, default=0.3, help="Clients sample rate")
    parser.add_argument('--suffix', type=str, default='default', help="Suffix for file")
    parser.add_argument('--device', type=int, default=0, help="Device to use")

    parser.add_argument('--rnd', type=int, default=10, help="Communication rounds")
    parser.add_argument('--bs', type=int, default=10, help="Batch size")
    parser.add_argument('--epoch', type=int, default=3, help="Epoch num")
    parser.add_argument('--lr', type=float, default=0.1, help="Learning rate")
    parser.add_argument('--gamma', type=float, default=0.99, help="Exponential decay of learning rate")

    parser.add_argument('--test_gap', type=int, default=1, help='Rounds between two test phases')

    # ===== System Heterogeneity Setting =====
    parser.add_argument('--lag_level', type=int, default=3, help="Lag level used to simulate latency of device")
    parser.add_argument('--lag_rate', type=float, default=0.3, help="Proportion of stale device")

    # ===== Other Setting =====
    # Asynchronous aggregation
    parser.add_argument('--alpha', type=float, default=0.3, help='Weight decay')

    # ===== Method Specific Setting =====
    spec_alg = sys.argv[1]
    trainer_module = importlib.import_module(f'trainer.alg.{spec_alg}')
    return trainer_module.add_args(parser)
