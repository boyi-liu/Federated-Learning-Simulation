import importlib
import sys
import numpy as np
import os
import wandb

from alg.asyncbase import AsyncBaseServer
from utils.options import args_parser
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

class FedSim:
    def __init__(self, args):
        self.args = args

        if not os.path.exists(f'./{args.suffix}'):
            os.makedirs(f'./{args.suffix}')

        output_path = f'{args.suffix}/{args.alg}_{args.dataset}_{args.model}_' \
                      f'{args.total_num}c_{args.epoch}E_lr{args.lr}'
        self.output = open(f'./{output_path}.txt', 'a')
        wandb.init(project=args.wandb_project, name=output_path)

        # === load trainer ===
        trainer_module = importlib.import_module(f'alg.{args.alg}')

        args.output = self.output

        # === init clients & server ===
        self.clients = [trainer_module.Client(idx, args) for idx in tqdm(range(args.total_num))]
        self.server = trainer_module.Server(0, args, self.clients)

    def simulate(self):
        acc_list = []
        TEST_GAP = self.args.test_gap

        # check if it is an async methods
        if isinstance(self.server, AsyncBaseServer):
            TEST_GAP *= int(args.total_num * args.sr)
            self.server.total_round *= int(args.total_num * args.sr)
        try:
            for rnd in tqdm(range(0, self.server.total_round), desc='Communication Round', leave=False):
                # ===================== train =====================
                self.server.round = rnd
                self.server.run()

                # ===================== test =====================
                if rnd % TEST_GAP != (TEST_GAP-1):
                    continue
                ret_dict = self.server.test_all()
                acc_list.append(ret_dict['acc'])

                self.output.write(f'========== Round {rnd} ==========\n')
                self.output.write('acc: %.2f seconds\n' % ret_dict['acc'])
                self.output.write('wall clock time: %.2f seconds\n' % self.server.wall_clock_time)
                self.output.flush()

                wandb.log({'acc': ret_dict['acc']})

        except KeyboardInterrupt:
            ...
        finally:
            avg_count = 5
            acc_avg = np.mean(acc_list[-avg_count:]).item()
            acc_max = np.max(acc_list).item()

            self.output.write('==========Summary==========\n')

            self.output.write('server, max accuracy: %.2f\n' % acc_max)
            self.output.write('server, final accuracy: %.2f\n' % acc_avg)


if __name__ == '__main__':
    args = args_parser()
    fed = FedSim(args=args)
    fed.simulate()