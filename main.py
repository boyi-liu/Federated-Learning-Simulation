import importlib
import sys
import numpy as np
import os

from utils.options import args_parser
from utils.dataprocess import DataProcessor
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class FedSim:
    def __init__(self, args):
        self.args = args

        # === load trainer ===
        trainer_module = importlib.import_module(f'trainer.alg.{args.alg}')

        # === init other config ===
        self.acc_processor = DataProcessor()
        self.acc_shift_processor = DataProcessor()

        if not os.path.exists(f'./{args.suffix}'):
            os.makedirs(f'./{args.suffix}')

        self.config_output()
        args.output = self.output

        # === init clients & server ===
        self.clients = [trainer_module.Client(idx, args) for idx in range(args.total_num)]
        self.server = trainer_module.Server(0, args, self.clients)

    def simulate(self):
        TEST_GAP = self.args.test_gap
        try:
            for rnd in tqdm(range(0, self.server.total_round, TEST_GAP), desc='Communication Round', leave=False):
                # ===================== train =====================
                self.server.round = rnd
                self.server.run()

                # ===================== test =====================
                ret_dict = self.server.test_all()
                self.acc_processor.append(ret_dict['acc'])

                self.output.write(f'========== Round {rnd} ==========\n')
                self.output.write('server, accuracy: %.2f, ' % ret_dict['acc'])
                self.output.write('wall clock time: %.2f seconds\n' % self.server.wall_clock_time)
                self.output.flush()

        except KeyboardInterrupt:
            ...
        finally:
            acc_list = self.acc_processor.data
            # np.save(f'./{self.args.suffix}/{self.args.alg}_{self.args.dataset}'
            #         f'_{self.args.model}_{self.args.total_num}c_{self.args.epoch}E_lr{args.lr}.npy',
            #         np.array(acc_list))
            avg_count = 2
            acc_avg = np.mean(acc_list[-avg_count:]).item()
            acc_std = ret_dict['std']
            acc_max = np.max(acc_list).item()

            self.output.write('==========Summary==========\n')

            self.output.write('server, max accuracy: %.2f\n' % acc_max)
            self.output.write('server, final accuracy: %.2f +- %.2f\n' % (acc_avg, acc_std))
            self.res_output.write(f'{self.args.alg}, acc: {acc_avg:.2f}+-{acc_std:.2f}\n')

    def config_output(self):
        output_path = f'./{args.suffix}/{args.alg}_{args.dataset}_{args.model}_' \
                      f'{args.total_num}c_{args.epoch}E_lr{args.lr}.txt'
        self.output = open(output_path, 'a')
        result_path = f'./{args.suffix}/result_{args.dataset}_{args.model}_' \
                      f'{args.total_num}c_{args.epoch}E_lr{args.lr}.txt'
        self.res_output = open(result_path, 'a')


if __name__ == '__main__':
    args = args_parser()
    fed = FedSim(args=args)
    fed.simulate()
