import importlib
import os
import sys

import numpy as np
import ujson
from tqdm import tqdm

from utils.dataprocess import DataProcessor
from utils.options import args_parser

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
        self.clients = [trainer_module.Client(idx, args) for idx in
                        tqdm(range(args.total_num), desc='Generating Clients', leave=True)]
        self.server = trainer_module.Server(0, args, self.clients)

    def simulate(self):
        start_round = self.recover()
        try:
            for rnd in tqdm(range(start_round, self.server.total_round), desc='Communication Round', leave=False):
                # ===================== train =====================
                self.server.round = rnd
                self.server.run()

                # ===================== test =====================
                ret_dict = self.server.test_all()
                self.acc_processor.append(ret_dict['acc'])

                if rnd % 5 == 0:
                    self.save_context()

                self.output.write(f'========== Round {rnd} ==========\n')

                for k, v in ret_dict.items():
                    if 'loss' in k or 'std' in k:
                        continue
                    std_str = f'{k}_std'
                    self.output.write(f'{k}: {v:.2f}+-{ret_dict[std_str]:.2f}\n')

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
            acc_std = ret_dict['acc_std']
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

    def recover(self):
        if args.recover == 0:
            return 0
        config_path = f'./{args.suffix}/{args.alg}_{args.dataset}_{args.model}_{args.total_num}c_{args.epoch}E_lr{args.lr}.json'
        dir_path = f'./{args.suffix}/checkpoints_{args.alg}_{args.dataset}_{args.model}_{args.total_num}c_{args.epoch}E_lr{args.lr}'

        # check config
        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                config = ujson.load(f)
            if config['suffix'] == args.suffix and \
                    config['alg'] == args.alg and \
                    config['dataset'] == args.dataset and \
                    config['model'] == args.model and \
                    config['total_num'] == args.total_num and \
                    config['epoch'] == args.epoch and \
                    config['lr'] == args.lr:
                print('Matched.')
                print(config)
            else:
                exit('Config not the same, please check the config!')

        # load model
        self.server.load_model(f'{dir_path}/global.pth')
        for idx in range(args.total_num):
            self.clients[idx].load_model(f'{dir_path}/{idx}.pth')

        return config['round']

    def save_context(self):
        config = {
            'suffix': args.suffix,
            'alg': args.alg,
            'dataset': args.dataset,
            'model': args.model,
            'total_num': args.total_num,
            'epoch': args.epoch,
            'lr': args.lr,
            'round': self.server.round
        }

        dir_path = f'./{args.suffix}/checkpoints_{args.alg}_{args.dataset}_{args.model}_{args.total_num}c_{args.epoch}E_lr{args.lr}'

        # save model
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)
        with open(f'{dir_path}/config.json', 'w') as f:
            ujson.dump(config, f)
        self.server.save_model(f'{dir_path}/global.pth')
        for idx in range(args.total_num):
            self.clients[idx].save_model(f'{dir_path}/{idx}.pth')


if __name__ == '__main__':
    args = args_parser()
    fed = FedSim(args=args)
    fed.simulate()
