from trainer.asyncbase import AsyncBaseClient, AsyncBaseServer

def add_args(parser):
    parser.add_argument('--a', type=int, default=1)
    parser.add_argument('--b', type=int, default=4)
    parser.add_argument('--strategy', type=str, default='hinge', help='constant/poly/hinge')
    return parser.parse_args()

class Client(AsyncBaseClient):
    def run(self):
        self.train()


class Server(AsyncBaseServer):
    def run(self):
        self.sample()
        self.reset_staleness()
        self.downlink()
        self.client_update()
        self.uplink()
        self.aggregate()
        self.update_staleness()

    def weight_decay(self):
        a = self.args.a
        b = self.args.b
        strategy = self.args.strategy
        tau = self.staleness[self.aggr_id]
        if strategy == 'constant':
            return 1
        elif strategy == 'poly':
            return 1 / ((tau+1) ** abs(a))
        elif strategy == 'hinge':
            if tau <= b:
                return 1
            else:
                return 1 / (a * (tau+b) + 1)
        else:
            exit(-1)
