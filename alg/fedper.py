from alg.base import BaseClient, BaseServer
from utils.time_utils import time_record


class Client(BaseClient):
    def __init__(self, id, args):
        super().__init__(id, args)
        self.local_keys = ['ee_fc']
        self.p_params = [any(key in name for key in self.local_keys)
                         for name, _ in self.model.named_parameters()]

    @time_record
    def run(self):
        self.train()


# extend Client to get the self.p_params
class Server(BaseServer, Client):
    def run(self):
        self.sample()
        self.downlink()
        self.client_update()
        self.uplink()
        self.aggregate()