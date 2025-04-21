from alg.base import BaseClient, BaseServer
from utils.time_utils import time_record


class Client(BaseClient):
    @time_record
    def run(self):
        self.train()

    def clone_model(self, target):
        # NOTE: no downlink here
        pass

# extend Client to get the self.p_params
class Server(BaseServer):
    def run(self):
        self.sample()
        self.client_update()