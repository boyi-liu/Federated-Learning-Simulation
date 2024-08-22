from trainer.base import BaseServer, BaseClient

class Client(BaseClient):
    def run(self):
        self.train()

    def clone_model(self, target):
        # NOTE: no downlink here
        pass


class Server(BaseServer):
    def run(self):
        # NOTE: no uplink, no downlink, no aggregation
        self.sample()
        self.client_update()