import heapq
import random
import time

from trainer.base import BaseClient, BaseServer


class AsyncBaseClient(BaseClient):
    def __init__(self, id, args):
        super().__init__(id, args)
        self.train_state = {}

    def run(self):
        raise NotImplementedError()

    def local_test(self):
        # NOTE: we need to cache the model state, to avoid the current model is overwritten by global model
        self.train_state = self.model.state_dict()
        super().local_test()
        self.model.load_state_dict(self.train_state)


class AsyncBaseServer(BaseServer):
    def __init__(self, id, args, clients):
        super().__init__(id, args, clients)
        self.priority_queue = []
        self.alpha = args.alpha
        self.MAX_CONCURRENCY = int(self.client_num * self.sample_rate)
        self.staleness = [0 for _ in self.clients]

        # NOTE: sampled_clients means those sampled in current round
        # NOTE: active_clients means those are currently running
        self.active_clients = []
        self.aggr_id = -1

    def run(self):
        raise NotImplementedError()

    def reset_staleness(self):
        for c in self.sampled_clients:
            self.staleness[c.id] = 0

    def sample(self):
        if len(self.active_clients) < self.MAX_CONCURRENCY:
            clients_filtered = [client for client in self.clients if client not in self.active_clients]
            sample_scale = self.MAX_CONCURRENCY - len(self.active_clients)

            self.sampled_clients = random.sample(clients_filtered, sample_scale)
            self.active_clients.extend(self.sampled_clients)

    def downlink(self):
        super().downlink()

    def client_update(self):
        for client in self.sampled_clients:
            client.model.train()
            client.reset_optimizer(False)
            start_time = time.time()
            client.run()
            end_time = time.time()
            client.training_time = (end_time - start_time) * client.lag_level
            print(client.training_time)
            print(self.wall_clock_time + client.training_time)
            heapq.heappush(self.priority_queue, (self.wall_clock_time + client.training_time, client))

    def uplink(self):
        self.wall_clock_time, client = heapq.heappop(self.priority_queue)
        self.aggr_id = client.id
        self.active_clients.remove(client)

    def aggregate(self):
        client = self.clients[self.aggr_id]

        t_global = self.model.parameters_to_tensor(self.local_params)
        t_local = client.model.parameters_to_tensor(self.local_params)
        t_aggr = self.weight_decay() * t_local + (1 - self.weight_decay()) * t_global
        self.model.tensor_to_parameters(t_aggr, self.local_params)

    def update_staleness(self):
        for c in self.active_clients:
            self.staleness[c.id] += 1

    def weight_decay(self):
        return self.alpha
