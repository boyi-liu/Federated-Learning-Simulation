# A Simple FL Simulation Framework

A *simple and easy-to-extend* 
federated learning (FL) simulation framework.

We support both synchronous and asynchronous federated learning simulation.

The asynchronous simulation is based on a prior queue.
+ Insert training-finished clients into the prior queue based on supposed finished time
+ Pop the first item in the prior queue
+ Update the simulated wall-clock time based on the item



## Getting Started

+ Download this project 
```
git clone https://github.com/boyi-liu/Asynchronous-Federated-Learning-Simulation.git
```
+ Install required packages
+ Partition datasets
```
cd dataset
python generate_cifar10.py noniid balance dir
```
Some important hyperparameters in partitioning datasets:
`alpha` in `dataset_utils.py` is a measurement for the non-iidness of the dataset.
 `num_clients` in `generate_mnist.py` is the number of clients, i.e. how many shards you slice the datasets into.
+ Config hyperparameters

There are two places to config, one in `/script/config.yaml`, another in `utils/options.py`.
The priority follows: `args>yaml`.
If you config hyperparameters in `utils/options.py`, it will overwrite that in `args`.

+ Run evaluation
```
cd script
bash run.sh {your_suffix}
```

## How to extend
### Create a new file
Create a new `{your_algorithm}.py` file inside `trainer.alg`
### Extend the Client and Server
If you are working on a **synchronous** FL algorithm, just extend the Client and Server class in `trainer.base`
```
from trainer.base import BaseServer, BaseClient

class Client(BaseClient):
    def __init__(self, id, args, dataset):
        super().__init__(id, args, dataset)

class Server(BaseServer):
    def __init__(self, id, args, dataset, clients):
        super().__init__(id, args, dataset, clients)
```
Otherwise, you may extend the Client and Server in `trainer.asyncbase`
```
from trainer.asyncbase import AsyncBaseServer, AsyncBaseClient

class Client(AsyncBaseClient):
    def __init__(self, id, args, dataset):
        super().__init__(id, args, dataset)

class Server(AsyncBaseServer):
    def __init__(self, id, args, dataset, clients):
        super().__init__(id, args, dataset, clients)
```

### Config your hyperparameters
For algorithm-specific hyperparameters, 
it is recommended to add a `add_args()` function inside your file
```
def add_args(parser):
    parser.add_argument('--{your_param}', type=int, default=1)
    return parser.parse_args()
```

And all general args could be found in `utils/options.py`

### Implement your algorithms
We claim that each algorithm should overwrite the function `run()`,
because it stands for the main workflow of your algorithm.

You can overwrite or add any function as you want then.

## Acknowledgements
The data partitioning module is adopted from [PFLlib](https://github.com/TsingZ0/PFLlib).

