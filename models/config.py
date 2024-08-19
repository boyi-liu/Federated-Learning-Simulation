import importlib

dataset_params = {
    'mnist': 10,
    'cifar10': 10
}

model_params = {
    'mnist': {
        'mlp': {'dim_in': 784,
                'hidden_layer': 256},
        'cnnmnist': {}
    },
    'cifar10': {
        'cnncifar': {}
    },
}

def load_model(args):
    model_arg = args.model
    dataset_arg = args.dataset
    args.class_num = dataset_params[dataset_arg]

    if dataset_arg not in model_params.keys():
        exit('Dataset params not exist (in config.py)!')
    if model_arg not in model_params[dataset_arg].keys():
        exit('Model params not exist (in config.py)!')
    model_module = importlib.import_module(f'models.{model_arg}')
    return getattr(model_module, model_arg)(args, {**model_params[dataset_arg][model_arg]})