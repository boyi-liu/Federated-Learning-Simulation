import importlib

dataset2class = {'mnist': 10, 'cifar10': 10}

def name_filter(dataset_arg):
    if 'cifar10' in dataset_arg:
        return 'cifar10'
    elif 'mnist' in dataset_arg:
        return 'mnist'
    return 'Unknown dataset'

def load_model(args):
    dataset_arg = name_filter(args.dataset)
    args.class_num = dataset2class[dataset_arg]
    if args.class_num == -1:
        exit('Dataset params not exist (in config.py)!')

    model_arg = args.model
    model_module = importlib.import_module(f'models.{model_arg}')
    return getattr(model_module, f'{model_arg}_{dataset_arg}')(args)
