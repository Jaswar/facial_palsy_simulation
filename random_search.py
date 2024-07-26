import torch as th
import numpy as np
from models import INRModel
from dataset import TetmeshDataset
import time
import copy
import json
import argparse


def sample_configuration():
    config = {
        'num_hidden_layers': np.random.randint(3, 12),
        'hidden_size': 2 ** np.random.randint(3, 10),
        'learning_rate': 10 ** np.random.uniform(-6, -2),
        'batch_size': 2 ** np.random.randint(10, 14),
        'fourier_features': np.random.randint(5, 20),
        'optimizer': np.random.choice(['adam', 'rmsprop']),
        'w_surface': 10 ** np.random.uniform(-1., 1.),
        'w_deformation': 10 ** np.random.uniform(-3., -1.),
        'w_jaw': 10 ** np.random.uniform(-1., 1.),
        'w_skull': 10 ** np.random.uniform(-1., 1.),
        # 'use_sigmoid_output': np.random.choice(['true', 'false']),  # booleans are not json serializable
    }
    return config


def run_configuration(config, dataset, budget):
    model = INRModel(num_hidden_layers=config['num_hidden_layers'], 
                  hidden_size=config['hidden_size'],
                  fourier_features=config['fourier_features'],
                  w_surface=config['w_surface'],
                  w_deformation=config['w_deformation'],
                  w_jaw=config['w_jaw'],
                  w_skull=config['w_skull'])
    model = th.compile(model)
    model.to(dataset.device)
    # get the optimizer
    # considered putting objects in sample_configuration, but they do not serialize
    optimizers = {
        'adam': th.optim.Adam,
        'sgd': th.optim.SGD,
        'rmsprop': th.optim.RMSprop,
    }
    optimizer = optimizers[config['optimizer']](model.parameters(), lr=config['learning_rate'])
    lr_scheduler = th.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=budget, eta_min=1e-8)
    start_time = time.time()
    last_lr_step = start_time

    best_loss = float('inf')
    best_model = None
    epoch = 0
    while time.time() - start_time < budget:
        epoch += 1
        train_loss = model.train_epoch(optimizer, dataset, config['batch_size'])
        if train_loss < best_loss:
            best_loss = train_loss
            best_model = copy.deepcopy(model)

        if time.time() - last_lr_step > 1:
            lr_scheduler.step()
            last_lr_step = time.time()

        print(f'\rTime {(time.time() - start_time):.2f} - Epoch {epoch} - Loss: {train_loss:.8f} - LR: {lr_scheduler.get_last_lr()[0]:.8f}', end='')
    print()
    return best_model, best_loss


def random_search(dataset, model_path, config_path, budget):
    best_loss = float('inf')
    while True:
        config = sample_configuration()
        print(f'Running configuration: {config}')
        model, loss = run_configuration(config, dataset, budget)
        if loss < best_loss:
            print(f'New best loss: {loss}')
            best_loss = loss
            th.save(model.state_dict(), model_path)
            with open(config_path, 'w') as f:
                json.dump(config, f, indent=4)


def main(args):
    if th.cuda.is_available():
        device = 'cuda'
    else:
        device = 'cpu'
    print(f'Using device: {device}')

    dataset = TetmeshDataset(args.tetmesh_path, args.jaw_path, args.skull_path, args.neutral_path, args.deformed_path, device=device)
    random_search(dataset, args.model_path, args.config_path, args.budget)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--tetmesh_path', type=str, required=True)
    parser.add_argument('--jaw_path', type=str, required=True)
    parser.add_argument('--skull_path', type=str, required=True)
    parser.add_argument('--neutral_path', type=str, required=True)
    parser.add_argument('--deformed_path', type=str, required=True)
    parser.add_argument('--model_path', type=str, required=True)
    parser.add_argument('--config_path', type=str, required=True)
    parser.add_argument('--budget', type=int, default=60 * 10)  # 10 minutes
    args = parser.parse_args()
    main(args)
