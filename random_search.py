import torch as th
import numpy as np
from models import INRModel, SimulatorModel
from dataset import TetmeshDataset
import time
import copy
import json
import argparse
from common import get_optimizer


def sample_configuration(simulator):
    if simulator:
        config = {
            'num_hidden_layers': 9, #np.random.randint(3, 12),
            'hidden_size': 64, # 2 ** np.random.randint(3, 10),
            'learning_rate': 10 ** np.random.uniform(-6, -2),
            'min_lr': 10 ** np.random.uniform(-8, -4),
            'batch_size': 2 ** np.random.randint(10, 14),
            'fourier_features': 8, # np.random.randint(5, 20),
            'optimizer': np.random.choice(['adam', 'rmsprop', 'sgd']),
            'w_jaw': 1.0, # 10 ** np.random.uniform(-1., 1.),
            'w_skull': 2.0, # 10 ** np.random.uniform(-1., 1.),
            'w_energy': 0.5,
        }
    else:
        config = {
            'num_hidden_layers': 9, #np.random.randint(3, 12),
            'hidden_size': 64, # 2 ** np.random.randint(3, 10),
            'learning_rate': 10 ** np.random.uniform(-6, -2),
            'min_lr': 10 ** np.random.uniform(-8, -4),
            'batch_size': 2 ** np.random.randint(10, 14),
            'fourier_features': 8, # np.random.randint(5, 20),
            'optimizer': np.random.choice(['adam', 'rmsprop', 'sgd']),
            'w_surface': 10.0, # 10 ** np.random.uniform(-1., 1.),
            'w_deformation': 0.02, #10 ** np.random.uniform(-3., -1.),
            'w_jaw': 1.0, # 10 ** np.random.uniform(-1., 1.),
            'w_skull': 2.0, # 10 ** np.random.uniform(-1., 1.),
        }
    return config


def run_configuration(config, dataset, budget, simulator):
    if simulator:
        model = SimulatorModel(num_hidden_layers=config['num_hidden_layers'],
                               hidden_size=config['hidden_size'],
                               fourier_features=config['fourier_features'],
                               w_jaw=config['w_jaw'],
                               w_skull=config['w_skull'],
                               w_energy=config['w_energy'])
    else:
        model = INRModel(num_hidden_layers=config['num_hidden_layers'], 
                         hidden_size=config['hidden_size'],
                         fourier_features=config['fourier_features'],
                         w_surface=config['w_surface'],
                         w_deformation=config['w_deformation'],
                         w_jaw=config['w_jaw'],
                         w_skull=config['w_skull'])
    model = th.compile(model)
    model.to(dataset.device)
    optimizer = get_optimizer(config, model)
    config['min_lr'] = min(config['min_lr'], config['learning_rate'] / 2)
    lr_scheduler = th.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=budget, eta_min=config['min_lr'])
    start_time = time.time()
    last_lr_step = start_time

    model.load_state_dict(th.load('checkpoints/prior.pth'))

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


def random_search(dataset, model_path, config_path, budget, simulator, num_runs):
    best_loss = float('inf')
    run = 0
    while run < num_runs or num_runs == -1:
        config = sample_configuration(simulator)
        print(f'Running configuration: {config}')
        model, loss = run_configuration(config, dataset, budget, simulator)
        if loss < best_loss:
            print(f'New best loss: {loss}')
            best_loss = loss
            th.save(model.state_dict(), model_path)
            with open(config_path, 'w') as f:
                json.dump(config, f, indent=4)
        run += 1


def main(args):
    assert not args.simulator or args.actuations_path is not None, 'Actuations path must be provided for the simulator'

    if th.cuda.is_available():
        device = 'cuda'
    else:
        device = 'cpu'
    print(f'Using device: {device}')

    dataset = TetmeshDataset(args.tetmesh_path, args.jaw_path, args.skull_path, args.neutral_path, args.deformed_path, actuations_path=args.actuations_path, device=device)
    random_search(dataset, args.model_path, args.config_path, args.budget, args.simulator, args.num_runs)


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
    parser.add_argument('--num_runs', type=int, default=-1)

    parser.add_argument('--simulator', action='store_true')
    parser.add_argument('--actuations_path', type=str, default=None)

    args = parser.parse_args()
    main(args)
