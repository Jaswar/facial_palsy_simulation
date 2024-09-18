import torch as th
import numpy as np
from models import INRModel, SimulatorModel, SurfaceINRModel
from datasets import INRDataset, SimulatorDataset, SurfaceINRDataset
from actuation_predictor import ActuationPredictor
import time
import copy
import json
import argparse
from common import get_optimizer


def sample_configuration(mode):
    if mode == 'simulator':
        config = {
            'num_hidden_layers': 5, #np.random.randint(3, 12),
            'hidden_size': 256, #2 ** np.random.randint(3, 10),
            'learning_rate': 10 ** np.random.uniform(-6, -2),
            'min_lr': 10 ** np.random.uniform(-8, -4),
            'batch_size': 2 ** np.random.randint(10, 14),
            'fourier_features': 8, #np.random.randint(5, 20),
            'optimizer': np.random.choice(['adam', 'rmsprop', 'sgd']),
            'w_fixed': 2.0, # 10 ** np.random.uniform(-1., 1.),
            'w_energy': 0.5 # 10 ** np.random.uniform(-1., 1.),
        }
    elif mode == 'inr':
        config = {
            'num_hidden_layers': 5, #np.random.randint(3, 12),
            'hidden_size': 256,#2 ** np.random.randint(3, 10),
            'learning_rate': 10 ** np.random.uniform(-6, -2),
            'min_lr': 10 ** np.random.uniform(-8, -4),
            'batch_size': 2 ** np.random.randint(10, 14),
            'fourier_features': 8, #np.random.randint(5, 20),
            'optimizer': np.random.choice(['adam', 'rmsprop', 'sgd']),
            'w_surface': 15.0, # 10 ** np.random.uniform(-1., 2.),
            'w_deformation': 0.05, # 10 ** np.random.uniform(-3., -1.),
            'w_jaw': 1.0, # 10 ** np.random.uniform(-1., 1.),
            'w_skull': 2.0, # 10 ** np.random.uniform(-1., 1.),
        }
    elif mode == 'surface_inr':
        config = {
            'num_hidden_layers': np.random.randint(3, 12),
            'hidden_size': 2 ** np.random.randint(3, 10),
            'learning_rate': 10 ** np.random.uniform(-6, -2),
            'min_lr': 10 ** np.random.uniform(-8, -4),
            'batch_size': 2 ** np.random.randint(10, 14),
            'fourier_features': np.random.randint(5, 20),
            'optimizer': np.random.choice(['adam']),
            'w_flame': 10 ** np.random.uniform(-2., 2.),
            'w_boundary': 10 ** np.random.uniform(-2., 2.),
            'w_deformation': 10 ** np.random.uniform(-4., -1.),
        }
    return config


def run_configuration(config, dataset, budget, mode, pretrained_model_path):
    if mode == 'simulator':
        model = SimulatorModel(num_hidden_layers=config['num_hidden_layers'],
                               hidden_size=config['hidden_size'],
                               fourier_features=config['fourier_features'],
                               w_fixed=config['w_fixed'],
                               w_energy=config['w_energy'])
    elif mode == 'inr':
        model = INRModel(num_hidden_layers=config['num_hidden_layers'], 
                         hidden_size=config['hidden_size'],
                         fourier_features=config['fourier_features'],
                         w_surface=config['w_surface'],
                         w_deformation=config['w_deformation'],
                         w_jaw=config['w_jaw'],
                         w_skull=config['w_skull'])
    elif mode == 'surface_inr':
        model = SurfaceINRModel(num_hidden_layers=config['num_hidden_layers'],
                                hidden_size=config['hidden_size'],
                                fourier_features=config['fourier_features'],
                                w_flame=config['w_flame'],
                                w_boundary=config['w_boundary'],
                                w_deformation=config['w_deformation'])
    model = th.compile(model)
    model.to(dataset.device)
    optimizer = get_optimizer(config, model)
    config['min_lr'] = min(config['min_lr'], config['learning_rate'] / 2)
    lr_scheduler = th.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=budget, eta_min=config['min_lr'])
    start_time = time.time()
    last_lr_step = start_time

    if pretrained_model_path is not None:
        model.load_state_dict(th.load(pretrained_model_path))

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


def random_search(dataset, model_path, config_path, predicted_jaw_path, budget, mode, num_runs, pretrained_model_path):
    best_loss = float('inf')
    run = 0
    while run < num_runs or num_runs == -1:
        config = sample_configuration(mode)
        print(f'Running configuration: {config}')
        model, loss = run_configuration(config, dataset, budget, mode, pretrained_model_path)
        if loss < best_loss:
            print(f'New best loss: {loss}')
            best_loss = loss
            th.save(model.state_dict(), model_path)
            with open(config_path, 'w') as f:
                json.dump(config, f, indent=4)
        if predicted_jaw_path is not None and mode == 'inr':
            jaw_nodes = dataset.nodes[dataset.jaw_mask]
            predicted_jaw = model.predict(jaw_nodes).cpu().numpy()
            predicted_jaw = predicted_jaw * (dataset.maxv - dataset.minv) + dataset.minv
            np.save(predicted_jaw_path, predicted_jaw)
        run += 1


def main(args):
    if th.cuda.is_available():
        device = 'cuda'
    else:
        device = 'cpu'
    print(f'Using device: {device}')

    pretrained_model_path = args.pretrained_model_path if args.use_pretrained else None
    if args.mode == 'inr':
        dataset = INRDataset(args.tetmesh_path, args.jaw_path, args.skull_path, args.neutral_path, args.deformed_path, device=device)
    elif args.mode == 'simulator':
        actuation_predictor = ActuationPredictor(args.main_actuation_model_path, args.inr_config_path, args.tetmesh_path, args.contour_path, args.reflected_contour_path, 
                                                 secondary_model_path=args.secondary_actuation_model_path, device=device)
        dataset = SimulatorDataset(args.tetmesh_path, args.jaw_path, args.skull_path, args.predicted_jaw_path, actuation_predictor, device=device)
    elif args.mode == 'surface_inr':
        dataset = SurfaceINRDataset(args.neutral_path, args.neutral_flame_path, args.deformed_flame_path, device=device)
    random_search(dataset, args.model_path, args.config_path, args.predicted_jaw_path, args.budget, args.mode, args.num_runs, pretrained_model_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--tetmesh_path', type=str, default=None)
    parser.add_argument('--jaw_path', type=str, default=None)
    parser.add_argument('--skull_path', type=str, default=None)
    parser.add_argument('--model_path', type=str, default=None)
    parser.add_argument('--config_path', type=str, default=None)
    parser.add_argument('--budget', type=int, default=60 * 10)  # 10 minutes
    parser.add_argument('--num_runs', type=int, default=-1)
    parser.add_argument('--predicted_jaw_path', type=str, default=None)

    parser.add_argument('--use_pretrained', action='store_true')
    parser.add_argument('--pretrained_model_path', type=str, default=None)

    parser.add_argument('--neutral_path', type=str, default=None)
    parser.add_argument('--deformed_path', type=str, default=None)

    parser.add_argument('--mode', type=str, required=True, choices=['inr', 'simulator', 'surface_inr'])

    parser.add_argument('--main_actuation_model_path', type=str, default=None)
    parser.add_argument('--secondary_actuation_model_path', type=str, default=None)
    parser.add_argument('--contour_path', type=str, default=None)
    parser.add_argument('--reflected_contour_path', type=str, default=None)
    parser.add_argument('--inr_config_path', type=str, default='configs/config_inr.json')

    parser.add_argument('--neutral_flame_path', type=str, default=None)
    parser.add_argument('--deformed_flame_path', type=str, default=None)

    args = parser.parse_args()
    main(args)
