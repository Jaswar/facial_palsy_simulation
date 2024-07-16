import torch as th
import numpy as np
from model import Model
from dataset import TetmeshDataset
import time
import copy
import json


def sample_configuration():
    config = {
        'num_hidden_layers': np.random.randint(3, 12),
        'hidden_size': 2 ** np.random.randint(5, 13),
        'learning_rate': 10 ** np.random.uniform(-6, -2),
        'batch_size': 2 ** np.random.randint(5, 14),
        'fourier_features': np.random.randint(5, 20),
        'optimizer': np.random.choice(['adam', 'sgd', 'rmsprop']),
    }
    return config


def run_configuration(config, dataset, budget=10 * 60):
    model = Model(num_hidden_layers=config['num_hidden_layers'], 
                  hidden_size=config['hidden_size'],
                  fourier_features=config['fourier_features'])
    model.to(dataset.device)
    # get the optimizer
    # considered putting objects in sample_configuration, but they do not serialize
    optimizers = {
        'adam': th.optim.Adam,
        'sgd': th.optim.SGD,
        'rmsprop': th.optim.RMSprop,
    }
    optimizer = optimizers[config['optimizer']](model.parameters(), lr=config['learning_rate'])
    loader = th.utils.data.DataLoader(dataset, batch_size=config['batch_size'], shuffle=False)

    start_time = time.time()

    best_loss = float('inf')
    best_model = None
    epoch = 0
    while time.time() - start_time < budget:
        epoch += 1
        train_loss = model.train_epoch(loader, optimizer)
        if train_loss < best_loss:
            best_loss = train_loss
            best_model = copy.deepcopy(model)
        print(f'\rTime {(time.time() - start_time):.2f} - Epoch {epoch} - Loss: {train_loss:.8f}', end='')
    print()
    return best_model, best_loss


def random_search(dataset, model_path, config_path):
    best_loss = float('inf')
    while True:
        config = sample_configuration()
        print(f'Running configuration: {config}')
        model, loss = run_configuration(config, dataset)
        if loss < best_loss:
            print(f'New best loss: {loss}')
            best_loss = loss
            th.save(model.state_dict(), model_path)
            with open(config_path, 'w') as f:
                json.dump(config, f, indent=4)


def main():
    tetmesh_path = 'data/tetmesh'
    jaw_path = 'data/jaw.obj'
    skull_path = 'data/skull.obj'
    neutral_path = 'data/tetmesh_face_surface.obj'
    deformed_path = 'data/ground_truths/deformed_surface_001.obj'

    model_path = 'checkpoints/best_model_rs.pth'
    config_path = 'checkpoints/best_config_rs.json'

    if th.cuda.is_available():
        device = 'cuda'
    else:
        device = 'cpu'
    print(f'Using device: {device}')

    dataset = TetmeshDataset(tetmesh_path, jaw_path, skull_path, neutral_path, deformed_path, device=device)
    random_search(dataset, model_path, config_path)


if __name__ == '__main__':
    main()
