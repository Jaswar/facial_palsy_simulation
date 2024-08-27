import torch as th
import pyvista as pv
import numpy as np
import os
import json
import argparse
from models import INRModel, SimulatorModel
from datasets import INRDataset, SimulatorDataset
from common import visualize_displacements


def main(args):
    if th.cuda.is_available():
        device = 'cuda'
    else:
        device = 'cpu'
    print(f'Using device: {device}')

    dataset = INRDataset(args.tetmesh_path, args.jaw_path, args.skull_path, args.neutral_path, args.deformed_path, device=device)

    for file in os.listdir(args.checkpoints_path):
        if file.endswith('.pth') and 'sim' not in file:
            print(f'Visualizing {file}')
            config_path = os.path.join(args.checkpoints_path, file.replace('.pth', '.json').replace('model', 'config'))
            with open(config_path, 'r') as f:
                config = json.load(f)
            model = INRModel(num_hidden_layers=config['num_hidden_layers'], 
                     hidden_size=config['hidden_size'], 
                     fourier_features=config['fourier_features'])
            model = th.compile(model)
            model.load_state_dict(th.load(os.path.join(args.checkpoints_path, file)))
            model.to(device)

            visualize_displacements(model, dataset)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--tetmesh_path', type=str, required=True)
    parser.add_argument('--jaw_path', type=str, required=True)
    parser.add_argument('--skull_path', type=str, required=True)
    parser.add_argument('--neutral_path', type=str, required=True)
    parser.add_argument('--deformed_path', type=str, required=True)
    parser.add_argument('--checkpoints_path', type=str, required=True)
    
    args = parser.parse_args()
    main(args)
