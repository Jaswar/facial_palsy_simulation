import torch as th
import pyvista as pv
import numpy as np
import os
import json
import argparse
from models import SimulatorModel
from datasets import TetmeshDataset
from common import visualize_displacements


def main(args):
    if th.cuda.is_available():
        device = 'cuda'
    else:
        device = 'cpu'
    print(f'Using device: {device}')

    dataset = TetmeshDataset(args.tetmesh_path, args.jaw_path, args.skull_path, args.neutral_path, args.deformed_path, device=device)

    for file in os.listdir(args.checkpoints_path):
        if file.endswith('.pth') and 'sim' in file:
            print(f'Visualizing {file}')
            model = SimulatorModel(num_hidden_layers=9, hidden_size=64, fourier_features=8)
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
