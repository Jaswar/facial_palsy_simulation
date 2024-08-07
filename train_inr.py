import argparse
import pyvista as pv
import torch as th
import numpy as np
from models import INRModel
from dataset import TetmeshDataset
from common import visualize_displacements, train_model, get_optimizer
import json


def main(args):
    with open(args.config_path, 'r') as f:
        config = json.load(f)

    assert not args.benchmark or args.train, 'Cannot benchmark without training'
    
    if th.cuda.is_available():
        device = 'cuda'
    else:
        device = 'cpu'
    print(f'Using device: {device}')

    dataset = TetmeshDataset(args.tetmesh_path, args.jaw_path, args.skull_path, args.neutral_path, args.deformed_path,
                             num_samples=args.num_samples, device=device)
    dataset.visualize()
    
    model = INRModel(num_hidden_layers=config['num_hidden_layers'], 
                     hidden_size=config['hidden_size'], 
                     fourier_features=config['fourier_features'], 
                     w_surface=config['w_surface'],
                     w_jaw=config['w_jaw'],
                     w_skull=config['w_skull'],
                     w_deformation=config['w_deformation'])
    model = th.compile(model)
    model.to(device)

    if args.use_pretrained:
        try:        
            model.load_state_dict(th.load(args.pretrained_path))
        except KeyError as ex:
            print(f'Pretrained INR model architecture must match the simulation model architecture. Error: {ex}')
            return
    
    if args.benchmark:
        args.epochs = 100

    if args.train:
        optimizer = get_optimizer(config, model)
        lr_scheduler = th.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=config['min_lr'])
        if args.benchmark:  # initialization that compiles some of the methods, must be done here to exclude from benchmark
            model.train_epoch(optimizer, dataset, config['batch_size'])
        train_model(model, dataset, 
                    optimizer, lr_scheduler, config['batch_size'], args.epochs, 
                    args.print_interval, args.vis_interval, args.benchmark, args.checkpoint_path)

    if not args.benchmark:
        model.load_state_dict(th.load(args.checkpoint_path))
        visualize_displacements(model, dataset)

        jaw_nodes = dataset.nodes[dataset.jaw_mask]
        predicted_jaw = model.predict(jaw_nodes).cpu().numpy()
        predicted_jaw = predicted_jaw * (dataset.maxv - dataset.minv) + dataset.minv
        np.save(args.predicted_jaw_path, predicted_jaw)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--tetmesh_path', type=str, required=True)
    parser.add_argument('--jaw_path', type=str, required=True)
    parser.add_argument('--skull_path', type=str, required=True)
    parser.add_argument('--neutral_path', type=str, required=True)
    parser.add_argument('--deformed_path', type=str, required=True)
    parser.add_argument('--config_path', type=str, default='configs/config_inr.json')
    parser.add_argument('--checkpoint_path', type=str, required=True)
    parser.add_argument('--predicted_jaw_path', type=str, required=True)

    parser.add_argument('--use_pretrained', action='store_true')
    parser.add_argument('--pretrained_path', type=str)

    parser.add_argument('--train', action='store_true')
    parser.add_argument('--epochs', type=int, default=300)
    parser.add_argument('--num_samples', type=int, default=10000)
    parser.add_argument('--print_interval', type=int, default=1)
    parser.add_argument('--vis_interval', type=int, default=1000)
    parser.add_argument('--benchmark', action='store_true')

    args = parser.parse_args()
    main(args)


