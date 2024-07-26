import argparse
import pyvista as pv
import torch as th
import numpy as np
from models import INRModel
from dataset import TetmeshDataset
import time
from common import visualize_displacements, train_model


def main(args):
    assert not args.benchmark or args.train, 'Cannot benchmark without training'
    
    if th.cuda.is_available():
        device = 'cuda'
    else:
        device = 'cpu'
    print(f'Using device: {device}')

    prestrain_model = None
    if args.use_prestrain:
        prestrain_model = INRModel(num_hidden_layers=9, hidden_size=64, fourier_features=8)
        prestrain_model = th.compile(prestrain_model)
        prestrain_model.load_state_dict(th.load(args.prestrain_model_path))
    dataset = TetmeshDataset(args.tetmesh_path, args.jaw_path, args.skull_path, args.neutral_path, args.deformed_path, 
                             generate_prestrain=args.generate_prestrain, use_prestrain=args.use_prestrain, prestrain_model=prestrain_model,
                             num_samples=args.num_samples, device=device)
    dataset.visualize()
    
    model = INRModel(num_hidden_layers=9, hidden_size=64, fourier_features=8, w_surface=40. if args.generate_prestrain else 10.)
    model = th.compile(model)
    model.to(device)
    
    if args.benchmark:
        args.epochs = 100

    if args.train:
        optimizer = th.optim.Adam(model.parameters(), lr=0.000845248320219007)
        lr_scheduler = th.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-8)
        if args.benchmark:  # initialization that compiles some of the methods, must be done here to exclude from benchmark
            model.train_epoch(optimizer, dataset, args.batch_size)
        train_model(model, dataset, 
                    optimizer, lr_scheduler, args.batch_size, args.epochs, 
                    args.print_interval, args.vis_interval, args.benchmark, args.checkpoint_path)

    if not args.benchmark:
        model.load_state_dict(th.load(args.checkpoint_path))
        visualize_displacements(model, dataset)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--tetmesh_path', type=str, required=True)
    parser.add_argument('--jaw_path', type=str, required=True)
    parser.add_argument('--skull_path', type=str, required=True)
    parser.add_argument('--neutral_path', type=str, required=True)
    parser.add_argument('--deformed_path', type=str, required=True)
    parser.add_argument('--checkpoint_path', type=str, required=True)

    parser.add_argument('--generate_prestrain', action='store_true')
    parser.add_argument('--use_prestrain', action='store_true')
    parser.add_argument('--prestrain_model_path', type=str)

    parser.add_argument('--train', action='store_true')
    parser.add_argument('--epochs', type=int, default=10000)
    parser.add_argument('--batch_size', type=int, default=4096)
    parser.add_argument('--num_samples', type=int, default=10000)
    parser.add_argument('--print_interval', type=int, default=1)
    parser.add_argument('--vis_interval', type=int, default=1000)
    parser.add_argument('--benchmark', action='store_true')

    args = parser.parse_args()
    main(args)


