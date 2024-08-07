import argparse
import torch as th
from models import SimulatorModel, INRModel
from datasets import SimulatorDataset
from common import visualize_displacements, train_model, get_optimizer
from predict_actuations import ActuationPredictor
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

    actuation_predictor = ActuationPredictor(args.main_actuation_model_path, 
                                             args.actuation_model_config_path, 
                                             args.tetmesh_path, 
                                             args.contour_path, 
                                             args.reflected_contour_path, 
                                             secondary_model_path=args.secondary_actuation_model_path, 
                                             device=device)
    dataset = SimulatorDataset(args.tetmesh_path, args.jaw_path, args.skull_path, args.predicted_jaw_path,
                             actuation_predictor=actuation_predictor, num_samples=args.num_samples, device=device)
    dataset.visualize()
    
    model = SimulatorModel(num_hidden_layers=config['num_hidden_layers'], 
                     hidden_size=config['hidden_size'], 
                     fourier_features=config['fourier_features'], 
                     w_energy=config['w_energy'],
                     w_fixed=config['w_fixed'])
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


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--tetmesh_path', type=str, required=True)
    parser.add_argument('--jaw_path', type=str, required=True)
    parser.add_argument('--skull_path', type=str, required=True)
    parser.add_argument('--neutral_path', type=str, required=True)
    parser.add_argument('--deformed_path', type=str, required=True)
    parser.add_argument('--predicted_jaw_path', type=str, default=None)
    parser.add_argument('--config_path', type=str, default='configs/config_simulation.json')
    parser.add_argument('--checkpoint_path', type=str, required=True)

    parser.add_argument('--main_actuation_model_path', type=str, required=True)
    parser.add_argument('--actuation_model_config_path', type=str, default='configs/config_inr.json')
    parser.add_argument('--contour_path', type=str, required=True)
    parser.add_argument('--reflected_contour_path', type=str, required=True)
    parser.add_argument('--secondary_actuation_model_path', type=str, default=None)

    parser.add_argument('--use_pretrained', action='store_true')
    parser.add_argument('--pretrained_path', type=str)

    parser.add_argument('--train', action='store_true')
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--num_samples', type=int, default=10000)
    parser.add_argument('--print_interval', type=int, default=1)
    parser.add_argument('--vis_interval', type=int, default=1000)
    parser.add_argument('--benchmark', action='store_true')

    args = parser.parse_args()
    main(args)