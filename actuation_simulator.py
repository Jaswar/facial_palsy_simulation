import argparse
import pyvista as pv
import torch as th
import numpy as np
from models import SimulatorModel, INRModel
from dataset import TetmeshDataset
import time
from common import visualize_displacements, train_model


def main():
    tetmesh_path = 'data/tetmesh'
    jaw_path = 'data/jaw.obj'
    skull_path = 'data/skull.obj'
    neutral_path = 'data/tetmesh_face_surface.obj'
    deformed_path = 'data/ground_truths/deformed_surface_017.obj'  # 17 for figure 37 from the thesis
    actuations_path = 'data/act_sym_017_per_vertex.npy'
    checkpoint_path = 'checkpoints/best_model_simulator_017.pth'
    train = False

    # it is possible to use the deformation model as the starting point for this network
    # the model converges a bit faster, but it requires the deformation model to have the same architecture
    use_pretrained = False
    pretrained_path = 'checkpoints/best_model_017.pth'

    use_prestrain = False  # whether to use the INR for the symmetric face
    prestrain_model_path = 'checkpoints/best_model_prestrain.pth'  # path to the INR for the symmetric/prestrained face

    epochs = 10000
    batch_size = 4096
    num_samples = 10000  # how many nodes to sample from the tetmesh
    print_interval = 1
    vis_interval = 1000
    benchmark = False  # execute only 100 epochs, exclude compilation time, do not save the model

    assert not benchmark or train, 'Cannot benchmark without training'
    
    if th.cuda.is_available():
        device = 'cuda'
    else:
        device = 'cpu'
    print(f'Using device: {device}')

    prestrain_model = None
    if use_prestrain:
        prestrain_model = INRModel(num_hidden_layers=9, hidden_size=64, fourier_features=8)
        prestrain_model = th.compile(prestrain_model)
        prestrain_model.load_state_dict(th.load(prestrain_model_path))
    dataset = TetmeshDataset(tetmesh_path, jaw_path, skull_path, neutral_path, deformed_path, actuations_path,
                             use_prestrain=use_prestrain, prestrain_model=prestrain_model,
                             num_samples=num_samples, device=device)
    dataset.visualize()
    
    model = SimulatorModel(num_hidden_layers=9, hidden_size=64, fourier_features=8)
    model = th.compile(model)
    model.to(device)

    if use_pretrained:
        model.load_state_dict(th.load(pretrained_path))

    if benchmark:
        epochs = 100

    if train:
        optimizer = th.optim.Adam(model.parameters(), lr=0.00845248320219007)
        lr_scheduler = th.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-8)
        if benchmark:  # initialization that compiles some of the methods, must be done here to exclude from benchmark
            model.train_epoch(optimizer, dataset, batch_size)
        train_model(model, dataset, optimizer, lr_scheduler, batch_size, epochs, print_interval, vis_interval, benchmark, checkpoint_path)

    if not benchmark:
        model.load_state_dict(th.load(checkpoint_path))
        visualize_displacements(model, dataset)

if __name__ == '__main__':
    main()