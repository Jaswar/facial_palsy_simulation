import argparse
import pyvista as pv
import torch as th
import numpy as np
from model import Model
from dataset import TetmeshDataset
import time

def visualize_displacements(model, dataset, pass_all=False):
    if pass_all:
        mask = th.zeros(dataset.nodes.shape[0], dtype=th.bool)
        mask[dataset.relevant_indices] = True
        flipped_vertices = dataset.nodes.copy()
        flipped_vertices[~mask, 0] = dataset.midpoint - flipped_vertices[~mask, 0] + dataset.midpoint
        predicted_vertices = model.predict(th.tensor(flipped_vertices).to(dataset.device).float()).cpu().numpy()
        predicted_vertices[~mask, 0] = dataset.midpoint - predicted_vertices[~mask, 0] + dataset.midpoint
    else:
        predicted_vertices = dataset.deformed_nodes.copy()
        indices = np.zeros(dataset.nodes.shape[0], dtype=bool)
        indices[dataset.relevant_indices] = True
        part_indices = np.logical_or(dataset.mask == 1, dataset.mask == 3)
        part_indices = np.logical_or(part_indices, dataset.mask == 2)
        part_indices = np.logical_or(part_indices, dataset.mask == 0)
        indices = np.logical_and(indices, part_indices)
        predicted_vertices[indices] = model.predict(th.tensor(dataset.nodes).to(dataset.device).float()).cpu().numpy()[indices]

    cells = np.hstack([np.full((dataset.elements.shape[0], 1), 4, dtype=int), dataset.elements])
    celltypes = np.full(cells.shape[0], fill_value=pv.CellType.TETRA, dtype=int)

    neutral_grid = pv.UnstructuredGrid(cells, celltypes, dataset.nodes)
    predicted_grid = pv.UnstructuredGrid(cells, celltypes, predicted_vertices)
    ground_truth_grid = pv.UnstructuredGrid(cells, celltypes, dataset.deformed_nodes)
    
    plot = pv.Plotter(shape=(1, 3))
    plot.subplot(0, 0)
    plot.add_mesh(neutral_grid.copy(), color='yellow')
    plot.subplot(0, 1)
    plot.add_mesh(ground_truth_grid.copy(), color='yellow')
    plot.subplot(0, 2)
    plot.add_mesh(predicted_grid.copy(), color='yellow')
    plot.link_views()
    plot.show()


def main():
    tetmesh_path = 'data/tetmesh'
    jaw_path = 'data/jaw.obj'
    skull_path = 'data/skull.obj'
    neutral_path = 'data/tetmesh_face_surface.obj'
    deformed_path = 'data/ground_truths/deformed_surface_001.obj'
    checkpoint_path = 'checkpoints/best_model.pth'
    train = True
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

    dataset = TetmeshDataset(tetmesh_path, jaw_path, skull_path, neutral_path, deformed_path, num_samples=num_samples, device=device)
    model = Model(num_hidden_layers=9, hidden_size=64, fourier_features=8)
    model = th.compile(model)
    model.to(device)
    
    if benchmark:
        epochs = 100

    if train:
        optimizer = th.optim.Adam(model.parameters(), lr=0.000845248320219007)
        lr_scheduler = th.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-8)
        if benchmark:  # initialization that compiles some of the methods, must be done here to exclude from benchmark
            model.train_epoch(optimizer, dataset, batch_size, dataset.device)
        min_loss = float('inf')
        start = time.time()
        for epoch in range(1, epochs + 1):
            train_loss = model.train_epoch(optimizer, dataset, batch_size, dataset.device)
            if train_loss < min_loss and not benchmark:
                min_loss = train_loss
                th.save(model.state_dict(), checkpoint_path)
            lr_scheduler.step()

            if epoch % print_interval == 0:
                print(f'Epoch {epoch}/{epochs} - Loss: {train_loss:.8f} - LR: {lr_scheduler.get_last_lr()[0]:.8f}')
            if epoch % vis_interval == 0 and not benchmark:
                visualize_displacements(model, dataset)
        print(f'Training took: {time.time() - start:.2f}s')

    if not benchmark:
        model.load_state_dict(th.load(checkpoint_path))
        visualize_displacements(model, dataset, pass_all=True)

if __name__ == '__main__':
    main()


