import argparse
import pyvista as pv
import torch as th
import numpy as np
from model import Model
from dataset import TetmeshDataset

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
    checkpoint_path = 'checkpoints/best_model_1.pth'
    epochs = 1000
    batch_size = 32
    train = True
    vis_interval = 100

    if th.cuda.is_available():
        device = 'cuda'
    else:
        device = 'cpu'
    print(f'Using device: {device}')

    dataset = TetmeshDataset(tetmesh_path, jaw_path, skull_path, neutral_path, deformed_path, device=device)
    loader = th.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False)
    model = Model(num_hidden_layers=8, hidden_size=256)
    model.to(device)
    
    if train:
        optimizer = th.optim.Adam(model.parameters(), lr=1e-4)
        lr_scheduler = th.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-8)
        min_loss = float('inf')
        for epoch in range(1, epochs + 1):
            train_loss = model.train_epoch(loader, optimizer)
            if train_loss < min_loss:
                min_loss = train_loss
                th.save(model.state_dict(), checkpoint_path)
            print(f'Epoch {epoch}/{epochs} - Loss: {train_loss:.8f} - LR: {lr_scheduler.get_last_lr()[0]:.8f}')
            if epoch % vis_interval == 0:
                visualize_displacements(model, dataset)
            lr_scheduler.step()

    model.load_state_dict(th.load(checkpoint_path))
    visualize_displacements(model, dataset, pass_all=True)

if __name__ == '__main__':
    main()


