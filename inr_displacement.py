import argparse
import pyvista as pv
import torch as th
import numpy as np
from model import Model
from dataset import MeshDataset

def visualize_displacements(model, dataset):
    predicted_displacements = model.predict(dataset.neutral_vertices)
    all_predicted_vertices = dataset.neutral_vertices + predicted_displacements
    ground_truth_vertices = dataset.neutral_vertices + dataset.displacements

    # only replace the healthy part of the face
    predicted_vertices = ground_truth_vertices.clone()
    predicted_vertices[dataset.relevant_indices] = all_predicted_vertices[dataset.relevant_indices]

    predicted_vertices = predicted_vertices.cpu().numpy()
    ground_truth_vertices = ground_truth_vertices.cpu().numpy()

    # the "3" is required by pyvista, it's the number of vertices per face
    cells = np.array([[3, *[v[0] - 1 for v in face]] for face in dataset.faces])
    # fill value of 5 to specify VTK_TRIANGLE
    celltypes = np.full(cells.shape[0], fill_value=5, dtype=int)

    neutral_grid = pv.UnstructuredGrid(cells, celltypes, dataset.neutral_vertices.cpu().numpy())
    predicted_grid = pv.UnstructuredGrid(cells, celltypes, predicted_vertices)
    ground_truth_grid = pv.UnstructuredGrid(cells, celltypes, ground_truth_vertices)
    
    plot = pv.Plotter(shape=(1, 3))
    plot.subplot(0, 0)
    plot.add_mesh(neutral_grid.copy(), color='yellow')
    plot.subplot(0, 1)
    plot.add_mesh(predicted_grid.copy(), color='yellow')
    plot.subplot(0, 2)
    plot.add_mesh(ground_truth_grid.copy(), color='yellow')
    plot.link_views()
    plot.show()


def main():
    neutral_path = 'data/face_surface_with_uv3.obj'
    deformed_path = 'data/ground_truths/deformed_surface_001.obj'
    checkpoint_path = 'checkpoints/best_model.pth'
    epochs = 1000
    batch_size = 32
    train = True

    if th.cuda.is_available():
        device = 'cuda'
    else:
        device = 'cpu'
    print(f'Using device: {device}')
    # device = 'cpu'

    dataset = MeshDataset(neutral_path, deformed_path, device=device)
    loader = th.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False)
    model = Model(input_size=dataset.dimensionality(), output_size=dataset.dimensionality(), num_hidden_layers=8, hidden_size=256)
    model.to(device)
    
    if train:
        criterion = th.nn.L1Loss()
        optimizer = th.optim.Adam(model.parameters(), lr=1e-4)
        min_loss = float('inf')
        for epoch in range(1, epochs + 1):
            train_loss = model.train_epoch(loader, optimizer, criterion)
            if train_loss < min_loss:
                min_loss = train_loss
                th.save(model.state_dict(), checkpoint_path)
            print(f'Epoch {epoch}/{epochs} - Loss: {train_loss:.6f}')
            if epoch % 100 == 0:
                visualize_displacements(model, dataset)

    model.load_state_dict(th.load(checkpoint_path))
    visualize_displacements(model, dataset)

if __name__ == '__main__':
    main()


