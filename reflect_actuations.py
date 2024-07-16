import torch as th
import numpy as np
from tetmesh import Tetmesh
from model import Model
import pyvista as pv

def get_actuations(deformation_gradient):
    U, s, V = th.svd(deformation_gradient)
    s = th.diag_embed(s)
    A = th.bmm(V, th.bmm(s, V.permute(0, 2, 1)))
    return A


def visualize_actuations(nodes, elements, actuations):
    _, s, _ = th.svd(actuations)
    actuations = th.log(th.sum(s, dim=1))

    cells = np.hstack([np.full((elements.shape[0], 1), 4, dtype=int), elements])
    celltypes = np.full(cells.shape[0], fill_value=pv.CellType.TETRA, dtype=int)

    neutral_grid = pv.UnstructuredGrid(cells, celltypes, nodes)
    neutral_grid['actuations'] = actuations.cpu().numpy()

    plot = pv.Plotter()
    plot.add_mesh(neutral_grid, scalars='actuations', cmap='viridis')
    plot.show()


def main():
    tetmesh_path = 'data/tetmesh'
    model_path = 'checkpoints/best_model_16_07_init.pth'
    actuations_path = 'data/actuations.npy'

    nodes, elements, _ = Tetmesh.read_tetgen_file(tetmesh_path)
    minv = np.min(nodes)
    maxv = np.max(nodes)
    nodes = (nodes - minv) / (maxv - minv)

    midpoint = np.mean(nodes[:, 0])
    relevant_indices = nodes[:, 0] < midpoint

    model = Model(num_hidden_layers=9, hidden_size=64, fourier_features=8)
    model = th.compile(model)
    model.load_state_dict(th.load(model_path))

    with th.no_grad():
        flipped_vertices = nodes.copy()
        flipped_vertices[~relevant_indices, 0] = midpoint - flipped_vertices[~relevant_indices, 0] + midpoint
        deformation_gradient = model.construct_jacobian(th.tensor(flipped_vertices).float())
    
    actuations = get_actuations(deformation_gradient)
    visualize_actuations(nodes, elements, actuations)
    np.save(actuations_path, actuations.cpu().numpy())

if __name__ == '__main__':
    main()
