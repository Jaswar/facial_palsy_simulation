import torch as th
import numpy as np
from tetmesh import Tetmesh
from model import Model
import pyvista as pv
import argparse
import json

def get_actuations(deformation_gradient):
    U, s, V = th.svd(deformation_gradient)
    s = th.diag_embed(s)
    return V, s


def visualize_actuations(nodes, elements, actuations):
    _, s, _ = th.svd(actuations)
    actuations = th.sum(s, dim=1)

    cells = np.hstack([np.full((elements.shape[0], 1), 4, dtype=int), elements])
    celltypes = np.full(cells.shape[0], fill_value=pv.CellType.TETRA, dtype=int)

    neutral_grid = pv.UnstructuredGrid(cells, celltypes, nodes)
    neutral_grid['actuations'] = actuations.cpu().numpy()

    plot = pv.Plotter()
    plot.add_mesh(neutral_grid, scalars='actuations', clim=(2, 4), cmap='RdBu')
    plot.show()


def main(args):
    tetmesh_path = 'data/tetmesh'
    model_path = args.model_path
    V_path = args.V_path
    s_path = args.s_path
    config_path = args.config_path

    with open(config_path, 'r') as f:
        config = json.load(f)

    nodes, elements, _ = Tetmesh.read_tetgen_file(tetmesh_path)
    minv = np.min(nodes)
    maxv = np.max(nodes)
    nodes = (nodes - minv) / (maxv - minv)
    barries = nodes[elements]
    barries = np.mean(barries, axis=1)
    barries = th.tensor(barries).float()

    model = Model(num_hidden_layers=config['num_hidden_layers'], 
                  hidden_size=config['hidden_size'],
                  fourier_features=config['fourier_features'],
                  w_surface=config['w_surface'],
                  w_tissue=config['w_tissue'],
                  w_jaw=config['w_jaw'],
                  w_skull=config['w_skull'])
    model = th.compile(model)
    model.load_state_dict(th.load(model_path))

    with th.no_grad():
        deformation_gradient = model.construct_jacobian(barries)
    
    V, s = get_actuations(deformation_gradient)
    # visualize_actuations(nodes, elements, actuations)
    np.save(V_path, V.cpu().numpy())
    np.save(s_path, s.cpu().numpy())

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, default='checkpoints/best_model_017_from_prestrain.pth')
    parser.add_argument('--V_path', type=str, default='data/V_017_from_prestrain.npy')
    parser.add_argument('--s_path', type=str, default='data/s_017_from_prestrain.npy')
    parser.add_argument('--config_path', type=str, default='checkpoints/best_config.json')
    args = parser.parse_args()
    main(args)
