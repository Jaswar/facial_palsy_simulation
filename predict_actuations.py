import torch as th
import numpy as np
from tetmesh import Tetmesh
from models import INRModel
import pyvista as pv
import argparse
import json
from scipy.spatial import KDTree


def get_actuations(deformation_gradient):
    U, s, V = th.svd(deformation_gradient)
    s = th.diag_embed(s)
    A = th.bmm(V, th.bmm(s, V.permute(0, 2, 1)))
    return V, s, A


def visualize_actuations(nodes, elements, A, A_sym):
    _, s, _ = th.svd(th.tensor(A).float())
    actuations = th.sum(s, dim=1)
    _, s, _ = th.svd(th.tensor(A_sym).float())
    actuations_sym = th.sum(s, dim=1)

    cells = np.hstack([np.full((elements.shape[0], 1), 4, dtype=int), elements])
    celltypes = np.full(cells.shape[0], fill_value=pv.CellType.TETRA, dtype=int)

    neutral_grid = pv.UnstructuredGrid(cells, celltypes, nodes)
    neutral_grid['actuations'] = actuations
    neutral_grid['actuations_sym'] = actuations_sym

    plot = pv.Plotter(shape=(1, 2))
    plot.subplot(0, 0)
    plot.add_mesh(neutral_grid.copy(), scalars='actuations', clim=(2, 4), cmap='RdBu')
    plot.subplot(0, 1)
    plot.add_mesh(neutral_grid.copy(), scalars='actuations_sym', clim=(2, 4), cmap='RdBu')
    plot.link_views()
    plot.show()


def bary_transform(points, surface, deformed_surface, kdtree):
    if points.ndim == 1:
        points = points[None, :]
    n_points = points.shape[0]
    # find NNs on undeformed mesh
    d, idcs = kdtree.query(points, k=5)
    old_nns = surface.points[idcs]
    # look up positions on deformed, flipped mesh (implicit mapping)
    new_nns = deformed_surface.points[idcs]
    # find the "offset" from the query point to the NNs and flip the x
    offset = old_nns - points[:, None, :]
    offset[:, :, 0] *= -1

    # find the possible "new" positions and calculate weighted average
    new_pts = new_nns - offset
    d = d[:, :, None]  # n_points, k, 1
    new_pos = np.average(new_pts, weights=1.0/(d+1e-6).repeat(3, axis=2), axis=1)
    return new_pos


def flip_actuations(V, s, flipped_points, mappped_indices):
    V_sym = V.copy()
    s_sym = s.copy()
    V_sym[flipped_points] = V[mappped_indices]
    s_sym[flipped_points] = s[mappped_indices]
    V_sym[flipped_points, 0, :] *= -1
    A_sym = V_sym @ s_sym @ np.transpose(V_sym, [0, 2, 1])
    return A_sym


def main(args):
    tetmesh_path = 'data/tetmesh'
    model_path = args.model_path
    config_path = args.config_path
    tetmesh_contour_path = 'data/tetmesh_contour.obj'
    tetmesh_reflected_deformed_path = 'data/tetmesh_contour_ref_deformed.obj'
    out_actuations_path = 'data/act_sym_017_per_vertex.npy'

    with open(config_path, 'r') as f:
        config = json.load(f)

    nodes, elements, _ = Tetmesh.read_tetgen_file(args.tetmesh_path)
    minv = np.min(nodes)
    maxv = np.max(nodes)
    nodes = (nodes - minv) / (maxv - minv)

    model = INRModel(num_hidden_layers=config['num_hidden_layers'], 
                  hidden_size=config['hidden_size'],
                  fourier_features=config['fourier_features'])
    model = th.compile(model)
    model.load_state_dict(th.load(args.model_path))

    with th.no_grad():
        deformation_gradient = model.construct_jacobian(th.tensor(nodes).float())
    
    V, s, A = get_actuations(deformation_gradient)
    V, s, A = V.cpu().numpy(), s.cpu().numpy(), A.cpu().numpy()

    surface = pv.PolyData(args.tetmesh_contour_path)
    deformed_surface = pv.PolyData(args.tetmesh_reflected_deformed_path)
    kdtree = KDTree(surface.points)

    nodes = nodes * (maxv - minv) + minv  # denormalize
    flipped_points = nodes[:, 0] > np.mean(nodes[:, 0])
    new_points = bary_transform(nodes, surface, deformed_surface, kdtree)

    kdtree = KDTree(nodes)
    _, mapped_indices = kdtree.query(new_points[flipped_points])
    A_sym = flip_actuations(V, s, flipped_points, mapped_indices)

    visualize_actuations(nodes, elements, A, A_sym)
    np.save(args.out_actuations_path, A_sym)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--tetmesh_path', type=str, required=True)
    parser.add_argument('--tetmesh_contour_path', type=str, required=True)
    parser.add_argument('--tetmesh_reflected_deformed_path', type=str, required=True)
    parser.add_argument('--model_path', type=str, required=True)
    parser.add_argument('--config_path', type=str, required=True)

    args = parser.parse_args()
    main(args)
