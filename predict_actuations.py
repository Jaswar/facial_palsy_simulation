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


def flip_actuations(V, s, flipped_points):
    V_sym = V.copy()
    s_sym = s.copy()
    V_sym[flipped_points, 0, :] *= -1
    A_sym = V_sym @ s_sym @ np.transpose(V_sym, [0, 2, 1])
    return A_sym


class ActuationPredictor(object):

    def __init__(self, model_path, config_path, 
                 tetmesh_path, tetmesh_contour_path, tetmesh_reflected_deformed_path, device='cpu'):
        self.model_path = model_path
        self.config_path = config_path
        self.tetmesh_path = tetmesh_path
        self.tetmesh_contour_path = tetmesh_contour_path
        self.tetmesh_reflected_deformed_path = tetmesh_reflected_deformed_path
        self.device = device

        self.__load()

        self.surface_kdtree = KDTree(self.surface.points)
        self.nodes_kdtree = KDTree(self.nodes)


    def predict(self, points, denormalize=False):
        if th.is_tensor(points):
            points = points.cpu().numpy()

        if denormalize:
            points = points * (self.maxv - self.minv) + self.minv

        flipped_points = points[:, 0] > np.mean(self.nodes[:, 0])
        new_points = points.copy()
        new_points[flipped_points] = self.__bary_transform(new_points[flipped_points])

        new_points = th.tensor(new_points).float().to(self.device)
        new_points = (new_points - self.minv) / (self.maxv - self.minv)
        with th.no_grad():
            deformation_gradient = self.model.construct_jacobian(new_points)
        V, s, A = get_actuations(deformation_gradient)
        self.V, self.s, self.A = V.cpu().numpy(), s.cpu().numpy(), A.cpu().numpy()

        # _, mapped_indices = self.nodes_kdtree.query(new_points)
        A_sym = flip_actuations(self.V, self.s, flipped_points)
        return self.A, A_sym


    def __load(self):
        with open(self.config_path, 'r') as f:
            config = json.load(f)
        
        self.nodes, self.elements, _ = Tetmesh.read_tetgen_file(self.tetmesh_path)
        self.minv = np.min(self.nodes)
        self.maxv = np.max(self.nodes)
        
        self.model = INRModel(num_hidden_layers=config['num_hidden_layers'],
                              hidden_size=config['hidden_size'],
                              fourier_features=config['fourier_features'])
        self.model = th.compile(self.model)
        self.model.load_state_dict(th.load(self.model_path))
        self.model.to(self.device)

        self.surface = pv.PolyData(self.tetmesh_contour_path)
        self.deformed_surface = pv.PolyData(self.tetmesh_reflected_deformed_path)
        

    def __bary_transform(self, points):
        if points.ndim == 1:
            points = points[None, :]
        n_points = points.shape[0]
        # find NNs on undeformed mesh
        d, idcs = self.surface_kdtree.query(points, k=5)
        old_nns = self.surface.points[idcs]
        # look up positions on deformed, flipped mesh (implicit mapping)
        new_nns = self.deformed_surface.points[idcs]
        # find the "offset" from the query point to the NNs and flip the x
        offset = old_nns - points[:, None, :]
        offset[:, :, 0] *= -1

        # find the possible "new" positions and calculate weighted average
        new_pts = new_nns - offset
        d = d[:, :, None]  # n_points, k, 1
        new_pos = np.average(new_pts, weights=1.0/(d+1e-6).repeat(3, axis=2), axis=1)
        return new_pos


    def visualize(self, A, A_sym, points=None):
        _, s, _ = th.svd(th.tensor(A).float())
        actuations = th.sum(s, dim=1)
        _, s, _ = th.svd(th.tensor(A_sym).float())
        actuations_sym = th.sum(s, dim=1)

        if points is None:
            cells = np.hstack([np.full((self.elements.shape[0], 1), 4, dtype=int), self.elements])
            celltypes = np.full(cells.shape[0], fill_value=pv.CellType.TETRA, dtype=int)
            neutral_grid = pv.UnstructuredGrid(cells, celltypes, self.nodes)
        else:
            cells = np.hstack([np.full((points.shape[0], 1), 1, dtype=int), np.arange(points.shape[0])[:, None]])
            celltypes = np.full(cells.shape[0], fill_value=pv.CellType.VERTEX, dtype=int)
            neutral_grid = pv.UnstructuredGrid(cells, celltypes, points)

        neutral_grid['actuations'] = actuations
        neutral_grid['actuations_sym'] = actuations_sym

        plot = pv.Plotter(shape=(1, 2))
        plot.subplot(0, 0)
        plot.add_mesh(neutral_grid.copy(), scalars='actuations', clim=(2, 4), cmap='RdBu')
        plot.subplot(0, 1)
        plot.add_mesh(neutral_grid.copy(), scalars='actuations_sym', clim=(2, 4), cmap='RdBu')
        plot.link_views()
        plot.show()


def main(args):
    predictor = ActuationPredictor(args.model_path, 
                                   args.config_path, 
                                   args.tetmesh_path, 
                                   args.tetmesh_contour_path, 
                                   args.tetmesh_reflected_deformed_path)

    points = predictor.nodes.copy()
    A, A_sym = predictor.predict(points)

    if not args.silent:
        predictor.visualize(A, A_sym)
    np.save(args.out_actuations_path, A_sym)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--tetmesh_path', type=str, required=True)
    parser.add_argument('--tetmesh_contour_path', type=str, required=True)
    parser.add_argument('--tetmesh_reflected_deformed_path', type=str, required=True)
    parser.add_argument('--model_path', type=str, required=True)
    parser.add_argument('--config_path', type=str, default='configs/config_inr.json')
    parser.add_argument('--out_actuations_path', type=str, required=True)
    parser.add_argument('--silent', action='store_true')

    args = parser.parse_args()
    main(args)
