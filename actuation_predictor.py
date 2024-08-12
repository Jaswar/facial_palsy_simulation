import torch as th
import numpy as np
from tetmesh import Tetmesh
from models import INRModel
import pyvista as pv
import argparse
import json
from scipy.spatial import KDTree
from stiefel_exp.Stiefel_Exp_Log import Stiefel_Exp, Stiefel_Log
from tqdm import tqdm
import pyquaternion as pq


def positive_determinant(matrix):
    Z = th.eye(matrix.shape[1], device=matrix.device).unsqueeze(0)
    Z = Z.repeat(matrix.shape[0], 1, 1)
    Z[:, -1, -1] *= th.sign(th.det(matrix))
    return th.bmm(matrix, Z)


def get_actuations(deformation_gradient):
    U, s, V = th.svd(deformation_gradient)
    s = th.diag_embed(s)

    # make sure the determinant of V is positive
    V = positive_determinant(V)

    A = th.bmm(V, th.bmm(s, V.permute(0, 2, 1)))
    return V, s, A


def flip_actuations(V, s, flipped_points):
    V_sym = V.clone()
    s_sym = s.clone()
    V_sym[th.tensor(flipped_points).to(V.device), 0, :] *= -1
    A_sym = V_sym @ s_sym @ th.transpose(V_sym, 2, 1)
    return A_sym


# from https://www.researchgate.net/publication/330165271_Parametric_Model_Reduction_via_Interpolating_Orthonormal_Bases
def interpolate_stiefel(actuations_0, actuations_1, alpha):
    u0, s0, v0 = th.svd(actuations_0)
    u0, v0 = positive_determinant(u0), positive_determinant(v0)
    s0 = th.diag_embed(s0)
    u1, s1, v1 = th.svd(actuations_1)
    u1, v1 = positive_determinant(u1), positive_determinant(v1)
    s1 = th.diag_embed(s1)

    u0, s0, v0 = u0.cpu().numpy(), s0.cpu().numpy(), v0.cpu().numpy()
    u1, s1, v1 = u1.cpu().numpy(), s1.cpu().numpy(), v1.cpu().numpy()

    actuations = np.zeros(actuations_0.shape)
    for i in tqdm(range(actuations_0.shape[0])):
        delta_u, _ = Stiefel_Log(u0[i], u1[i], 1e-3)
        delta_s = s1[i] - s0[i]
        delta_v, _ = Stiefel_Log(v0[i], v1[i], 1e-3)
        actuations[i] = Stiefel_Exp(u0[i], alpha * delta_u) @ (s0[i] + alpha * delta_s) @ Stiefel_Exp(v0[i], alpha * delta_v).T
    return th.tensor(actuations).float().to(actuations_0.device)


def interpolate_slerp(v0, v1, s0, s1, alpha):
    device = v0.device
    v0, s0 = v0.cpu().numpy(), s0.cpu().numpy()
    v1, s1 = v1.cpu().numpy(), s1.cpu().numpy()

    v_out = np.zeros(v0.shape)
    for i in tqdm(range(v0.shape[0])):
        quaternion_0 = pq.Quaternion(matrix=v0[i], atol=1e-5)
        quaternion_1 = pq.Quaternion(matrix=v1[i], atol=1e-5)
        end = pq.Quaternion.slerp(quaternion_0, quaternion_1, alpha)
        v_out[i] = end.rotation_matrix
    s_out = s0 + alpha * (s1 - s0)
    s_out = th.tensor(s_out).float().to(device)
    v_out = th.tensor(v_out).float().to(device)
    A_out = th.bmm(v_out, th.bmm(s_out, v_out.permute(0, 2, 1)))
    return A_out



class ActuationPredictor(object):

    def __init__(self, model_path, config_path, 
                 tetmesh_path, tetmesh_contour_path, tetmesh_reflected_deformed_path, 
                 interpolation=None, alpha=0.5,
                 secondary_model_path=None, device='cpu'):
        self.model_path = model_path
        self.config_path = config_path
        self.tetmesh_path = tetmesh_path
        self.tetmesh_contour_path = tetmesh_contour_path
        self.tetmesh_reflected_deformed_path = tetmesh_reflected_deformed_path
        self.interpolation = interpolation
        self.alpha = alpha
        self.secondary_model_path = secondary_model_path
        self.device = device

        assert self.interpolation in [None, 'slerp', 'stiefel'], 'Invalid interpolation method, only slerp and stiefel are supported'
        assert 0.0 <= self.alpha <= 1.0, 'Alpha must be in the range [0, 1]'
        assert not self.interpolation or self.secondary_model_path is not None, 'Interpolation requires a secondary model'

        self.__load()

        self.surface_kdtree = KDTree(self.surface.points)
        self.nodes_kdtree = KDTree(self.nodes)

    def predict(self, points, denormalize=False):
        if denormalize:
            points = points * (self.maxv - self.minv) + self.minv
        
        if self.secondary_model is None:
            return self.__predict_symmetric(points)
        else:
            return self.__predict_with_secondary(points)
    
    def __predict_symmetric(self, points):
        points = points.cpu().numpy()

        flipped_points = points[:, 0] > np.mean(self.nodes[:, 0])
        new_points = points.copy()
        new_points[flipped_points] = self.__bary_transform(new_points[flipped_points])

        new_points = th.tensor(new_points).float().to(self.device)
        new_points = (new_points - self.minv) / (self.maxv - self.minv)
        with th.no_grad():
            deformation_gradient = self.model.construct_jacobian(new_points)
        V, s, A = get_actuations(deformation_gradient)

        A_sym = flip_actuations(V, s, flipped_points)
        return A, A_sym

    def __predict_with_secondary(self, points):
        flipped_points = points[:, 0] > th.mean(th.tensor(self.nodes[:, 0]).to(self.device))

        points = (points - self.minv) / (self.maxv - self.minv)
        main_points = points.clone()
        secondary_points = points[flipped_points]
        A = th.zeros((points.shape[0], 3, 3), device=self.device)

        with th.no_grad():
            main_gradient = self.model.construct_jacobian(main_points)
            secondary_gradient = self.secondary_model.construct_jacobian(secondary_points)
        main_V, main_s, main_A = get_actuations(main_gradient)
        secondary_V, secondary_s, secondary_A = get_actuations(secondary_gradient)

        if self.interpolation == 'slerp' and self.alpha != 0.0 and self.alpha != 1.0:
            flipped_A = interpolate_slerp(main_V[flipped_points], secondary_V, main_s[flipped_points], secondary_s, self.alpha)
        elif self.interpolation == 'stiefel' and self.alpha != 0.0 and self.alpha != 1.0:
            flipped_A = interpolate_stiefel(main_A[flipped_points], secondary_A, self.alpha)
        elif self.interpolation is None or self.alpha == 1.0:
            flipped_A = secondary_A
        elif self.alpha == 0.0:
            flipped_A = main_A[flipped_points]
        else:
            raise ValueError(f'Invalid interpolation method: {self.interpolation}, only slerp and stiefel are supported')
        
        A[~flipped_points] = main_A[~flipped_points]
        A[flipped_points] = flipped_A
        A_sym = A.clone()
        return A, A_sym

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

        self.secondary_model = None
        if self.secondary_model_path is not None:
            self.secondary_model = INRModel(num_hidden_layers=config['num_hidden_layers'],
                                            hidden_size=config['hidden_size'],
                                            fourier_features=config['fourier_features'])
            self.secondary_model = th.compile(self.secondary_model)
            self.secondary_model.load_state_dict(th.load(self.secondary_model_path))
            self.secondary_model.to(self.device)

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

    def visualize(self, points=None):
        A, A_sym = self.predict(th.tensor(self.nodes).float().to(self.device))
        self.__visualize(A, A_sym, points)

    def __visualize(self, A, A_sym, points=None):
        _, s, _ = th.svd(A)
        actuations = th.sum(s, dim=1)
        _, s, _ = th.svd(A_sym)
        actuations_sym = th.sum(s, dim=1)
        actuations = actuations.cpu().numpy()
        actuations_sym = actuations_sym.cpu().numpy()

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
                                   args.tetmesh_reflected_deformed_path,
                                   args.secondary_model_path)

    points = predictor.nodes.copy()
    A, A_sym = predictor.predict(points)

    if not args.silent:
        predictor.visualize()
    np.save(args.out_actuations_path, A_sym)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--tetmesh_path', type=str, required=True)
    parser.add_argument('--tetmesh_contour_path', type=str, required=True)
    parser.add_argument('--tetmesh_reflected_deformed_path', type=str, required=True)
    parser.add_argument('--model_path', type=str, required=True)
    parser.add_argument('--secondary_model_path', type=str, default=None)
    parser.add_argument('--config_path', type=str, default='configs/config_inr.json')
    parser.add_argument('--out_actuations_path', type=str, required=True)
    parser.add_argument('--silent', action='store_true')

    args = parser.parse_args()
    main(args)
