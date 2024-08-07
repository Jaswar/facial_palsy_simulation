import torch as th
import numpy as np
from tetmesh import Tetmesh
import pyvista as pv
from models import SimulatorModel
from scipy.spatial import KDTree
import igl


def main():
    if th.cuda.is_available():
        device = 'cuda'
    else:
        device = 'cpu'
    print(f'Using device: {device}')

    surface = pv.PolyData('data/tetmesh_face_surface.obj')
    print('Loading high res surface')
    high_res_surface = pv.PolyData('data/high_res_surface.obj')

    neutral_projection = surface.project_points_to_plane()
    high_res_projection = high_res_surface.project_points_to_plane()
    neutral_projection.points[:, 2] = 0.
    high_res_projection.points[:, 2] = 0.

    print('Removing points')
    dp, _, _ = igl.point_mesh_squared_distance(high_res_projection.points, neutral_projection.points, neutral_projection.regular_faces)
    d3d, _, _ = igl.point_mesh_squared_distance(high_res_surface.points, surface.points, surface.regular_faces)
    in_bounds = np.logical_and(dp < 0.001, d3d < 20.0)
    high_res_surface, _ = high_res_surface.remove_points(~in_bounds)
    print('Points removed')

    maxv = np.max(surface.points)
    minv = np.min(surface.points)

    high_res_surface.points = (high_res_surface.points - minv) / (maxv - minv)
    surface.points = (surface.points - minv) / (maxv - minv)

    model = SimulatorModel(num_hidden_layers=9, hidden_size=64, fourier_features=8)
    model = th.compile(model)
    model.load_state_dict(th.load('checkpoints/best_model_simulator_017_pair_sampling.pth'))
    model.to(device)
    
    inputs = th.tensor(high_res_surface.points).float().to(device)
    outputs = model.predict(inputs).cpu().numpy()
    outputs = outputs * (maxv - minv) + minv
    high_res_surface.points = outputs    

    plot = pv.Plotter()
    plot.add_mesh(high_res_surface, color='lightblue')
    plot.show()


if __name__ == '__main__':
    main()

