import torch as th
import numpy as np
import pyvista as pv
from models import SimulatorModel
import igl
import argparse
import json


def main(args):
    if th.cuda.is_available():
        device = 'cuda'
    else:
        device = 'cpu'
    print(f'Using device: {device}')

    surface = pv.PolyData(args.neutral_path)
    print('Loading high res surface')
    high_res_surface = pv.PolyData(args.high_res_path)

    neutral_projection = surface.project_points_to_plane()
    high_res_projection = high_res_surface.project_points_to_plane()
    neutral_projection.points[:, 2] = 0.
    high_res_projection.points[:, 2] = 0.

    print('Removing points')
    dp, _, _ = igl.point_mesh_squared_distance(high_res_projection.points, neutral_projection.points, neutral_projection.regular_faces)
    d3d, _, _ = igl.point_mesh_squared_distance(high_res_surface.points, surface.points, surface.regular_faces)
    in_bounds = np.logical_and(dp < 1e-3, d3d < args.tol_3d)
    high_res_surface, _ = high_res_surface.remove_points(~in_bounds)
    print('Points removed')

    maxv = np.max(surface.points)
    minv = np.min(surface.points)

    high_res_surface.points = (high_res_surface.points - minv) / (maxv - minv)
    surface.points = (surface.points - minv) / (maxv - minv)

    with open(args.config_path, 'r') as f:
        config = json.load(f)

    model = SimulatorModel(num_hidden_layers=config['num_hidden_layers'], 
                     hidden_size=config['hidden_size'], 
                     fourier_features=config['fourier_features'])
    model = th.compile(model)
    model.load_state_dict(th.load(args.model_path))
    model.to(device)
    
    inputs = th.tensor(high_res_surface.points).float().to(device)
    outputs = model.predict(inputs).cpu().numpy()
    outputs = outputs * (maxv - minv) + minv
    high_res_surface.points = outputs    

    plot = pv.Plotter()
    plot.add_mesh(high_res_surface, color='lightblue')
    plot.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--neutral_path', type=str, required=True)
    parser.add_argument('--high_res_path', type=str, required=True)
    parser.add_argument('--model_path', type=str, required=True)
    parser.add_argument('--config_path', type=str, default='configs/config_simulation.json')
    parser.add_argument('--tol_3d', type=float, default=20.0)
    args = parser.parse_args()
    main(args)

