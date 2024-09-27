import torch as th
import numpy as np
import pyvista as pv
from models import SimulatorModel, INRModel
import igl
import argparse
import json
from obj_parser import ObjParser
from common import detect_components
import os
os.environ['DISPLAY'] = ':99'

def visualize(surface, plot, index, camera_position, rgb):
    surface.points[:, 0] -= camera_position[0]
    surface.points[:, 1] -= camera_position[1]
    camera_position = [0, 0, camera_position[2]]
    plot.subplot(0, index)
    plot.view_xy()
    plot.set_viewup([0, -1, 0])
    plot.set_position(camera_position)
    if rgb:
        plot.add_mesh(surface, scalars='RGB', rgb=True)
    else:
        plot.add_mesh(surface, color='lightblue')


def get_model(args, model_type):
    if model_type == 'simulator':
        with open(args.simulator_config_path, 'r') as f:
            config = json.load(f)

        model = SimulatorModel(num_hidden_layers=config['num_hidden_layers'], 
                        hidden_size=config['hidden_size'], 
                        fourier_features=config['fourier_features'])
        model = th.compile(model)
        model.load_state_dict(th.load(args.simulator_model_path))
    else:
        with open(args.inr_config_path, 'r') as f:
            config = json.load(f)

        model = INRModel(num_hidden_layers=config['num_hidden_layers'], 
                        hidden_size=config['hidden_size'], 
                        fourier_features=config['fourier_features'])
        model = th.compile(model)
        model.load_state_dict(th.load(args.inr_model_path))
    return model


def predict_high_res(args, device, plot, index, camera_position, rgb, model_type):
    surface = pv.PolyData(args.neutral_path)
    print('Loading high res surface')
    # the color information will be stored in the 'RGB' array
    high_res_surface = pv.PolyData(args.high_res_path)

    neutral_projection = surface.copy()
    high_res_projection = high_res_surface.copy()
    neutral_projection.points[:, 2] = 0.
    high_res_projection.points[:, 2] = 0.

    edges = neutral_projection.extract_feature_edges(boundary_edges=True, 
                                                     non_manifold_edges=False, 
                                                     manifold_edges=False, 
                                                     feature_edges=False)
    # the format of lines is (n, a, b) where n is the number of points in the line and a, b are the indices of the points
    # they are all stacked into a single array, so we need to seperate them
    lines = [(edges.lines[i + 1], edges.lines[i + 2]) for i in range(0, len(edges.lines), 3)]
    components = detect_components(lines)
    assert len(components) == 2, f'Expected 2 components (outline and mouth), found {len(components)}'
    mouth_component = components[1]  # 0th component is the outer boundary
    # detect the bounding box of the mouth component
    max_x = np.max(edges.points[mouth_component, 0])
    min_x = np.min(edges.points[mouth_component, 0])
    max_y = np.max(edges.points[mouth_component, 1])
    min_y = np.min(edges.points[mouth_component, 1])

    mouth_roi = np.logical_and(high_res_projection.points[:, 0] >= min_x, high_res_projection.points[:, 0] <= max_x)
    mouth_roi = np.logical_and(mouth_roi, high_res_projection.points[:, 1] >= min_y)
    mouth_roi = np.logical_and(mouth_roi, high_res_projection.points[:, 1] <= max_y)

    print('Removing points')    
    dp, _, _ = igl.point_mesh_squared_distance(high_res_projection.points, neutral_projection.points, neutral_projection.regular_faces)
    # only points in the mouth area that are too far away should be removed
    proj_in_bounds = np.logical_or(dp < 1e-3, ~mouth_roi)
    d3d, _, _ = igl.point_mesh_squared_distance(high_res_surface.points, surface.points, surface.regular_faces)
    in_bounds = np.logical_and(proj_in_bounds, d3d < args.tol_3d)
    high_res_surface, _ = high_res_surface.remove_points(~in_bounds)
    print('Points removed')

    maxv = np.max(surface.points)
    minv = np.min(surface.points)

    high_res_surface.points = (high_res_surface.points - minv) / (maxv - minv)
    surface.points = (surface.points - minv) / (maxv - minv)

    model = get_model(args, model_type)    
    model.to(device)
    
    inputs = th.tensor(high_res_surface.points).float().to(device)
    outputs = model.predict(inputs).cpu().numpy()
    outputs = outputs * (maxv - minv) + minv
    high_res_surface.points = outputs    

    visualize(high_res_surface, plot, index, camera_position, rgb)


def predict_low_res(args, device, plot, index, camera_position, rgb, model_type):
    surface = pv.PolyData(args.neutral_path)

    maxv = np.max(surface.points)
    minv = np.min(surface.points)
    surface.points = (surface.points - minv) / (maxv - minv)

    model = get_model(args, model_type)    
    model.to(device)
    
    inputs = th.tensor(surface.points).float().to(device)
    outputs = model.predict(inputs).cpu().numpy()
    outputs = outputs * (maxv - minv) + minv
    surface.points = outputs
    
    visualize(surface, plot, index, camera_position, rgb)    


def visualize_original_high_res(arguments, plot, index, camera_position, rgb):
    surface = pv.PolyData(arguments.original_high_res_path)
    visualize(surface, plot, index, camera_position, rgb)


def main(args):
    if th.cuda.is_available():
        device = 'cuda'
    else:
        device = 'cpu'
    print(f'Using device: {device}')

    neutral_surface = pv.PolyData(args.neutral_path)
    x_coord = np.mean(neutral_surface.points[:, 0])
    y_coord = np.mean(neutral_surface.points[:, 1])
    z_coord = np.max(neutral_surface.points[:, 2]) + 400
    camera_position = [x_coord, y_coord, z_coord]

    plot = pv.Plotter(off_screen=True, shape=(1, 6))
    
    visualize_original_high_res(args, plot, 0, camera_position, True)
    predict_low_res(args, device, plot, 1, camera_position, False, 'inr')
    predict_high_res(args, device, plot, 2, camera_position, True, 'inr')
    predict_low_res(args, device, plot, 3, camera_position, False, 'simulator')
    predict_high_res(args, device, plot, 4, camera_position, True, 'simulator')
    predict_high_res(args, device, plot, 5, camera_position, False, 'simulator')

    plot.screenshot(args.save_path, window_size=(500 * 6, 800))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--neutral_path', type=str, required=True)
    parser.add_argument('--high_res_path', type=str, required=True)
    parser.add_argument('--original_high_res_path', type=str, required=True)
    parser.add_argument('--simulator_model_path', type=str, required=True)
    parser.add_argument('--simulator_config_path', type=str, default='configs/config_simulation.json')
    parser.add_argument('--inr_model_path', type=str, required=True)
    parser.add_argument('--inr_config_path', type=str, default='configs/config_simulation.json')
    parser.add_argument('--save_path', type=str, required=True)
    parser.add_argument('--tol_3d', type=float, default=20.0)
    args = parser.parse_args()
    main(args)

