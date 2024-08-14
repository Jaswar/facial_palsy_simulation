import torch as th
import numpy as np
import pyvista as pv
from models import SimulatorModel
import igl
import argparse
import json
from obj_parser import ObjParser


def detect_components(lines):
    graph = {}
    for (a, b) in lines:
        if a not in graph:
            graph[a] = []
        if b not in graph:
            graph[b] = []
        graph[a].append(b)
        graph[b].append(a)

    components = []
    visited = set()
    for node in graph:
        if node in visited:
            continue
        component = []
        stack = [node]
        while len(stack) > 0:
            node = stack.pop()
            if node in visited:
                continue
            visited.add(node)
            component.append(node)
            stack.extend(graph[node])
        components.append(component)
    components = sorted(components, key=lambda x: len(x), reverse=True)
    return components


def main(args):
    if th.cuda.is_available():
        device = 'cuda'
    else:
        device = 'cpu'
    print(f'Using device: {device}')

    parser = ObjParser()  # needed for the MRGB values
    _, _, _, _, rgb_values = parser.parse(args.high_res_path, progress_bar=True, mrgb_only=True)

    surface = pv.PolyData(args.neutral_path)
    print('Loading high res surface')
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
    rgb_values = rgb_values[in_bounds]
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
    high_res_surface['rgb'] = rgb_values
    plot.add_mesh(high_res_surface, scalars='rgb', rgb=True)
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

