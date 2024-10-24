import torch as th
import numpy as np
import pyvista as pv
from models import SimulatorModel, INRModel
from actuation_predictor import ActuationPredictor
import igl
import argparse
import json
from obj_parser import ObjParser
from common import detect_components
from tetmesh import Tetmesh
import os
# os.system('/usr/bin/Xvfb :99 -screen 0 1024x768x24 &')
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


def visualize_tetmesh(points, elements, plot, index, camera_position):
    points[:, 0] -= camera_position[0]
    points[:, 1] -= camera_position[1]
    camera_position = [0, 0, camera_position[2]]
    plot.subplot(0, index)
    plot.view_xy()
    plot.set_viewup([0, -1, 0])
    plot.set_position(camera_position)

    cells = np.hstack([np.full((elements.shape[0], 1), 4, dtype=int), elements])
    celltypes = np.full(cells.shape[0], fill_value=pv.CellType.TETRA, dtype=int)
    neutral_grid = pv.UnstructuredGrid(cells, celltypes, points)
    plot.add_mesh(neutral_grid, color='lightblue')


def get_model(args, model_type):
    if model_type == 'simulator':
        with open(args.simulator_config_path, 'r') as f:
            config = json.load(f)

        model = SimulatorModel(num_hidden_layers=config['num_hidden_layers'], 
                        hidden_size=config['hidden_size'], 
                        fourier_features=config['fourier_features'])
        model = th.compile(model)
        model.load_state_dict(th.load(args.simulator_model_path))
    elif model_type == 'inr_healthy':
        with open(args.inr_config_path, 'r') as f:
            config = json.load(f)

        model = INRModel(num_hidden_layers=config['num_hidden_layers'], 
                        hidden_size=config['hidden_size'], 
                        fourier_features=config['fourier_features'])
        model = th.compile(model)
        model.load_state_dict(th.load(args.healthy_inr_model_path))
    elif model_type == 'inr_unhealthy':
        with open(args.inr_config_path, 'r') as f:
            config = json.load(f)

        model = INRModel(num_hidden_layers=config['num_hidden_layers'], 
                        hidden_size=config['hidden_size'], 
                        fourier_features=config['fourier_features'])
        model = th.compile(model)
        model.load_state_dict(th.load(args.unhealthy_inr_model_path))
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
    in_bounds = np.logical_and(proj_in_bounds, d3d < args.tol_3d) if not args.close_mouth else d3d < args.tol_3d
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


def predict_low_res_tetmesh(args, device, plot, index, camera_position, model_type):
    points, elements, _ = Tetmesh.read_tetgen_file(args.tetmesh_path)

    maxv = np.max(points)
    minv = np.min(points)
    points = (points - minv) / (maxv - minv)

    model = get_model(args, model_type)    
    model.to(device)
    
    inputs = th.tensor(points).float().to(device)
    outputs = model.predict(inputs).cpu().numpy()
    outputs = outputs * (maxv - minv) + minv
    
    visualize_tetmesh(outputs, elements, plot, index, camera_position)  


def predict_actuations(args, device, plot, index, camera_position):
    predictor = ActuationPredictor(args.healthy_inr_model_path, 
                                             args.inr_config_path, 
                                             args.tetmesh_path, 
                                             args.contour_path, 
                                             args.reflected_contour_path, 
                                             secondary_model_path=args.unhealthy_inr_model_path, 
                                             device=device)
    A, _ = predictor.predict(th.tensor(predictor.nodes).float().to(device))
    _, s, _ = th.svd(A)
    actuations = th.sum(s, dim=1).cpu().numpy()

    points = predictor.nodes
    points[:, 0] -= camera_position[0]
    points[:, 1] -= camera_position[1]
    camera_position = [0, 0, camera_position[2]]

    cells = np.hstack([np.full((predictor.elements.shape[0], 1), 4, dtype=int), predictor.elements])
    celltypes = np.full(cells.shape[0], fill_value=pv.CellType.TETRA, dtype=int)
    neutral_grid = pv.UnstructuredGrid(cells, celltypes, points)
    neutral_grid['actuations'] = actuations

    plot.subplot(0, index)
    plot.view_xy()
    plot.set_viewup([0, -1, 0])
    plot.set_position(camera_position)
    plot.add_mesh(neutral_grid.copy(), scalars='actuations', clim=(2, 4), cmap='RdBu', show_scalar_bar=False)


def visualize_flame_model(model_path, plot, index, camera_position):
    surface = pv.PolyData(model_path)
    surface.points *= 1000  # convert to the correct scale (mm)
    visualize(surface, plot, index, camera_position, False)


def visualize_mirrored_expression(neutral_surface_path, expression_surface_path, plot, index, camera_position):
    neutral_surface = pv.PolyData(neutral_surface_path)
    expression_surface = pv.PolyData(expression_surface_path)
    midpoint = np.mean(neutral_surface.points[:, 0])
    to_remove = neutral_surface.points[:, 0] > midpoint
    expression_surface, _ = expression_surface.remove_points(to_remove)

    pv_vertices, pv_faces = expression_surface.points, expression_surface.faces.reshape(-1, 4)[:, 1:]
    new_pv_vertices = np.vstack([pv_vertices, pv_vertices])
    new_pv_vertices[pv_vertices.shape[0]:, 0] = 2 * midpoint - new_pv_vertices[pv_vertices.shape[0]:, 0]
    new_pv_faces = np.vstack([pv_faces, pv_faces + pv_vertices.shape[0]])
    new_pv_scan = pv.PolyData(new_pv_vertices, np.hstack([np.full((new_pv_faces.shape[0], 1), 3), new_pv_faces]))

    visualize(new_pv_scan, plot, index, camera_position, False)


def visualize_mesh(mesh_path, plot, index, camera_position, rgb):
    surface = pv.PolyData(mesh_path)
    visualize(surface, plot, index, camera_position, rgb)

# for FLAME:
# 1. original high-res
# 2. registered low-res
# 3. flipped mesh
# 4. fitted FLAME model
# 5. predicted low-res 
# 6. predicted high-res

# for actuations:
# 1. original high-res
# 2. registered low-res
# 3. predicted low-res healthy (INR)
# 4. predicted high-res healthy (INR) - optional
# 5. predicted low-res unhealthy (INR)
# 6. predicted high-res unheatlhy (INR) - optional
# 7. predicted actuations
# 8. predicted simulator low-res
# 9. predicted simulator high-res
# 10. predicted simulator high-res (no rgb) - optional
def main(args):
    if th.cuda.is_available():
        device = 'cuda'
    else:
        device = 'cpu'
    print(f'Using device: {device}')

    # pv.global_theme.transparent_background = True

    neutral_surface = pv.PolyData(args.neutral_path)
    x_coord = np.mean(neutral_surface.points[:, 0])
    y_coord = np.mean(neutral_surface.points[:, 1])
    z_coord = np.max(neutral_surface.points[:, 2]) + 400
    camera_position = [x_coord, y_coord, z_coord]

    plot = pv.Plotter(off_screen=True, shape=(1, 1))
    
    # approach 1 result plot:
    # visualize_mesh(args.original_high_res_path, plot, 0, camera_position, True)
    # visualize_mesh(args.original_low_res_path, plot, 1, camera_position, False)
    # visualize_mirrored_expression(args.neutral_path, args.original_low_res_path, plot, 2, camera_position)
    # visualize_flame_model(args.flame_path, plot, 3, camera_position)
    # predict_low_res(args, device, plot, 4, camera_position, False, 'inr_healthy')
    # predict_high_res(args, device, plot, 5, camera_position, True, 'inr_healthy')


    # approach 2 result plots:
    # visualize_mesh(args.original_high_res_path, plot, 0, camera_position, True)
    # visualize_mesh(args.original_low_res_path, plot, 1, camera_position, False)
    # predict_low_res_tetmesh(args, device, plot, 2, camera_position, 'inr_healthy')
    # predict_low_res_tetmesh(args, device, plot, 3, camera_position, 'inr_unhealthy')
    # predict_actuations(args, device, plot, 4, camera_position)
    # predict_low_res_tetmesh(args, device, plot, 5, camera_position, 'simulator')
    # predict_high_res(args, device, plot, 6, camera_position, True, 'simulator')

    # approach 2 method plot (one at a time):
    # visualize_mesh(args.original_high_res_path, plot, 0, camera_position, True)
    # visualize_mesh(args.original_low_res_path, plot, 0, camera_position, False)
    # predict_low_res_tetmesh(args, device, plot, 0, camera_position, 'inr_healthy')
    # predict_low_res_tetmesh(args, device, plot, 0, camera_position, 'inr_unhealthy')
    # predict_actuations(args, device, plot, 0, camera_position)
    # predict_low_res_tetmesh(args, device, plot, 0, camera_position, 'simulator')
    # predict_high_res(args, device, plot, 0, camera_position, False, 'simulator')

    # approach 1 method plot (one at a time):
    # visualize_mesh(args.original_high_res_path, plot, 0, camera_position, False)
    # visualize_mirrored_expression(args.neutral_path, args.original_low_res_path, plot, 0, camera_position)
    # visualize_flame_model(args.flame_path, plot, 0, camera_position)
    # predict_low_res(args, device, plot, 0, camera_position, False, 'inr_healthy')
    # predict_high_res(args, device, plot, 0, camera_position, False, 'inr_healthy')

    # appendix plots
    # visualize_mesh(args.original_high_res_path, plot, 0, camera_position, True)

    # new preprocessing
    # visualize_mesh(args.original_high_res_path, plot, 0, camera_position, False)
    # visualize_mesh(args.original_low_res_path, plot, 1, camera_position, False)

    # new approach 1
    # visualize_mirrored_expression(args.neutral_path, args.original_low_res_path, plot, 0, camera_position)
    # visualize_flame_model(args.flame_path, plot, 1, camera_position)
    # predict_low_res(args, device, plot, 2, camera_position, False, 'inr_healthy')
    # predict_high_res(args, device, plot, 3, camera_position, False, 'inr_healthy')

    # new approach 2
    # predict_low_res_tetmesh(args, device, plot, 0, camera_position, 'inr_healthy')
    # predict_low_res_tetmesh(args, device, plot, 1, camera_position, 'inr_unhealthy')
    # predict_actuations(args, device, plot, 2, camera_position)
    # predict_low_res_tetmesh(args, device, plot, 3, camera_position, 'simulator')
    # predict_high_res(args, device, plot, 4, camera_position, False, 'simulator')

    # plot.screenshot(args.save_path, window_size=(500 * 1, 800))

    # original expressions
    folder = os.path.dirname(args.original_high_res_path)
    for file in os.listdir(folder):
        if not file.endswith('.obj'):
            continue

        print(f'Processing {file}')
        plot = pv.Plotter(off_screen=True, shape=(1, 1))
        index = ''.join(c for c in file if c.isdigit())
        visualize_mesh(os.path.join(folder, file), plot, 0, camera_position, False)
        plot.screenshot(os.path.join(args.save_path, f'take_{index}.jpg'), window_size=(500 * 1, 800))
        plot.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--neutral_path', type=str, required=True)
    parser.add_argument('--high_res_path', type=str, required=True)

    parser.add_argument('--flame_path', type=str, required=True)

    parser.add_argument('--tetmesh_path', type=str, required=True)
    parser.add_argument('--contour_path', type=str, required=True)
    parser.add_argument('--reflected_contour_path', type=str, required=True)

    parser.add_argument('--original_high_res_path', type=str, required=True)
    parser.add_argument('--original_low_res_path', type=str, required=True)
    parser.add_argument('--simulator_model_path', type=str, required=True)
    parser.add_argument('--simulator_config_path', type=str, default='configs/config_simulation.json')
    parser.add_argument('--healthy_inr_model_path', type=str, required=True)
    parser.add_argument('--unhealthy_inr_model_path', type=str, required=False)
    parser.add_argument('--inr_config_path', type=str, default='configs/config_inr.json')
    parser.add_argument('--save_path', type=str, required=True)
    parser.add_argument('--tol_3d', type=float, default=20.0)
    parser.add_argument('--close_mouth', action='store_true')
    args = parser.parse_args()
    main(args)

