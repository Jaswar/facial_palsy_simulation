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
    high_res_surface = pv.read('../medusa_scans/rawMeshes/take_001.obj')

    maxv = np.max(surface.points)
    minv = np.min(surface.points)

    high_res_surface.points = (high_res_surface.points - minv) / (maxv - minv)
    surface.points = (surface.points - minv) / (maxv - minv)

    model = SimulatorModel(num_hidden_layers=9, hidden_size=64, fourier_features=8)
    model = th.compile(model)
    model.load_state_dict(th.load('checkpoints/best_model_simulator_001_fast_with_jaw.pth'))
    model.to(device)
    
    d, _, _ = igl.point_mesh_squared_distance(high_res_surface.points, surface.points, surface.regular_faces)
    in_bounds = d < 0.001

    inputs = th.tensor(high_res_surface.points[in_bounds]).float().to(device)
    outputs = model.predict(inputs).cpu().numpy()
    high_res_surface.points[in_bounds] = outputs
    print('Removing points')
    high_res_surface, _ = high_res_surface.remove_points(~in_bounds)

    plot = pv.Plotter()
    plot.add_mesh(high_res_surface, color='lightblue')
    plot.show()


if __name__ == '__main__':
    main()

