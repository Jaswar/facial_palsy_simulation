import torch as th
import numpy as np
from tetmesh import Tetmesh
import pyvista as pv
from models import SimulatorModel
from scipy.spatial import KDTree


def main():
    if th.cuda.is_available():
        device = 'cuda'
    else:
        device = 'cpu'
    print(f'Using device: {device}')

    nodes, _, _ = Tetmesh.read_tetgen_file('data/tetmesh')
    high_res_surface = pv.read('../medusa_scans/rawMeshes/take_001.obj')
    high_res_surface = high_res_surface.clean()

    maxv = np.max(nodes)
    minv = np.min(nodes)

    high_res_surface.points = (high_res_surface.points - minv) / (maxv - minv)
    nodes = (nodes - minv) / (maxv - minv)

    model = SimulatorModel(num_hidden_layers=9, hidden_size=64, fourier_features=8)
    model = th.compile(model)
    model.load_state_dict(th.load('checkpoints/best_model_simulator_001_fast_with_jaw.pth'))
    model.to(device)

    kdtree = KDTree(nodes)
    d, _ = kdtree.query(high_res_surface.points)
    in_bounds = d < 0.02

    inputs = th.tensor(high_res_surface.points[in_bounds]).float().to(device)
    outputs = model.predict(inputs).cpu().numpy()
    high_res_surface.points[in_bounds] = outputs

    plot = pv.Plotter()
    plot.add_mesh(high_res_surface)
    plot.show()


if __name__ == '__main__':
    main()

