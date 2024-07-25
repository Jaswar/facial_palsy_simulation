import pyvista as pv
import os
from tetmesh import Tetmesh
import numpy as np


def main():
    nodes_original, elements, _ = Tetmesh.read_tetgen_file('data/tetmesh')
    root = os.path.join(
        os.path.abspath(''),
        '../FP-FEM/face_simulation/results_sim_2/017/random_search'
    )
    for file in os.listdir(root):
        if not file.startswith('deformed_nodes'):
            continue
        print(f'Showing {file}')
        nodes_deformed = np.load(os.path.join(root, file))

        cells = np.hstack([np.full((elements.shape[0], 1), 4, dtype=int), elements])
        celltypes = np.full(cells.shape[0], fill_value=pv.CellType.TETRA, dtype=int)
        tetmesh_original = pv.UnstructuredGrid(cells, celltypes, nodes_original)
        tetmesh_deformed = pv.UnstructuredGrid(cells, celltypes, nodes_deformed)

        plot = pv.Plotter(shape=(1,2))
        plot.subplot(0, 0)
        plot.add_mesh(tetmesh_original, color='yellow')
        plot.subplot(0, 1)
        plot.add_mesh(tetmesh_deformed, color='yellow')
        plot.link_views()
        plot.show()

if __name__ == '__main__':
    main()  

