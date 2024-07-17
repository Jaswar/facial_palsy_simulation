import numpy as np
from tetmesh import Tetmesh
from scipy.spatial import KDTree
import pyvista as pv


def main():
    tetmesh_path = 'data/tetmesh'
    actuations_in_path = 'data/actuations_017.npy'
    actuations_out_path = 'data/actuations_017_converted.npy'

    nodes, elements, _ = Tetmesh.read_tetgen_file(tetmesh_path)
    actuations_in = np.load(actuations_in_path)

    bary_coords = nodes[elements]
    bary_coords = np.mean(bary_coords, axis=1)

    kdtree = KDTree(nodes)
    _, indices = kdtree.query(bary_coords)

    actuations_out = np.zeros((elements.shape[0], actuations_in.shape[1], actuations_in.shape[2]), dtype=np.float32)
    for i, idx in enumerate(indices):
        actuations_out[i] = actuations_in[idx]
    
    cells = np.hstack([np.full((elements.shape[0], 1), 4, dtype=int), elements])
    celltypes = np.full(cells.shape[0], fill_value=pv.CellType.TETRA, dtype=int)
    tetmesh = pv.UnstructuredGrid(cells, celltypes, nodes)

    _, s, _ = np.linalg.svd(actuations_out)
    s = np.sum(s, axis=1)
    tetmesh['actuations'] = s

    plot = pv.Plotter()
    plot.add_mesh(tetmesh, scalars='actuations', clim=(2, 4), cmap='RdBu')
    plot.show()

    np.save(actuations_out_path, actuations_out)

if __name__ == '__main__':
    main()
