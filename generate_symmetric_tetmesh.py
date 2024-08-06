import numpy as np
import pyvista as pv
from scipy.spatial import KDTree
from tetmesh import Tetmesh


def bary_transform(points, surface, deformed_surface, kdtree):
    if points.ndim == 1:
        points = points[None, :]
    n_points = points.shape[0]
    # find NNs on undeformed mesh
    d, idcs = kdtree.query(points, k=5)
    old_nns = surface.points[idcs]
    # look up positions on deformed, flipped mesh (implicit mapping)
    new_nns = deformed_surface.points[idcs]
    # find the "offset" from the query point to the NNs and flip the x
    offset = old_nns - points[:, None, :]
    offset[:, :, 0] *= -1

    # find the possible "new" positions and calculate weighted average
    new_pts = new_nns - offset
    d = d[:, :, None]  # n_points, k, 1
    new_pos = np.average(new_pts, weights=1.0/(d+1e-6).repeat(3, axis=2), axis=1)
    return new_pos


def main():
    tetmesh_path = 'data/tetmesh'
    neutral_surface_path = 'data/tetmesh_face_surface.obj'
    deformed_surface_path = 'data/ground_truths/deformed_surface_017.obj'
    contour_path = 'data/tetmesh_contour.obj'
    reflected_contour_path = 'data/tetmesh_contour_ref_deformed.obj'
    deformed_out_path = 'data/symmetric_tetmesh'

    nodes, elements, _ = Tetmesh.read_tetgen_file(tetmesh_path)
    neutral_surface = pv.PolyData(neutral_surface_path)
    deformed_surface = pv.PolyData(deformed_surface_path)
    contour = pv.PolyData(contour_path)
    reflected_contour = pv.PolyData(reflected_contour_path)
    
    kdtree = KDTree(contour.points)
    midpoint = np.mean(nodes[:, 0])
    flipped_indices = nodes[:, 0] > midpoint
    new_nodes = bary_transform(nodes, contour, reflected_contour, kdtree)
    new_nodes[:, 0] = 2 * midpoint - new_nodes[:, 0]
    nodes[flipped_indices] = new_nodes[flipped_indices]

    cells = np.hstack([np.full((elements.shape[0], 1), 4, dtype=int), elements])
    celltypes = np.full(cells.shape[0], fill_value=pv.CellType.TETRA, dtype=int)
    grid = pv.UnstructuredGrid(cells, celltypes, nodes)

    plot = pv.Plotter()
    plot.add_mesh(grid, color='lightblue')
    plot.link_views()
    plot.show()

    pv.save_meshio(deformed_out_path + '.node', grid)
    pv.save_meshio(deformed_out_path + '.ele', grid)


if __name__ == '__main__':
    main()