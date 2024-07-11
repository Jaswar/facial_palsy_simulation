from tetmesh import Tetmesh
import pyvista as pv
import igl
import numpy as np

tetmesh_path = 'data/tetmesh'
jaw_path = 'data/jaw.obj'
skull_path = 'data/skull.obj'

nodes, elements, _ = Tetmesh.read_tetgen_file(tetmesh_path)

jaw = pv.read(jaw_path)
jaw = jaw.clean()
jaw = pv.PolyData(jaw)

skull = pv.read(skull_path)
skull = skull.clean()
skull = pv.PolyData(skull)

tol = 1.0

ds, _, _ = igl.point_mesh_squared_distance(nodes, skull.points, skull.regular_faces)
skull_boundary_mask = ds < tol

ds, _, _ = igl.point_mesh_squared_distance(nodes, jaw.points, jaw.regular_faces)
jaw_boundary_mask = ds < tol

cells = np.hstack([np.full((elements.shape[0], 1), 4, dtype=int), elements])
celltypes = np.full(cells.shape[0], fill_value=pv.CellType.TETRA, dtype=int)
grid = pv.UnstructuredGrid(cells, celltypes, nodes)

skull_nodes = nodes[skull_boundary_mask]
jaw_nodes = nodes[jaw_boundary_mask]
plot = pv.Plotter()
plot.add_mesh(grid, color='lightgray')
plot.add_points(skull_nodes, color='midnightblue', point_size=7.)
plot.add_points(jaw_nodes, color='red', point_size=7.)
plot.show()
