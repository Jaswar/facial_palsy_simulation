"""
author: daniel dorda
"""

import pyvista as pv
import numpy as np
from scipy.spatial import KDTree

from tetmesh import Tetmesh

dataroot = "data"
tetmesh_contour_fn = f"{dataroot}/tetmesh_contour.obj"
tetmesh_base_fn = f"{dataroot}/tetmesh"
tetmesh_reflected_deformed_fn = f"{dataroot}/tetmesh_contour_ref_deformed.obj"

act_fn = f'{dataroot}/actuations_017.npy'
act_out_fn = f'{dataroot}/act_sym.npy'

# X is the symmetry axis.
surf = pv.PolyData(tetmesh_contour_fn)

x_plane = pv.Plane(direction=(1.0, 0.0, 0.0), i_size=130, j_size=300)
x_plane.points[:, 2] += 15
x_plane.points[:, 0] -= 21

surf_ref = surf.copy(deep=True)
mean_pts = np.mean(surf_ref.points, axis=0)
surf_ref.points -= mean_pts
surf_ref.points[:, 0] *= -1
surf_ref.points += mean_pts
surf_ref.flip_normals()

# show the flipped surface and write it out. I did this at the start
# p = pv.Plotter()
# p.add_mesh(surf)
# p.add_mesh(surf_ref, color='blue', opacity=0.5)
# p.add_mesh(x_plane, color='red', opacity=0.5)
# p.show()
# igl.write_obj("C:/Users/ddorda/USZ-SIM/tetmesh/tetmesh_contour.obj", surf.points, surf.faces.reshape((-1, 4))[:, 1:])
# igl.write_obj("C:/Users/ddorda/USZ-SIM/tetmesh/tetmesh_contour_ref.obj", surf_ref.points, surf_ref.faces.reshape((-1, 4))[:, 1:])

deformed_surf = pv.PolyData(tetmesh_reflected_deformed_fn)
kdt = KDTree(surf.points)


def bary_transform(points):
    if points.ndim == 1:
        points = points[None, :]
    n_points = points.shape[0]
    # find NNs on undeformed mesh
    d, idcs = kdt.query(points, k=5)
    old_nns = surf.points[idcs]
    # look up positions on deformed, flipped mesh (implicit mapping)
    new_nns = deformed_surf.points[idcs]
    # find the "offset" from the query point to the NNs and flip the x
    offset = old_nns - points[:, None, :]
    offset[:, :, 0] *= -1

    # find the possible "new" positions and calculate weighted average
    new_pts = new_nns - offset
    d = d[:, :, None]  # n_points, k, 1
    new_pos = np.average(new_pts, weights=1.0/(d+1e-6).repeat(3, axis=2), axis=1)
    return new_pos


## Test the mapping of random query points. Seems robust enough.
p = np.random.permutation(surf.points.shape[0])
n_points_queried = 3

query_point = surf.points[p[:n_points_queried]] + np.random.randn(n_points_queried, 3) * 5
new_pt = bary_transform(query_point)
# p = pv.Plotter()
# p.add_mesh(surf, opacity=0.5)
# p.add_points(query_point, color='blue')
# p.add_points(new_pt, color='red')
# p.add_text('blue: old point/nred: new point')
# p.show()


## Open the tetmesh!
nodes, eles, _ = Tetmesh.read_tetgen_file(tetmesh_base_fn, require_face_file=False)
pv_eles = np.hstack([np.full((eles.shape[0], 1), 4, dtype=int), eles])
celltypes = np.full(pv_eles.shape[0], fill_value=10, dtype=int)
grid = pv.UnstructuredGrid(pv_eles, celltypes, nodes)

ele_pts = nodes[eles]
barries = ele_pts.mean(axis=1)
flipped_points = barries[:, 0] > -21
new_pt = bary_transform(barries)

# again for testing mapping on elements!
# p = pv.Plotter()
# p.add_mesh(surf, opacity=0.5)
# p.add_points(barries[flipped_points], color='blue')
# p.add_points(new_pt[flipped_points], color='red')
# p.add_text('blue: old point/nred: new point')
# p.show()

kdt_ele = KDTree(barries)
d, mapped_element = kdt_ele.query(new_pt[flipped_points])
print('Done!')

element_values = (barries[:, 0] + 21) / np.max(np.abs(barries[:, 0] + 21))  ## Ersatz for actuations - one per element
grid['values'] = element_values

element_values_sym = (barries[:, 0] + 21) / np.max(np.abs(barries[:, 0] + 21))  ## Ersatz for actuations - one per element
element_values_sym[flipped_points] = element_values_sym[mapped_element]

# grid['values symmetric'] = element_values_sym
# print(grid)
# p = pv.Plotter(shape=(1,2))
# p.subplot(0, 0)
# p.add_mesh(grid.copy(), scalars='values', clim=(-1, 1), cmap='magma')
# p.subplot(0, 1)
# p.add_mesh(grid, scalars='values symmetric', clim=(-1, 1), cmap='magma')
# p.add_mesh(x_plane, color='green', opacity=0.5)
# p.link_views()
# p.show()

# Real actuations
act = np.load(act_fn)
act_sym = act.copy()
act_sym[flipped_points] = act[mapped_element]
assert not np.allclose(act, act_sym)

grid['Trace(A)'] = np.trace(act, axis1=1, axis2=2)
grid['Trace(A) symmetric'] = np.trace(act_sym, axis1=1, axis2=2)

p = pv.Plotter(shape=(1,2))
p.subplot(0, 0)
p.add_mesh(grid.copy(), scalars='Trace(A)', clim=(2, 4), cmap='RdBu')
p.subplot(0, 1)
p.add_mesh(grid, scalars='Trace(A) symmetric', clim=(2, 4), cmap='RdBu')
p.add_mesh(x_plane, color='green', opacity=0.2)
p.link_views()
p.show()
print('Done! Saving...')

np.save(act_out_fn, act_sym)
print('Saved!')