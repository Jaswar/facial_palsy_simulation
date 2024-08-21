import numpy as np
import pyvista as pv
from tetmesh import Tetmesh
from scipy.spatial import KDTree
from simpleicp import PointCloud, SimpleICP


def get_triangle_area(vertices, faces):
    # Calculate the area of each triangle
    v0 = vertices[faces[:, 0]]
    v1 = vertices[faces[:, 1]]
    v2 = vertices[faces[:, 2]]
    e0 = v1 - v0
    e1 = v2 - v0
    cross = np.cross(e0, e1)
    area = 0.5 * np.linalg.norm(cross, axis=1)
    return area


def get_probabilities(vertices, faces):
    areas = get_triangle_area(vertices, faces)
    probabilities = areas / np.sum(areas)
    return probabilities


# uniform sampling on a simplex, see for example: http://blog.geomblog.org/2005/10/sampling-from-simplex.html
# this gives barycentric coordinates, which can be used to sample points on a triangle
def sample_on_simplex(n_samples):
    samples = np.random.rand(n_samples, 3)
    samples = -np.log(samples)
    samples = samples / np.sum(samples, axis=1)[:, None]
    return samples


def barycentric_sample(vertices, coords):
    v0 = vertices[:, 0]
    v1 = vertices[:, 1]
    v2 = vertices[:, 2]
    e0 = v1 - v0
    e1 = v2 - v0
    return v0 + coords[:, 0][:, None] * e0 + coords[:, 1][:, None] * e1


def sample_from_surface(vertices, faces, probabilities, n_samples):
    coords = sample_on_simplex(n_samples)
    face_indices = np.random.choice(faces.shape[0], n_samples, p=probabilities)
    face_vertices = vertices[faces[face_indices]]
    sampled = barycentric_sample(face_vertices, coords)
    return sampled, faces[face_indices], coords


def main():
    n_samples = 1000000
    neutral_surface = pv.PolyData('data/tetmesh_face_surface.obj')
    neutral_surface = neutral_surface.clean()
    deformed_surface = pv.PolyData('data/ground_truths/deformed_surface_003.obj')
    deformed_surface = deformed_surface.clean()
    neutral_high_res_surface = pv.PolyData('../medusa_scans/rawMeshes_ply/take_001.ply')
    deformed_high_res_surface = pv.PolyData('../medusa_scans/rawMeshes_ply/take_004.ply')
    print('Loaded surfaces')

    pc_fix = PointCloud(deformed_surface.points, columns=['x', 'y', 'z'])
    pc_mov = PointCloud(deformed_high_res_surface.points, columns=['x', 'y', 'z'])

    icp = SimpleICP()
    icp.add_point_clouds(pc_fix, pc_mov)
    _, deformed_high_res_surface.points, _, _ = icp.run()
    print('Aligned surfaces')

    probabilties = get_probabilities(neutral_surface.points, neutral_surface.regular_faces)    
    sampled_vertices, sampled_faces, bary_coords = sample_from_surface(neutral_surface.points, neutral_surface.regular_faces, probabilties, n_samples)

    neutral_kdtree = KDTree(neutral_high_res_surface.points)
    _, neutral_indices = neutral_kdtree.query(sampled_vertices)
    sampled_vertices = neutral_high_res_surface.points[neutral_indices]
    print('Sampled neutral points')

    deformed_vertices = deformed_surface.points[sampled_faces]
    deformed_vertices = barycentric_sample(deformed_vertices, bary_coords)
    
    deformed_kdtree = KDTree(deformed_high_res_surface.points)
    _, deformed_indices = deformed_kdtree.query(deformed_vertices)
    deformed_vertices = deformed_high_res_surface.points[deformed_indices]
    print('Sampled deformed points')

    sampled_vertices = pv.PolyData(sampled_vertices)
    sampled_vertices['RGB'] = neutral_high_res_surface['RGB'][neutral_indices]
    deformed_vertices = pv.PolyData(deformed_vertices)
    deformed_vertices['RGB'] = deformed_high_res_surface['RGB'][deformed_indices]

    plot = pv.Plotter(shape=(1, 2))
    plot.subplot(0, 0)
    plot.add_points(sampled_vertices, scalars='RGB', rgb=True)
    plot.subplot(0, 1)
    plot.add_points(deformed_vertices, scalars='RGB', rgb=True)
    plot.link_views()
    plot.show()


if __name__ == '__main__':
    main()

