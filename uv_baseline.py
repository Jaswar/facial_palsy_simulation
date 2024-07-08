from pathlib import Path
import matplotlib
import matplotlib.pyplot as plt
from collections import deque
import numpy as np
from scipy.spatial import KDTree
import pyvista as pv


# tried using libraries like pywavefront, but it seems like it doesn't support uv coordinates properly
class ObjParser(object):

    def __init__(self, path):
        self.path = path

    def parse(self):
        vertices = []
        faces = []
        normals = []
        uv_coords = []
        with open(self.path, 'r') as f:
            for line in f:
                if line.startswith('v '):
                    vertices.append(list(map(float, line.strip().split()[1:])))
                elif line.startswith('vt '):
                    uv_coords.append(list(map(float, line.strip().split()[1:])))
                elif line.startswith('vn '):
                    normals.append(list(map(float, line.strip().split()[1:])))
                elif line.startswith('f '):
                    faces.append(self.__parse_face(line))
        return np.array(vertices), np.array(faces), np.array(normals), np.array(uv_coords)


    def __parse_face(self, face_line):
        split = face_line.strip().split(' ')[1:]
        face = [[int(v) for v in vertex.split('/')] for vertex in split]
        return face


class MeshWithUV(object):

    def __init__(self, path):
        self.path = path
        self.parser = ObjParser(path)
        self.vertices, self.faces, self.normals, self.uv_coords = self.parser.parse()

        self.mid_point_3d = np.mean(self.vertices[:, 0])
        self.original_values_3d = np.array([v[0] for v in self.vertices]) - self.mid_point_3d  # to be replaced with displacement values
        self.original_values_3d = self.original_values_3d / np.max(np.abs(self.original_values_3d))
        
        self.mid_point_uv = 0.4  # hard coded for now, had a better idea (__extract_face_only) but didnt work :(
        self.kdtree = KDTree(self.uv_coords)

        self.transformed_values_3d = None
        self.original_values_uv = None
        self.transformed_values_uv = None

    def baseline_transform(self):
        self.mesh_to_uv()
        self.reflect()
        self.uv_to_mesh()

    def mesh_to_uv(self):
        self.original_values_uv = np.zeros(len(self.uv_coords))
        counts = np.zeros(len(self.uv_coords))
        for face in self.faces:
            for vertex in face:
                self.original_values_uv[vertex[1] - 1] += self.original_values_3d[vertex[0] - 1]
                counts[vertex[1] - 1] += 1
        self.original_values_uv /= counts

    def uv_to_mesh(self):
        self.transformed_values_3d = np.zeros(len(self.vertices))
        counts = np.zeros(len(self.vertices))
        for face in self.faces:
            for vertex in face:
                self.transformed_values_3d[vertex[0] - 1] += self.transformed_values_uv[vertex[1] - 1]
                counts[vertex[0] - 1] += 1
        self.transformed_values_3d /= counts

    def reflect(self):
        reflected_indices = np.where(self.uv_coords[:, 0] < self.mid_point_uv)
        reflected = self.uv_coords.copy()
        # imagine a point is at 0.3 and mid_point=0.4, then we want to reflect it to 0.5
        # hence the formula is 0.4 - 0.3 + 0.4 = 0.5
        reflected[reflected_indices, 0] = self.mid_point_uv - reflected[reflected_indices, 0] + self.mid_point_uv
        _, indc = self.kdtree.query(reflected[reflected_indices])
        self.transformed_values_uv = self.original_values_uv.copy()
        self.transformed_values_uv[reflected_indices] = self.original_values_uv[indc]
        

    def visualize_uv(self):
        us = np.array([u for u, _ in self.uv_coords])
        vs = np.array([v for _, v in self.uv_coords])
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))
        ax1.scatter(us, vs, c=self.original_values_uv, s=1, cmap='magma', clim=(-1, 1))
        ax1.axline((self.mid_point_uv, 0), (self.mid_point_uv, 1))
        ax2.scatter(us, vs, c=self.transformed_values_uv, s=1, cmap='magma', clim=(-1, 1))
        ax2.axline((self.mid_point_uv, 0), (self.mid_point_uv, 1))
        plt.show()

    def visualize_mesh(self):
        # the "3" is required by pyvista, it's the number of vertices per face
        cells = np.array([[3, *[v[0] - 1 for v in face]] for face in self.faces])
        # fill value of 5 to specify VTK_TRIANGLE
        celltypes = np.full(cells.shape[0], fill_value=5, dtype=int)
        grid = pv.UnstructuredGrid(cells, celltypes, self.vertices)
        grid['transformed_values'] = self.transformed_values_3d
        grid['original_values'] = self.original_values_3d
        
        plot = pv.Plotter(shape=(1, 2))
        plot.subplot(0, 0)
        plot.add_mesh(grid.copy(), scalars='original_values', cmap='magma', clim=(-1, 1))
        plot.subplot(0, 1)
        plot.add_mesh(grid.copy(), scalars='transformed_values', cmap='magma', clim=(-1, 1))
        plot.link_views()
        plot.show()


    # def __extract_face_only(self):
    #     vertices, edges = self.__construct_graph()
    #     connected_components = self.__find_connected_components(vertices, edges)
    #     print([len(c) for c in connected_components])
        

    # def __construct_graph(self):
    #     vertices = [i + 1 for i in range(len(self.vertices))]  # 1-indexed
    #     edges = {}
    #     for face in self.faces:
    #         for i in range(3):
    #             start = face[i][0]
    #             end = face[(i+1)%3][0]
    #             if start not in edges:
    #                 edges[start] = set()
    #             edges[start].add(end)
    #     return vertices, edges
    
    # def __find_connected_components(self, vertices, edges):
    #     queue = deque()
    #     unvisited_vertices = set(vertices)
    #     components = []
    #     while len(unvisited_vertices) > 0:
    #         start = list(unvisited_vertices)[0]
    #         queue.append(start)
    #         current_component = []
    #         while len(queue) > 0:
    #             current = queue.pop()
    #             if current not in unvisited_vertices:
    #                 continue
    #             current_component.append(current)
    #             unvisited_vertices.remove(current)
    #             for neighbour in edges[current]:
    #                 if neighbour in unvisited_vertices:
    #                     queue.append(neighbour)
    #         components.append(current_component)
    #     return components
    

if __name__ == "__main__":
    matplotlib.use('Qt5Agg')

    mesh = MeshWithUV('data/face_surface_with_uv.obj')
    mesh.baseline_transform()
    mesh.visualize_uv()
    mesh.visualize_mesh()