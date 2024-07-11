import torch as th
import numpy as np
from obj_parser import ObjParser
from tetmesh import Tetmesh
import pyvista as pv
import igl


class MeshDataset(th.utils.data.Dataset):

    def __init__(self, neutral_path, deformed_path, device='cpu'):
        super(MeshDataset, self).__init__()
        self.neutral_path = neutral_path
        self.deformed_path = deformed_path

        self.parser = ObjParser()
        self.neutral_vertices, self.faces, _, _ = self.parser.parse(neutral_path)
        self.deformed_vertices, _, _, _ = self.parser.parse(deformed_path)
        assert len(self.neutral_vertices) == len(self.deformed_vertices)

        self.neutral_vertices = self.neutral_vertices.astype(np.float32)
        self.deformed_vertices = self.deformed_vertices.astype(np.float32)

        self.minv = np.min(self.neutral_vertices)
        self.maxv = np.max(self.neutral_vertices)

        self.neutral_vertices = (self.neutral_vertices - self.minv) / (self.maxv - self.minv)
        self.deformed_vertices = (self.deformed_vertices - self.minv) / (self.maxv - self.minv)
        self.displacements = self.deformed_vertices - self.neutral_vertices

        self.neutral_vertices = th.tensor(self.neutral_vertices).to(device)
        self.deformed_vertices = th.tensor(self.deformed_vertices).to(device)
        self.displacements = th.tensor(self.displacements).to(device)

        self.midpoint = th.mean(self.neutral_vertices[:, 0])
        self.relevant_indices = th.where(self.neutral_vertices[:, 0] < self.midpoint)[0]
        self.healthy_neutral_vertices = self.neutral_vertices[self.relevant_indices]
        self.healthy_deformed_vertices = self.deformed_vertices[self.relevant_indices]
        self.healthy_displacements = self.displacements[self.relevant_indices]
    
    def dimensionality(self):
        return self.neutral_vertices.shape[1]

    def __len__(self):
        return len(self.healthy_displacements)
    
    def __getitem__(self, idx):
        if th.is_tensor(idx):
            idx = idx.tolist()

        return self.healthy_neutral_vertices[idx], self.healthy_displacements[idx]


class TetmeshDataset(th.utils.data.Dataset):

    def __init__(self, tetmesh_path, jaw_path, skull_path, neutral_path, deformed_path, tol=1.0, stol=1e-5, device='cpu'):
        super(TetmeshDataset, self).__init__()
        self.tetmesh_path = tetmesh_path
        self.jaw_path = jaw_path
        self.skull_path = skull_path
        self.neutral_path = neutral_path
        self.deformed_path = deformed_path
        self.tol = tol
        self.stol = stol
        self.device = device
        
        # the different parts of the face
        self.skull_nodes = None
        self.skull_mask = None
        self.jaw_nodes = None
        self.jaw_mask = None
        self.surface_mask = None  # no surface nodes as we will use self.neutral_surface and self.deformed_surface
        self.tissue_nodes = None
        self.tissue_mask = None

        # for normalization (denormalization if needed)
        self.minv = None
        self.maxv = None

        # combined mask used during training to specify the type of each node
        self.mask = None
        
        self.__read()

        self.__detect_skull()
        self.__detect_jaw()
        self.__detect_surface()
        assert self.surface_mask.sum() == self.neutral_surface.points.shape[0], \
            'Surface nodes and neutral surface points do not match. Perhaps try different stol?'
        assert self.surface_mask.sum() == self.deformed_surface.points.shape[0]
        self.__detect_tissue()
        self.__combine_masks()

        self.__normalize()

        self.displacements = self.deformed_surface.points - self.neutral_surface.points


    def visualize(self):
        cells = np.hstack([np.full((self.elements.shape[0], 1), 4, dtype=int), self.elements])
        celltypes = np.full(cells.shape[0], fill_value=pv.CellType.TETRA, dtype=int)
        grid = pv.UnstructuredGrid(cells, celltypes, self.nodes)

        plot = pv.Plotter()
        plot.add_mesh(grid, color='lightgray')
        plot.add_points(self.skull_nodes, color='midnightblue', point_size=7.)
        plot.add_points(self.jaw_nodes, color='red', point_size=7.)
        plot.add_points(self.neutral_surface.points, color='yellow', point_size=7.)
        plot.add_points(self.tissue_nodes, color='green', point_size=7.)
        plot.show()

    def __read(self):
        self.nodes, self.elements, _ = Tetmesh.read_tetgen_file(self.tetmesh_path)

        self.jaw = pv.read(self.jaw_path)
        self.jaw = self.jaw.clean()
        self.jaw = pv.PolyData(self.jaw)

        self.skull = pv.read(self.skull_path)
        self.skull = self.skull.clean()
        self.skull = pv.PolyData(self.skull)

        self.neutral_surface = pv.read(self.neutral_path)
        self.neutral_surface = self.neutral_surface.clean()
        self.neutral_surface = pv.PolyData(self.neutral_surface)

        self.deformed_surface = pv.read(self.deformed_path)
        self.deformed_surface = self.deformed_surface.clean()
        self.deformed_surface = pv.PolyData(self.deformed_surface)
    
    def __normalize(self):
        self.minv = np.min(self.nodes)
        self.maxv = np.max(self.nodes)

        self.nodes = (self.nodes - self.minv) / (self.maxv - self.minv)
        self.skull_nodes = (self.skull_nodes - self.minv) / (self.maxv - self.minv)
        self.jaw_nodes = (self.jaw_nodes - self.minv) / (self.maxv - self.minv)
        self.neutral_surface.points = (self.neutral_surface.points - self.minv) / (self.maxv - self.minv)
        self.deformed_surface.points = (self.deformed_surface.points - self.minv) / (self.maxv - self.minv)
        self.tissue_nodes = (self.tissue_nodes - self.minv) / (self.maxv - self.minv)

    def __detect_skull(self):
        self.skull_mask = self.__detect(self.skull, self.tol)
        self.skull_nodes = self.nodes[self.skull_mask]

    def __detect_jaw(self):
        self.jaw_mask = self.__detect(self.jaw, self.tol)
        self.jaw_nodes = self.nodes[self.jaw_mask]

    def __detect_surface(self):
        self.surface_mask, I = self.__detect(self.neutral_surface, self.stol, return_indices=True)
        self.deformed_nodes = self.nodes.copy()
        self.deformed_nodes[I] = self.deformed_surface.points
    
    def __detect_tissue(self):
        self.tissue_mask = np.logical_and(np.logical_not(self.skull_mask), np.logical_not(self.jaw_mask))
        self.tissue_mask = np.logical_and(self.tissue_mask, np.logical_not(self.surface_mask))
        self.tissue_nodes = self.nodes[self.tissue_mask]

    def __detect(self, mesh, tol, return_indices=False):
        ds, I, _ = igl.point_mesh_squared_distance(self.nodes, mesh.points, mesh.regular_faces)
        boundary_mask = ds < tol
        if return_indices:
            return boundary_mask, I
        else:
            return boundary_mask
    
    def __combine_masks(self):
        self.mask = np.zeros(self.nodes.shape[0])
        self.mask[self.skull_mask] = 1
        self.mask[self.jaw_mask] = 2
        self.mask[self.surface_mask] = 3

    def __len__(self):
        return len(self.nodes)


if __name__ == '__main__':
    dataset = TetmeshDataset('data/tetmesh', 
                             'data/jaw.obj', 
                             'data/skull.obj', 
                             'data/tetmesh_face_surface.obj', 
                             'data/ground_truths/deformed_surface_001.obj')
    dataset.visualize()

