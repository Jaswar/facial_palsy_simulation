import torch as th
import numpy as np
from obj_parser import ObjParser
from tetmesh import Tetmesh
import pyvista as pv
import igl
from scipy.spatial import KDTree


# volume of the tetrahedron is 1/6th of the volume of the parallelepiped
# see https://en.wikipedia.org/wiki/Parallelepiped for the original volume formula
def get_tet_volume(nodes, elements):
    v0 = nodes[elements[:, 0]]
    v1 = nodes[elements[:, 1]]
    v2 = nodes[elements[:, 2]]
    v3 = nodes[elements[:, 3]]
    return np.abs(np.einsum('ij,ij->i', v0 - v3, np.cross(v1 - v3, v2 - v3))) / 6


def compute_probabilities(nodes, elements):
    volumes = get_tet_volume(nodes, elements)
    probabilities = volumes / np.sum(volumes)
    return probabilities


# based on https://www.researchgate.net/publication/2461576_Generating_Random_Points_in_a_Tetrahedron
def sample_from_tet(nodes, sampled_elements):
    vertices = nodes[sampled_elements]
    assert vertices.shape[1] == 4, 'Only tetrahedra are supported'
    assert vertices.shape[2] == 3, 'Only 3D points are supported'

    s, t, u = th.hsplit(th.rand(vertices.shape[0], 3).to(nodes.device), 3)

    cond1 = s + t > 1
    s[cond1], t[cond1], u[cond1] = 1 - s[cond1], 1 - t[cond1], u[cond1]

    cond2 = s + t + u > 1
    cond3 = t + u > 1

    cond4 = th.logical_and(cond2, cond3)
    s[cond4], t[cond4], u[cond4] = s[cond4], 1 - u[cond4], 1 - s[cond4] - t[cond4]

    cond5 = th.logical_and(cond2, th.logical_not(cond3))
    s[cond5], t[cond5], u[cond5] = 1 - t[cond5] - u[cond5], t[cond5], s[cond5] + t[cond5] + u[cond5] - 1

    a = vertices[:, 1, :] - vertices[:, 0, :]
    b = vertices[:, 2, :] - vertices[:, 0, :]
    c = vertices[:, 3, :] - vertices[:, 0, :]

    sampled_points = vertices[:, 0, :] + s * a + t * b + u * c
    return sampled_points


class TetmeshDataset(th.utils.data.Dataset):

    def __init__(self, tetmesh_path, jaw_path, skull_path,
                    num_samples=10000, tol=1.0, stol=1e-5, device='cpu'):
        super(TetmeshDataset, self).__init__()
        self.tetmesh_path = tetmesh_path
        self.jaw_path = jaw_path
        self.skull_path = skull_path

        self.num_samples = num_samples
        self.tol = tol
        self.stol = stol
        self.device = device

        # combined mask used during training to specify the type of each node
        self.mask = None

        # for normalization (denormalization if needed)
        self.minv = None
        self.maxv = None

        self.nodes = None
        self.elements = None
        self.deformed_nodes = None
        self.probabilites = None

        self.epoch_nodes = None
        self.epoch_mask = None
        self.epoch_targets = None

    def epilogue(self):
        self.midpoint = np.mean(self.nodes[:, 0])
        self.healthy_indices = self.nodes[:, 0] < self.midpoint

        self.nodes = th.tensor(self.nodes).to(self.device).float()
        self.mask = th.tensor(self.mask).to(self.device).float()
        self.targets = th.tensor(self.deformed_nodes).to(self.device).float()

    def visualize(self, masks):
        numpy_nodes = self.nodes.cpu().numpy()
        colors = ['midnightblue', 'red', 'green', 'yellow']
        assert len(masks) <= len(colors), f'Only up to {len(colors)} node types can be visualized'

        cells = np.hstack([np.full((self.elements.shape[0], 1), 4, dtype=int), self.elements])
        celltypes = np.full(cells.shape[0], fill_value=pv.CellType.TETRA, dtype=int)
        grid = pv.UnstructuredGrid(cells, celltypes, numpy_nodes)
        def_grid = pv.UnstructuredGrid(cells, celltypes, self.deformed_nodes)
        
        plot = pv.Plotter(shape=(1, 2))
        plot.subplot(0, 0)
        plot.add_mesh(grid, color='lightgray')
        for i, mask in enumerate(masks):
            plot.add_points(numpy_nodes[mask], color=colors[i], point_size=7.)
        plot.subplot(0, 1)
        plot.add_mesh(def_grid, color='lightgray')
        plot.link_views()
        plot.show()

    def read(self):
        self.nodes, self.elements, _ = Tetmesh.read_tetgen_file(self.tetmesh_path)
        self.deformed_nodes = self.nodes.copy()
        self.probabilities = compute_probabilities(self.nodes, self.elements)

        self.jaw = pv.read(self.jaw_path)
        self.jaw = self.jaw.clean()
        self.jaw = pv.PolyData(self.jaw)

        self.skull = pv.read(self.skull_path)
        self.skull = self.skull.clean()
        self.skull = pv.PolyData(self.skull)
    
    def normalize(self):
        self.minv = np.min(self.nodes)
        self.maxv = np.max(self.nodes)

        self.nodes = (self.nodes - self.minv) / (self.maxv - self.minv)
        self.deformed_nodes = (self.deformed_nodes - self.minv) / (self.maxv - self.minv)

    def detect(self, mesh, tol):
        ds, _, _ = igl.point_mesh_squared_distance(self.nodes, mesh.points, mesh.regular_faces)
        boundary_mask = ds < tol
        return boundary_mask
    
    def sample_nodes(self, num_nodes):
        if th.is_tensor(num_nodes):
            num_nodes = num_nodes.cpu().item()

        idx = np.random.choice(self.elements.shape[0], num_nodes, p=self.probabilities)
        sampled_elements = self.elements[idx]
        sampled_points = sample_from_tet(self.nodes, sampled_elements)
        return sampled_points

    def prepare_for_epoch(self):
        self.epoch_nodes = self.nodes.clone()
        self.epoch_mask = self.mask.clone()
        self.epoch_targets = self.targets.clone()

        idx = th.randperm(self.epoch_nodes.shape[0])
        self.epoch_nodes = self.epoch_nodes[idx]
        self.epoch_mask = self.epoch_mask[idx]
        self.epoch_targets = self.epoch_targets[idx]  

        where_tissue = self.epoch_mask == 0
        # self.epoch_nodes[where_tissue] = self.sample_nodes(where_tissue.sum())
        return idx

    def __len__(self):
        return min(self.num_samples, self.nodes.shape[0])
    
    def __getitem__(self, idx):
        if th.is_tensor(idx):
            idx = idx.tolist()
        return idx


class INRDataset(th.utils.data.Dataset):

    def __init__(self, tetmesh_path, jaw_path, skull_path, neutral_path, deformed_path,
                 num_samples=10000, tol=1.0, stol=1e-5, device='cpu'):
        super(INRDataset, self).__init__()
        self.tetmesh_path = tetmesh_path
        self.jaw_path = jaw_path
        self.skull_path = skull_path
        self.neutral_path = neutral_path
        self.deformed_path = deformed_path

        self.num_samples = num_samples
        self.tol = tol
        self.stol = stol
        self.device = device

        # the different parts of the face
        self.skull_mask = None
        self.jaw_mask = None
        self.surface_mask = None 
        self.tissue_mask = None
        # combined mask used during training to specify the type of each node
        self.mask = None

        # for normalization (denormalization if needed)
        self.minv = None
        self.maxv = None

        # nodes from the original tetmesh with surface points deformed
        # optionally also the jaw nodes are replaced with prediction if predicted_jaw_path is provided
        self.deformed_nodes = None

        # num_samples sampled nodes from the tetmesh, reshuffled at each epoch in the prepare_for_epoch method
        self.epoch_nodes = None
        self.epoch_mask = None
        self.epoch_targets = None
        
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

        self.midpoint = np.mean(self.nodes[:, 0])
        self.healthy_indices = self.nodes[:, 0] < self.midpoint

        self.nodes = th.tensor(self.nodes).to(device).float()
        self.mask = th.tensor(self.mask).to(device).float()
        self.targets = th.tensor(self.deformed_nodes).to(device).float()

    def visualize(self):
        numpy_nodes = self.nodes.cpu().numpy()
        skull_nodes = numpy_nodes[self.skull_mask]
        jaw_nodes = numpy_nodes[self.jaw_mask]
        surface_nodes = numpy_nodes[self.surface_mask]
        tissue_nodes = numpy_nodes[self.tissue_mask]

        cells = np.hstack([np.full((self.elements.shape[0], 1), 4, dtype=int), self.elements])
        celltypes = np.full(cells.shape[0], fill_value=pv.CellType.TETRA, dtype=int)
        grid = pv.UnstructuredGrid(cells, celltypes, numpy_nodes)
        def_grid = pv.UnstructuredGrid(cells, celltypes, self.deformed_nodes)

        plot = pv.Plotter(shape=(1, 2))
        plot.subplot(0, 0)
        plot.add_mesh(grid, color='lightgray')
        plot.add_points(skull_nodes, color='midnightblue', point_size=7.)
        plot.add_points(jaw_nodes, color='red', point_size=7.)
        plot.add_points(surface_nodes, color='yellow', point_size=7.)
        plot.add_points(tissue_nodes, color='green', point_size=7.)
        plot.subplot(0, 1)
        plot.add_mesh(def_grid, color='lightgray')
        plot.link_views()
        plot.show()

    def __read(self):
        self.nodes, self.elements, _ = Tetmesh.read_tetgen_file(self.tetmesh_path)
        self.probabilities = compute_probabilities(self.nodes, self.elements)

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
        self.deformed_surface = self.deformed_surface.clean(point_merging=False)
        self.deformed_surface = pv.PolyData(self.deformed_surface)
    
    def __normalize(self):
        self.minv = np.min(self.nodes)
        self.maxv = np.max(self.nodes)

        self.nodes = (self.nodes - self.minv) / (self.maxv - self.minv)
        self.deformed_nodes = (self.deformed_nodes - self.minv) / (self.maxv - self.minv)

    def __detect_skull(self):
        self.skull_mask = self.__detect(self.skull, self.tol)

    def __detect_jaw(self):
        self.jaw_mask = self.__detect(self.jaw, self.tol)

    def __detect_surface(self):
        kdtree = KDTree(self.nodes)
        _, indices = kdtree.query(self.neutral_surface.points)
        self.surface_mask = np.zeros(self.nodes.shape[0], dtype=bool)
        self.surface_mask[indices] = True
        
        self.deformed_nodes = self.nodes.copy()
        self.deformed_nodes[indices] = self.deformed_surface.points
    
    def __detect_tissue(self):
        self.tissue_mask = np.logical_and(np.logical_not(self.skull_mask), np.logical_not(self.jaw_mask))
        self.tissue_mask = np.logical_and(self.tissue_mask, np.logical_not(self.surface_mask))

    def __detect(self, mesh, tol):
        ds, _, _ = igl.point_mesh_squared_distance(self.nodes, mesh.points, mesh.regular_faces)
        boundary_mask = ds < tol
        return boundary_mask
    
    def __combine_masks(self):
        self.mask = np.zeros(self.nodes.shape[0])
        self.mask[self.skull_mask] = 1
        self.mask[self.jaw_mask] = 2
        self.mask[self.surface_mask] = 3

    def __sample_nodes(self, num_nodes):
        if th.is_tensor(num_nodes):
            num_nodes = num_nodes.cpu().item()

        idx = np.random.choice(self.elements.shape[0], num_nodes, p=self.probabilities)
        sampled_elements = self.elements[idx]
        sampled_points = sample_from_tet(self.nodes, sampled_elements)
        return sampled_points

    def prepare_for_epoch(self):
        self.epoch_nodes = self.nodes.clone()
        self.epoch_mask = self.mask.clone()
        self.epoch_targets = self.targets.clone()

        idx = th.randperm(self.epoch_nodes.shape[0])
        self.epoch_nodes = self.epoch_nodes[idx]
        self.epoch_mask = self.epoch_mask[idx]
        self.epoch_targets = self.epoch_targets[idx]  

        where_tissue = self.epoch_mask == 0
        self.epoch_nodes[where_tissue] = self.__sample_nodes(where_tissue.sum())

    def __len__(self):
        return min(self.num_samples, self.nodes.shape[0])
    
    def __getitem__(self, idx):
        if th.is_tensor(idx):
            idx = idx.tolist()

        return self.epoch_nodes[idx], self.epoch_mask[idx], self.epoch_targets[idx]
        

class SimulatorDataset(TetmeshDataset):

    def __init__(self, tetmesh_path, jaw_path, skull_path, predicted_jaw_path, 
                 actuations_path=None, actuation_predictor=None,
                 num_samples=10000, tol=1.0, stol=1e-5, device='cpu'):
        super(SimulatorDataset, self).__init__(tetmesh_path, jaw_path, skull_path,
                                               num_samples, tol, stol, device)
        self.actuations_path = actuations_path
        self.actuation_predictor = actuation_predictor
        self.predicted_jaw_path = predicted_jaw_path
        assert actuations_path is not None or actuation_predictor is not None, \
            'Either actuations_path or actuation_predictor must be provided'
        assert actuations_path is None or actuation_predictor is None, \
            'Only one of actuations_path or actuation_predictor can be provided'

        self.actuations = None

        # the different parts of the face
        self.skull_mask = None
        self.jaw_mask = None
        self.tissue_mask = None

        self.epoch_actuations = None  # set only if actuations_path is provided, to be used in the simulation model

        self.__read()

        self.__detect_skull()
        self.__detect_jaw()
        self.__detect_tissue()
        self.__combine_masks()

        self.__replace_jaw_with_prediction()

        self.normalize()
        
        self.epilogue()

    def epilogue(self):
        super(SimulatorDataset, self).epilogue()
        self.actuations = th.tensor(self.actuations).to(self.device).float()

    def visualize(self):
        super(SimulatorDataset, self).visualize(
            [self.skull_mask, self.jaw_mask, self.tissue_mask]
        )

    def __read(self):
        super(SimulatorDataset, self).read()

        if self.actuations_path is not None:
            self.actuations = np.load(self.actuations_path)
        else:
            self.actuations = self.actuation_predictor.predict(self.nodes).cpu().numpy()

    def __detect_skull(self):
        self.skull_mask = self.detect(self.skull, self.tol)
        self.skull_mask = np.logical_or(self.skull_mask, self.nodes[:, 2] < (np.min(self.nodes[:, 2]) + self.tol))

    def __detect_jaw(self):
        self.jaw_mask = self.detect(self.jaw, self.tol)
    
    def __detect_tissue(self):
        self.tissue_mask = np.logical_and(np.logical_not(self.skull_mask), 
                                          np.logical_not(self.jaw_mask))

    def __combine_masks(self):
        self.mask = np.zeros(self.nodes.shape[0])
        self.mask[self.skull_mask] = 1
        self.mask[self.jaw_mask] = 2

    def __replace_jaw_with_prediction(self):
        predicted_jaw = np.load(self.predicted_jaw_path)
        self.deformed_nodes[self.jaw_mask] = predicted_jaw

    def prepare_for_epoch(self):
        idx = super(SimulatorDataset, self).prepare_for_epoch()
        self.epoch_actuations = self.actuations[idx]
        if self.actuation_predictor is not None:
            A, self.epoch_actuations = self.actuation_predictor.predict(self.epoch_nodes, denormalize=True)
            self.epoch_actuations = th.tensor(self.epoch_actuations).to(self.device)

    def __getitem__(self, idx):
        idx = super(SimulatorDataset, self).__getitem__(idx)        
        return self.epoch_nodes[idx], self.epoch_mask[idx], self.epoch_targets[idx], self.epoch_actuations[idx]
        
