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
                 sample, num_samples, tol, device):
        super(TetmeshDataset, self).__init__()
        self.tetmesh_path = tetmesh_path
        self.jaw_path = jaw_path
        self.skull_path = skull_path

        self.sample = sample
        self.num_samples = num_samples
        self.tol = tol
        self.device = device

        self.jaw_mask = None
        self.skull_mask = None
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

    def detect_skull(self):
        self.skull_mask = self.detect(self.skull, self.tol)

    def detect_jaw(self):
        self.jaw_mask = self.detect(self.jaw, self.tol)

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

        if self.sample:
            where_tissue = self.epoch_mask == 0
            self.epoch_nodes[where_tissue] = self.sample_nodes(where_tissue.sum())
        return idx

    def __len__(self):
        return min(self.num_samples, self.nodes.shape[0])
    
    def __getitem__(self, idx):
        if th.is_tensor(idx):
            idx = idx.tolist()
        return idx


class INRDataset(TetmeshDataset):
    SKULL_MASK = 1
    JAW_MASK = 2
    SURFACE_MASK = 3

    def __init__(self, tetmesh_path, jaw_path, skull_path, neutral_path, deformed_path,
                 sample=False, num_samples=10000, tol=1.0, stol=1e-5, device='cpu'):
        super(INRDataset, self).__init__(tetmesh_path, jaw_path, skull_path,
                                         sample, num_samples, tol, device)
        self.stol = stol
        self.neutral_path = neutral_path
        self.deformed_path = deformed_path

        # the different parts of the face
        self.skull_mask = None
        self.jaw_mask = None
        self.surface_mask = None 
        self.tissue_mask = None

        self.__read()

        self.detect_skull()
        self.detect_jaw()
        self.__detect_surface()
        assert self.surface_mask.sum() == self.neutral_surface.points.shape[0], \
            'Surface nodes and neutral surface points do not match. Perhaps try different stol?'
        assert self.surface_mask.sum() == self.deformed_surface.points.shape[0]
        self.__detect_tissue()
        self.__combine_masks()

        self.normalize()

        self.epilogue()

    def visualize(self):
        super(INRDataset, self).visualize(
            [self.skull_mask, self.jaw_mask, self.surface_mask, self.tissue_mask]
        )

    def __read(self):
        super(INRDataset, self).read()

        self.neutral_surface = pv.read(self.neutral_path)
        self.neutral_surface = self.neutral_surface.clean()
        self.neutral_surface = pv.PolyData(self.neutral_surface)

        self.deformed_surface = pv.read(self.deformed_path)
        self.deformed_surface = self.deformed_surface.clean(point_merging=False)
        self.deformed_surface = pv.PolyData(self.deformed_surface)

    def __detect_surface(self):
        kdtree = KDTree(self.nodes)
        _, indices = kdtree.query(self.neutral_surface.points)
        self.surface_mask = np.zeros(self.nodes.shape[0], dtype=bool)
        self.surface_mask[indices] = True
        self.deformed_nodes[indices] = self.deformed_surface.points
    
    def __detect_tissue(self):
        self.tissue_mask = np.logical_and(np.logical_not(self.skull_mask), np.logical_not(self.jaw_mask))
        self.tissue_mask = np.logical_and(self.tissue_mask, np.logical_not(self.surface_mask))
    
    def __combine_masks(self):
        self.mask = np.zeros(self.nodes.shape[0])
        self.mask[self.skull_mask] = INRDataset.SKULL_MASK
        self.mask[self.jaw_mask] = INRDataset.JAW_MASK
        self.mask[self.surface_mask] = INRDataset.SURFACE_MASK
    
    def __getitem__(self, idx):
        idx = super(INRDataset, self).__getitem__(idx)
        return self.epoch_nodes[idx], self.epoch_mask[idx], self.epoch_targets[idx]
        

class SimulatorDataset(TetmeshDataset):
    FIXED_MASK = 1

    def __init__(self, tetmesh_path, jaw_path, skull_path, predicted_jaw_path, actuation_predictor,
                 sample=False, num_samples=10000, tol=1.0, device='cpu'):
        super(SimulatorDataset, self).__init__(tetmesh_path, jaw_path, skull_path,
                                               sample, num_samples, tol, device)
        self.actuation_predictor = actuation_predictor
        self.predicted_jaw_path = predicted_jaw_path

        self.actuations = None

        # the different parts of the face
        self.box_mask = None
        self.tissue_mask = None

        self.epoch_actuations = None

        self.__read()

        self.detect_skull()
        self.detect_jaw()
        self.__detect_box()
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
            [self.skull_mask, self.jaw_mask, self.tissue_mask, self.box_mask]
        )

    def __read(self):
        super(SimulatorDataset, self).read()
        _, self.actuations = self.actuation_predictor.predict(self.nodes)
        self.actuations = th.tensor(self.actuations).to(self.device).float()

    def __detect_box(self):
        self.box_mask = self.nodes[:, 2] < (np.min(self.nodes[:, 2]) + self.tol)
    
    def __detect_tissue(self):
        self.tissue_mask = np.logical_and(np.logical_not(self.skull_mask), 
                                          np.logical_not(self.jaw_mask))
        self.tissue_mask = np.logical_and(self.tissue_mask, np.logical_not(self.box_mask))

    def __combine_masks(self):
        self.mask = np.zeros(self.nodes.shape[0])
        self.mask[~self.tissue_mask] = SimulatorDataset.FIXED_MASK

    def __replace_jaw_with_prediction(self):
        predicted_jaw = np.load(self.predicted_jaw_path)
        self.deformed_nodes[self.jaw_mask] = predicted_jaw

    def prepare_for_epoch(self):
        idx = super(SimulatorDataset, self).prepare_for_epoch()
        self.epoch_actuations = self.actuations[idx]
        if self.sample:
            # if sampling, it is necessary to recalculate the actuations
            # since the points are anywhere within the tetmesh
            _, self.epoch_actuations = self.actuation_predictor.predict(self.epoch_nodes, denormalize=True)
            self.epoch_actuations = th.tensor(self.epoch_actuations).to(self.device)

    def __getitem__(self, idx):
        idx = super(SimulatorDataset, self).__getitem__(idx)        
        return self.epoch_nodes[idx], self.epoch_mask[idx], self.epoch_targets[idx], self.epoch_actuations[idx]
        
