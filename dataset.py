import torch as th
import numpy as np
from obj_parser import ObjParser
from tetmesh import Tetmesh
import pyvista as pv
import igl
from scipy.spatial import KDTree


class TetmeshDataset(th.utils.data.Dataset):

    def __init__(self, tetmesh_path, jaw_path, skull_path, neutral_path, deformed_path, actuations_path=None, predicted_jaw_path=None,
                 generate_prestrain=False, use_prestrain=False, prestrain_model=None, 
                 num_samples=10000, tol=1.0, stol=1e-5, device='cpu'):
        super(TetmeshDataset, self).__init__()
        self.tetmesh_path = tetmesh_path
        self.jaw_path = jaw_path
        self.skull_path = skull_path
        self.neutral_path = neutral_path
        self.deformed_path = deformed_path
        self.actuations_path = actuations_path
        self.predicted_jaw_path = predicted_jaw_path

        self.generate_prestrain = generate_prestrain
        self.use_prestrain = use_prestrain
        self.prestrain_model = prestrain_model
        assert self.prestrain_model is not None or not self.use_prestrain, 'Prestrain model must be provided if use_prestrain is True'
        assert not generate_prestrain or not use_prestrain, 'Cannot generate and use prestrain at the same time'

        self.num_samples = num_samples
        self.tol = tol
        self.stol = stol
        self.device = device

        self.actuations = None

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
        self.epoch_actuations = None  # set only if actuations_path is provided, to be used in the simulation model
        
        self.__read()

        self.__detect_skull()
        self.__detect_jaw()
        self.__detect_surface()
        assert self.surface_mask.sum() == self.neutral_surface.points.shape[0], \
            'Surface nodes and neutral surface points do not match. Perhaps try different stol?'
        assert self.surface_mask.sum() == self.deformed_surface.points.shape[0]
        self.__detect_tissue()
        self.__combine_masks()

        if self.predicted_jaw_path is not None:
            self.__replace_jaw_with_prediction()

        self.__normalize()

        if self.use_prestrain:
            self.__replace_with_prestrain()

        self.midpoint = np.mean(self.nodes[:, 0])
        self.healthy_indices = self.nodes[:, 0] < self.midpoint

        if self.generate_prestrain:
            self.__symmetrize_surface()

        self.nodes = th.tensor(self.nodes).to(device).float()
        self.mask = th.tensor(self.mask).to(device).float()
        self.targets = th.tensor(self.deformed_nodes).to(device).float()
        if self.actuations is not None:
            self.actuations = th.tensor(self.actuations).to(device).float()

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

        if self.actuations_path is not None:
            self.actuations = np.load(self.actuations_path)
    
    def __normalize(self):
        self.minv = np.min(self.nodes)
        self.maxv = np.max(self.nodes)

        self.nodes = (self.nodes - self.minv) / (self.maxv - self.minv)
        self.deformed_nodes = (self.deformed_nodes - self.minv) / (self.maxv - self.minv)

    def __detect_skull(self):
        self.skull_mask = self.__detect(self.skull, self.tol)
        if self.actuations_path is not None:
            self.skull_mask = np.logical_or(self.skull_mask, self.nodes[:, 2] < (np.min(self.nodes[:, 2]) + self.tol))

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

    def __replace_with_prestrain(self):
        self.nodes = self.prestrain_model.predict(th.tensor(self.nodes).float()).numpy()

    def __replace_jaw_with_prediction(self):
        predicted_jaw = np.load(self.predicted_jaw_path)
        self.deformed_nodes[self.jaw_mask] = predicted_jaw

    def __symmetrize_surface(self):
        healthy_surface_idx = np.logical_and(self.surface_mask, self.healthy_indices)
        unhealthy_surface_idx = np.logical_and(self.surface_mask, ~self.healthy_indices)

        self.deformed_nodes = self.nodes.copy()
        self.deformed_nodes[healthy_surface_idx, 0] = self.midpoint - self.deformed_nodes[healthy_surface_idx, 0] + self.midpoint
        kdtree = KDTree(self.deformed_nodes[healthy_surface_idx])
        _, indices = kdtree.query(self.deformed_nodes[unhealthy_surface_idx])
        self.deformed_nodes[unhealthy_surface_idx] = self.deformed_nodes[healthy_surface_idx][indices]
        self.deformed_nodes[healthy_surface_idx, 0] = self.midpoint - self.deformed_nodes[healthy_surface_idx, 0] + self.midpoint

    def prepare_for_epoch(self):
        self.epoch_nodes = self.nodes.clone()
        self.epoch_mask = self.mask.clone()
        self.epoch_targets = self.targets.clone()
        if self.actuations is not None:
            self.epoch_actuations = self.actuations.clone()

        idx = th.randperm(self.epoch_nodes.shape[0])
        self.epoch_nodes = self.epoch_nodes[idx]
        self.epoch_mask = self.epoch_mask[idx]
        self.epoch_targets = self.epoch_targets[idx]  
        if self.actuations is not None:
            self.epoch_actuations = self.epoch_actuations[idx]  

    def __len__(self):
        return min(self.num_samples, self.nodes.shape[0])
    
    def __getitem__(self, idx):
        if th.is_tensor(idx):
            idx = idx.tolist()

        if self.actuations is None:
            return self.epoch_nodes[idx], self.epoch_mask[idx], self.epoch_targets[idx], None
        else:
            return self.epoch_nodes[idx], self.epoch_mask[idx], self.epoch_targets[idx], self.epoch_actuations[idx]


