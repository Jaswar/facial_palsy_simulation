import torch as th
import numpy as np
from obj_parser import ObjParser


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
