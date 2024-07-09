import torch as th
import numpy as np
from obj_parser import ObjParser
import argparse
import pyvista as pv


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


# module implementation of a sine activation (doesn't exist in PyTorch)
class Sin(th.nn.Module):

    def __init__(self):
        super(Sin, self).__init__()
    
    def forward(self, x):
        return th.sin(x)


class Model(th.nn.Module):
    
    def __init__(self, input_size=3, output_size=3, num_hidden_layers=3, hidden_size=32):
        super(Model, self).__init__()
        self.num_hidden_layers = num_hidden_layers
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        self.layers = th.nn.ModuleList()
        self.layers.append(th.nn.Linear(input_size, hidden_size))
        self.layers.append(Sin())
        for _ in range(self.num_hidden_layers):
            self.layers.append(th.nn.Linear(hidden_size, hidden_size))
            self.layers.append(Sin())
        self.layers.append(th.nn.Linear(hidden_size, output_size))
        self.layers.append(th.nn.Tanh())
    
    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x
    
    def train_epoch(self, dataloader, optimizer, criterion): 
        self.train()       
        total_loss = 0.
        total_samples = 0
        for neutral, displacement in dataloader:
            prediction = self(neutral)
            loss = criterion(prediction, displacement)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_samples += len(neutral)
            total_loss += loss.item() * len(neutral)
        return total_loss / total_samples
    
    def predict(self, data):
        self.eval()
        with th.no_grad():
            result = self(data)
        return result


def visualize_displacements(model, dataset):
    predicted_displacements = model.predict(dataset.neutral_vertices)
    all_predicted_vertices = dataset.neutral_vertices + predicted_displacements
    ground_truth_vertices = dataset.neutral_vertices + dataset.displacements

    # only replace the healthy part of the face
    predicted_vertices = ground_truth_vertices.clone()
    predicted_vertices[dataset.relevant_indices] = all_predicted_vertices[dataset.relevant_indices]

    predicted_vertices = predicted_vertices.cpu().numpy()
    ground_truth_vertices = ground_truth_vertices.cpu().numpy()

    # the "3" is required by pyvista, it's the number of vertices per face
    cells = np.array([[3, *[v[0] - 1 for v in face]] for face in dataset.faces])
    # fill value of 5 to specify VTK_TRIANGLE
    celltypes = np.full(cells.shape[0], fill_value=5, dtype=int)

    neutral_grid = pv.UnstructuredGrid(cells, celltypes, dataset.neutral_vertices.cpu().numpy())
    predicted_grid = pv.UnstructuredGrid(cells, celltypes, predicted_vertices)
    ground_truth_grid = pv.UnstructuredGrid(cells, celltypes, ground_truth_vertices)
    
    plot = pv.Plotter(shape=(1, 3))
    plot.subplot(0, 0)
    plot.add_mesh(neutral_grid.copy(), color='yellow')
    plot.subplot(0, 1)
    plot.add_mesh(predicted_grid.copy(), color='yellow')
    plot.subplot(0, 2)
    plot.add_mesh(ground_truth_grid.copy(), color='yellow')
    plot.link_views()
    plot.show()


def main():
    neutral_path = 'data/face_surface_with_uv3.obj'
    deformed_path = 'data/ground_truths/deformed_surface_001.obj'
    epochs = 1000
    batch_size = 32

    if th.cuda.is_available():
        device = 'cuda'
    else:
        device = 'cpu'
    print(f'Using device: {device}')
    # device = 'cpu'

    dataset = MeshDataset(neutral_path, deformed_path, device=device)
    loader = th.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False)
    model = Model(input_size=dataset.dimensionality(), output_size=dataset.dimensionality(), num_hidden_layers=8, hidden_size=256)
    model.to(device)

    criterion = th.nn.L1Loss()
    optimizer = th.optim.Adam(model.parameters(), lr=1e-4)
    for epoch in range(1, epochs + 1):
        train_loss = model.train_epoch(loader, optimizer, criterion)
        print(f'Epoch {epoch}/{epochs} - Loss: {train_loss:.6f}')
        if epoch % 100 == 0:
            visualize_displacements(model, dataset)

    visualize_displacements(model, dataset)

if __name__ == '__main__':
    main()


