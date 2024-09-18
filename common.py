import pyvista as pv
import torch as th
import numpy as np
import time 


def visualize_displacements(model, dataset):
    predicted_vertices = model.predict(dataset.nodes).cpu().numpy()

    if hasattr(dataset, 'elements'):
        cells = np.hstack([np.full((dataset.elements.shape[0], 1), 4, dtype=int), dataset.elements])
        celltypes = np.full(cells.shape[0], fill_value=pv.CellType.TETRA, dtype=int)
    else:
        faces = dataset.neutral_surface.regular_faces
        cells = np.hstack([np.full((faces.shape[0], 1), 3, dtype=int), faces])
        celltypes = np.full(cells.shape[0], fill_value=pv.CellType.TRIANGLE, dtype=int)

    neutral_grid = pv.UnstructuredGrid(cells, celltypes, dataset.nodes.cpu().numpy())
    ground_truth_grid = pv.UnstructuredGrid(cells, celltypes, dataset.deformed_nodes)
    predicted_grid = pv.UnstructuredGrid(cells, celltypes, predicted_vertices)
    
    plot = pv.Plotter(shape=(1, 3))
    plot.subplot(0, 0)
    plot.add_mesh(neutral_grid.copy(), color='yellow')
    plot.subplot(0, 1)
    plot.add_mesh(ground_truth_grid.copy(), color='yellow')
    plot.subplot(0, 2)
    plot.add_mesh(predicted_grid.copy(), color='yellow')
    plot.link_views()
    plot.show()


def train_model(model, dataset, 
                optimizer, lr_scheduler, batch_size, epochs, 
                print_interval, vis_interval, benchmark, checkpoint_path):
    min_loss = float('inf')
    start = time.time()
    for epoch in range(1, epochs + 1):
        train_loss = model.train_epoch(optimizer, dataset, batch_size)
        if train_loss < min_loss and not benchmark:
            min_loss = train_loss
            th.save(model.state_dict(), checkpoint_path)
        lr_scheduler.step()

        if epoch % print_interval == 0:
            print(f'Epoch {epoch}/{epochs} - Loss: {train_loss:.8f} - LR: {lr_scheduler.get_last_lr()[0]:.8f}')
        if epoch % vis_interval == 0 and not benchmark:
            visualize_displacements(model, dataset)
    print(f'Training took: {time.time() - start:.2f}s')


def get_optimizer(config, model):
    return {'adam': th.optim.Adam,
            'rmsprop': th.optim.RMSprop,
            'sgd': th.optim.SGD}[config['optimizer']](model.parameters(), lr=config['learning_rate'])

