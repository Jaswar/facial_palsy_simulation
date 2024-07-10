import torch as th
import numpy as np


# module implementation of a sine activation (doesn't exist in PyTorch)
class Sin(th.nn.Module):

    def __init__(self):
        super(Sin, self).__init__()
    
    def forward(self, x):
        return th.sin(x)


class Model(th.nn.Module):
    
    def __init__(self, input_size=3, output_size=3, num_hidden_layers=3, hidden_size=32, with_fourier=True, fourier_features=10):
        super(Model, self).__init__()
        self.num_hidden_layers = num_hidden_layers
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.with_fourier = with_fourier
        self.fourier_features = fourier_features

        self.layers = th.nn.ModuleList()
        if not with_fourier:
            self.layers.append(th.nn.Linear(input_size, hidden_size))
        else:
            self.layers.append(th.nn.Linear(input_size * 2 * fourier_features, hidden_size))
        self.layers.append(Sin())
        for _ in range(self.num_hidden_layers):
            self.layers.append(th.nn.Linear(hidden_size, hidden_size))
            self.layers.append(Sin())
        self.layers.append(th.nn.Linear(hidden_size, output_size))
        self.layers.append(th.nn.Tanh())
    
    def fourier_encode(self, x):
        # based on https://github.com/jmclong/random-fourier-features-pytorch/blob/main/rff/functional.py
        features = th.arange(0, self.fourier_features, device=x.device)
        features = np.pi * 2 ** features
        xff = features * x.unsqueeze(-1)
        xff = th.cat([th.sin(xff), th.cos(xff)], dim=-1)
        xff = xff.view(xff.size(0), -1)
        return xff

    def forward(self, x):
        if self.with_fourier:
            x = self.fourier_encode(x)
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
