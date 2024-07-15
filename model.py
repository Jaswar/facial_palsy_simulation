import torch as th
import numpy as np


# pytorch implementation of procrustes
# based on https://gist.github.com/mkocabas/54ea2ff3b03260e3fedf8ad22536f427
def procrustes_loss(S1, S2):
    '''
    Computes a similarity transform (sR, t) that takes
    a set of 3D points S1 (3 x N) closest to a set of 3D points S2,
    where R is an 3x3 rotation matrix, t 3x1 translation, s scale.
    i.e. solves the orthogonal Procrutes problem.
    '''
    # the default shape is 3 x N but we have N x 3, so transpose
    transposed = True
    S1 = S1.T
    S2 = S2.T
    assert (S2.shape[1] == S1.shape[1])

    # 1. Remove mean.
    mu1 = S1.mean(axis=1, keepdims=True)
    mu2 = S2.mean(axis=1, keepdims=True)
    X1 = S1 - mu1
    X2 = S2 - mu2

    # print('X1', X1.shape)

    # 2. Compute variance of X1 used for scale.
    var1 = th.sum(X1 ** 2)

    # print('var', var1.shape)

    # 3. The outer product of X1 and X2.
    K = X1.mm(X2.T)

    # 4. Solution that Maximizes trace(R'K) is R=U*V', where U, V are
    # singular vectors of K.
    U, s, V = th.svd(K)
    # V = Vh.T
    # Construct Z that fixes the orientation of R to get det(R)=1.
    Z = th.eye(U.shape[0], device=S1.device)
    Z[-1, -1] *= th.sign(th.det(U @ V.T))
    # Construct R.
    R = V.mm(Z.mm(U.T))

    # print('R', X1.shape)

    # 5. Recover scale.
    scale = th.trace(R.mm(K)) / var1
    # print(R.shape, mu1.shape)
    # 6. Recover translation.
    t = mu2 - scale * (R.mm(mu1))
    # print(t.shape)

    # 7. Error:
    S1_hat = scale * R.mm(S1) + t

    if transposed:
        S1_hat = S1_hat.T
        S2 = S2.T

    error = th.mean(th.norm(S1_hat - S2, dim=1))
    return error


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
        # self.layers.append(th.nn.Linear())
    
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
    
    def train_epoch(self, dataloader, optimizer): 
        self.train()
        total_loss = 0.
        total_samples = 0
        for neutral, mask, target in dataloader:
            prediction = self(neutral)
            loss = self.compute_loss(prediction, target, mask)

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
    
    def compute_loss(self, prediction, target, mask):
        where_tissue = mask == 0
        where_skull = mask == 1
        where_jaw = mask == 2
        where_surface = mask == 3

        surface_loss = th.tensor(0., device=prediction.device)
        if where_surface.sum() > 0:
            surface_loss = th.nn.functional.l1_loss(prediction[where_surface], target[where_surface])
        
        skull_loss = th.tensor(0., device=prediction.device)
        if where_skull.sum() > 0:
            skull_loss = th.nn.functional.l1_loss(prediction[where_skull], target[where_skull])

        # loss[where_tissue] = th.nn.functional.l1_loss(prediction[where_tissue], target[where_tissue])
        # loss[where_jaw] = procrustes_loss(prediction[where_jaw], target[where_jaw])

        # loss[where_surface] /= where_surface.sum()
        # loss[where_skull] /= where_skull.sum()
        # loss[where_jaw] /= where_jaw.sum()

        # print(loss[where_surface].mean(), loss[where_skull].mean(), loss[where_jaw].mean())

        loss = surface_loss + skull_loss
        return loss

