'''
code from: http://d2l.ai/chapter_convolutional-modern/batch-norm.html?highlight=batchnorm2d
'''

import torch
import dgl
from torch import nn

def graph_norm(X, g, gamma, beta, moving_mean, moving_var, eps, momentum, mask2d=None):
    # Use `is_grad_enabled` to determine whether the current mode is training
    # mode or prediction mode
    if not torch.is_grad_enabled():
        # If it is prediction mode, directly use the mean and variance
        # obtained by moving average
        X_hat = (X - moving_mean) / torch.sqrt(moving_var + eps)
    else:
        g.ndata['p'] = X
        mean = dgl.mean_nodes(g, 'p')
        if mask2d is not None:
            mean = mean.reshape(mask2d.size(0), mask2d.size(1), -1)
            mask2d = mask2d.reshape(mask2d.size(0), mask2d.size(1), 1)
            mean = (mean * mask2d).sum(dim=(0, 1)) / torch.sum(mask2d)
            mean = mean.reshape(1, -1)
        else:
            mean = torch.mean(mean, dim=0, keepdim=True)
        del g.ndata['p']
        var = ((X - mean) ** 2).mean(dim=0, keepdim=True)
        X_hat = (X - mean) / torch.sqrt(var + eps)
        moving_mean = momentum * moving_mean + (1.0 - momentum) * mean
        moving_var = momentum * moving_var + (1.0 - momentum) * var
    Y = gamma * X_hat + beta  # Scale and shift
    return Y, moving_mean.data, moving_var.data


class GraphNorm(nn.Module):
    def __init__(self, num_features):
        super().__init__()
        shape = (1, num_features)
        # The scale parameter and the shift parameter (model parameters) are
        # initialized to 1 and 0, respectively
        self.gamma = nn.Parameter(torch.ones(shape))
        self.beta = nn.Parameter(torch.zeros(shape))
        # The variables that are not model parameters are initialized to 0 and 1
        self.moving_mean = torch.zeros(shape)
        self.moving_var = torch.ones(shape)

    def forward(self, X, g, mask2d=None):
        # If `X` is not on the main memory, copy `moving_mean` and
        # `moving_var` to the device where `X` is located
        if self.moving_mean.device != X.device:
            self.moving_mean = self.moving_mean.to(X.device)
            self.moving_var = self.moving_var.to(X.device)
        # Save the updated `moving_mean` and `moving_var`
        Y, self.moving_mean, self.moving_var = graph_norm(
            X, g, self.gamma, self.beta, self.moving_mean,
            self.moving_var, eps=1e-5, momentum=0.9, mask2d=mask2d)
        return Y