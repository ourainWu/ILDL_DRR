import torch
import torch.nn as nn
import torch.nn.functional as F

class Model(nn.Module):
    def __init__(self, n_features, n_outputs):
        super(Model, self).__init__()
        self.fc = nn.Linear(n_features, n_outputs)

    def forward(self, x):
        x = self.fc(x)
        return F.softmax(x, dim=1)
