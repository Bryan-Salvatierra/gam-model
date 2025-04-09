import torch
import torch.nn as nn
from torch import Tensor

__all__ = [
    "Discriminator", "Generator"
]

class Discriminator(nn.Module):
    def __init__(self) -> None:
        super(Discriminator, self).__init__()
        self.main = nn.Sequential(
            nn.Linear(1 * 28 * 28, 512, bias=True),
            nn.LeakyReLU(0.2, True),
            nn.Linear(512, 256, bias=True),
            nn.LeakyReLU(0.2, True),
            nn.Linear(256, 1, bias=True),
            nn.Sigmoid()
        )
    
    def forward(self, x: Tensor) -> Tensor:
        out = torch.flatten(x, 1)
        out = self.main(out)
        return out
        
class Generator(nn.Module):
    def __init__(self) -> None:
        super(Generator, self).__init__()
        self.main = nn.Sequential(
            nn.Linear(100, 128, bias=True),
            nn.LeakyReLU(0.2, True),
            nn.Linear(128, 256, bias=False),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(0.2, True),
            nn.Linear(256, 512, bias=False),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(0.2, True),
            nn.Linear(512, 1024, bias=False),
            nn.BatchNorm1d(1024),
            nn.LeakyReLU(0.2, True),
            nn.Linear(1024, 1 * 28 * 28, bias=True),
            nn.Tanh()
        )
    
    def forward(self, x: Tensor) -> Tensor:
        out = self.main(x)
        out = out.reshape(out.size(0), 1, 28, 28)
        return out
