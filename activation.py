import torch.nn as nn
from enum import Enum

class Activation(Enum):
    RELU = "relu"
    SIGMOID = "sigmoid"
    TANH = "tanh"
    SILU = "silu"
    GELU = "gelu"

    def build(self) -> nn.Module:
        return {
            Activation.RELU: nn.ReLU(inplace=True),
            Activation.SIGMOID: nn.Sigmoid(),
            Activation.TANH: nn.Tanh(),
            Activation.SILU: nn.SiLU(inplace=True),
            Activation.GELU: nn.GELU(),
        }[self]