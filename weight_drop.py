from typing import Sequence
import torch
import torch.nn.functional as F
from torch.nn import Module, Parameter


class WeightDrop(Module):

    def __init__(self, module: Module, weight_names: Sequence[str], p: float):
        super().__init__()
        self.module = module
        self.weight_names = weight_names
        self.p = p
        
        # Hack needed when running using CUDA. Taken from github.com/salesforce/awd-lstm-lm/blob/master/weight_drop.py
        if isinstance(self.module, torch.nn.RNNBase):
            self.module.flatten_parameters = self.void
        
        for name in self.weight_names:
            self.module.register_parameter(f"{name}_all", Parameter(getattr(self.module, name)))
            del self.module._parameters[name]

    def void(*args, **kwargs):
        return
            
    def forward(self, *args):
        for name in self.weight_names:
            setattr(self.module, name, F.dropout(getattr(self.module, f"{name}_all"), self.p, training=self.training))
        return self.module.forward(*args)
