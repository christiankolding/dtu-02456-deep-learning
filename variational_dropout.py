import torch
import torch.nn as nn
from torch.autograd import Variable


class VariationalDropout(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, tensor, p):
        if not self.training:
            return tensor
        dropout_mask = self.get_dropout_mask(tensor, p)
        dropped = dropout_mask * tensor
        return dropped

    @staticmethod
    def get_dropout_mask(tensor, p):
        mask = torch.bernoulli(torch.empty_like(tensor.data), 1 - p) / (1 - p)
        mask = Variable(mask, requires_grad=False).expand(tensor.size())
        return mask
