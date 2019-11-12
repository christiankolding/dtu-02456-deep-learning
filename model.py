import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from weight_drop import WeightDrop

class WD_LSTM(nn.Module):

    def __init__(self, ntoken, ninp, nhid, nlayers, dropout=0.5, weight_drop=0):
        super().__init__()
        self.drop = nn.Dropout(dropout)
        self.encoder = nn.Embedding(ntoken, ninp)
        self.rnn = WeightDrop(
            nn.LSTM(ninp, nhid, nlayers, dropout=0),
            [f"weight_hh_l{i}" for i in range(nlayers)],
            weight_drop
        )
        self.decoder = nn.Linear(nhid, ntoken)
        self.init_weights()
        self.nhid = nhid
        self.nlayers = nlayers

    def init_weights(self):
        initrange = 0.1
        self.encoder.weight.data.uniform_(-initrange, initrange)
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-initrange, initrange)

    def forward(self, input, hidden):
        emb = self.drop(self.encoder(input))
        output, hidden = self.rnn(emb, hidden)
        output = self.drop(output)
        decoded = self.decoder(output)
        return decoded, hidden

    def init_hidden(self, bsz):
        weight = next(self.parameters())
        return (weight.new_zeros(self.nlayers, bsz, self.nhid),
                weight.new_zeros(self.nlayers, bsz, self.nhid))
