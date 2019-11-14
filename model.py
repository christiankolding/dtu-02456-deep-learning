import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from weight_drop import WeightDrop

class WD_LSTM(nn.Module):

    def __init__(self, ntoken, ninp, nhid, nlayers, dropout=0.5, weight_drop=0, weight_tying=False):
        super().__init__()
        self.ninp = ninp
        self.nhid = nhid
        self.nlayers = nlayers
        self.weight_tying = weight_tying
        self.drop = nn.Dropout(dropout)
        self.encoder = nn.Embedding(ntoken, ninp)
        self.rnns = nn.ModuleList(
            [WeightDrop(nn.LSTM(self.get_input_size(i), self.get_hidden_size(i)), ["weight_hh_l0"], weight_drop) for i in range(nlayers)]
        )
        self.decoder = nn.Linear(nhid, ntoken)
        if self.weight_tying:
            self.decoder.weight = self.encoder.weight
        self.init_weights()

    def get_input_size(self, layer_num):
        """ The first LSTM layer must match the embedding size. """
        if layer_num == 0:
            return self.ninp
        return self.nhid
        
    def get_hidden_size(self, layer_num):
        """ The hidden size in the last LSTM layer must match the embedding size. """
        if layer_num == self.nlayers - 1:
            return self.ninp
        return self.nhid
        
    def init_weights(self):
        initrange = 0.1
        self.encoder.weight.data.uniform_(-initrange, initrange)
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-initrange, initrange)

    def forward(self, input, hiddens):
        emb = self.drop(self.encoder(input))
        outputs = []
        new_hiddens = []
        for rnn, hidden in zip(self.rnns, hiddens):
            output, new_hidden = rnn(emb if not outputs else outputs[-1], hidden)
            outputs.append(output)
            new_hiddens.append(new_hidden)
        output = self.drop(output)
        decoded = self.decoder(output)
        return decoded, new_hiddens

    def init_hidden(self, bsz):
        weight = next(self.parameters())
        return [
            (
                weight.new_zeros(1, bsz, self.get_hidden_size(i)), 
                weight.new_zeros(1, bsz, self.get_hidden_size(i))
            ) for i in range(self.nlayers)
        ]
