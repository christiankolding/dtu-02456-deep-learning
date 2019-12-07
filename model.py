import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from weight_drop import WeightDrop
from variational_dropout import VariationalDropout
from embedding_dropout import embedding_dropout

class WD_LSTM(nn.Module):

    def __init__(self, ntoken, ninp, nhid, nlayers, dropout, dropout_h, dropout_i, dropout_e, weight_drop, weight_tying):
        super().__init__()
        self.ninp = ninp
        self.nhid = nhid
        self.nlayers = nlayers
        self.weight_tying = weight_tying
        self.dropout = dropout
        self.dropout_h = dropout_h
        self.dropout_i = dropout_i
        self.dropout_e = dropout_e
        self.variational_dropout = VariationalDropout()
        self.encoder = nn.Embedding(ntoken, ninp)
        self.rnns = nn.ModuleList(
            [WeightDrop(nn.LSTM(self.get_input_size(i), self.get_hidden_size(i)), ["weight_hh_l0"], weight_drop) for i in range(nlayers)]
        )
        self.decoder = nn.Linear(ninp if weight_tying else nhid, ntoken)
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
            return self.ninp if self.weight_tying else self.nhid
        return self.nhid
        
    def init_weights(self):
        initrange = 0.1
        self.encoder.weight.data.uniform_(-initrange, initrange)
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-initrange, initrange)

    def forward(self, input, hiddens):
        emb = embedding_dropout(self.encoder, input, self.dropout_e if self.training else 0)
        emb = self.variational_dropout(emb, self.dropout_i)
        
        outputs = []
        dropped_outputs = []
        new_hiddens = []
        # LSTM module has been split up, since weight tying requires different hidden sizes.
        # Therefore we need to manage the forward propagation ourselves.
        output = None
        for layer_num, (rnn, hidden) in enumerate(zip(self.rnns, hiddens)):
            output, new_hidden = rnn(emb if output is None else output, hidden)
            outputs.append(output)
            if layer_num != self.nlayers - 1:  # Variational dropout on the recurrent layers
                output = self.variational_dropout(output, self.dropout_h)
                dropped_outputs.append(output)
            new_hiddens.append(new_hidden)
        output = self.variational_dropout(output, self.dropout)
        dropped_outputs.append(output)
        decoded = self.decoder(output)
        return decoded, new_hiddens, outputs, dropped_outputs

    def init_hidden(self, bsz):
        weight = next(self.parameters())
        return [
            (
                weight.new_zeros(1, bsz, self.get_hidden_size(i)), 
                weight.new_zeros(1, bsz, self.get_hidden_size(i))
            ) for i in range(self.nlayers)
        ]
