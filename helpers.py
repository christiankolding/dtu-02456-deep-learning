import yaml
import torch


class Config:
    """ Struct-like object based on a first level lookup in the config file. """
    
    def __init__(self, name, config_file_path='config.yml'):
        with open(config_file_path, 'r') as f:
            self.__dict__.update(**yaml.load(f)[name])

            
def repackage_hidden(h):
    """ Wraps hidden states in new Tensors to detach them from their history. """
    if isinstance(h, torch.Tensor):
        return h.detach()
    else:
        return tuple(repackage_hidden(v) for v in h)


def batchify(data, bsz, device):
    # Work out how cleanly we can divide the dataset into bsz parts.
    nbatch = data.size(0) // bsz
    # Trim off any extra elements that wouldn't cleanly fit (remainders).
    data = data.narrow(0, 0, nbatch * bsz)
    # Evenly divide the data across the bsz batches.
    data = data.view(bsz, -1).t().contiguous()
    return data.to(device)


def get_batch(source, i, seq_len):
    seq_len = min(seq_len, len(source) - 1 - i)
    data = source[i:i+seq_len]
    target = source[i+1:i+1+seq_len].view(-1)
    return data, target
