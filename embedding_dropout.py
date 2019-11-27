import torch
import torch.nn.functional as F


def embedding_dropout(embedding, words, p):
    dropout_mask = torch.Tensor(
        embedding.num_embeddings, 1
    ).to(
        embedding.weight.device
    ).bernoulli(1 - p).expand_as(embedding.weight) / (1 - p)
    masked_embed_weight = dropout_mask * embedding.weight
    dropped = F.embedding(words, masked_embed_weight)
    return dropped
