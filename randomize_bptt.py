from numpy import random


def get_bptt_sequence_lengths(
    n,  # data size
    seq,  # mean sequence length
    random_scaling,  # scaling factor for mean sequence length
    p,  # probability that mean sequence length remains unscaled
    s,  # standard deviation
    min_seq,  # minimum sequence length
):
    curr_index = 0
    while True:
        mean = seq
        if random.random() > p:
            mean /= random_scaling
        seq_len = int(random.normal(mean, s))
        seq_len = max(seq_len, min_seq)
        if curr_index + seq_len >= n - 1:
            seq_len = n - 1 - curr_index
            yield curr_index - 1, seq_len, seq_len / seq
            break
        else:
            yield curr_index, seq_len, seq_len / seq
            curr_index += seq_len
            
            
if __name__ == "__main__":
    for i in get_bptt_sequence_lengths(10000, 70, 2, 0.95, 4, 5, 100):
        print(i)
