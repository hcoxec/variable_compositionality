import json
import torch
import editdistance
import numpy as np

from scipy import spatial
from scipy.stats import spearmanr

'''
For comparison with previous work the code below is copied from the Facebook EGG 
codebase for Chaabouni et al. 2020. We compare with the posdis measure as it's most
relevant. The other measure from prior work we compare with is included in the 
scoring code because we needed to create a custom implementation which scaled
beyond just binary strings
'''

def entropy_dict(freq_table):
    H = 0
    n = sum(v for v in freq_table.values())

    for m, freq in freq_table.items():
        p = freq_table[m] / n
        H += -p * np.log(p)
    return H / np.log(2)

def entropy(messages):
    from collections import defaultdict

    freq_table = defaultdict(float)

    for m in messages:
        m = _hashable_tensor(m)
        freq_table[m] += 1.0

    return entropy_dict(freq_table)


def _hashable_tensor(t):
    if isinstance(t, tuple):
        return t
    if isinstance(t, int):
        return t

    try:
        t = t.item()
    except ValueError:
        t = tuple(t.view(-1).tolist())
    return t


def mutual_info(xs, ys):
    e_x = entropy(xs)
    e_y = entropy(ys)

    xys = []

    for x, y in zip(xs, ys):
        xy = (_hashable_tensor(x), _hashable_tensor(y))
        xys.append(xy)

    e_xy = entropy(xys)

    return e_x + e_y - e_xy


def ask_sender(n_attributes, n_values, dataset, sender, device):
    attributes = []
    strings = []
    meanings = []

    for i in range(len(dataset)):
        meaning = dataset[i]

        attribute = meaning.view(n_attributes, n_values).argmax(dim=-1)
        attributes.append(attribute)
        meanings.append(meaning.to(device))

        with torch.no_grad():
            string, *other = sender(meaning.unsqueeze(0).to(device))
        strings.append(string.squeeze(0))

    attributes = torch.stack(attributes, dim=0)
    strings = torch.stack(strings, dim=0)
    meanings = torch.stack(meanings, dim=0)

    return attributes, strings, meanings


def information_gap_representation(meanings, representations):
    gaps = torch.zeros(representations.size(1))
    non_constant_positions = 0.0

    for j in range(representations.size(1)):
        symbol_mi = []
        h_j = None
        for i in range(meanings.size(1)):
            x, y = meanings[:, i], representations[:, j]
            info = mutual_info(x, y)
            symbol_mi.append(info)

            if h_j is None:
                h_j = entropy(y)

        symbol_mi.sort(reverse=True)

        if h_j > 0.0:
            gaps[j] = (symbol_mi[0] - symbol_mi[1]) / h_j
            non_constant_positions += 1

    score = gaps.sum() / non_constant_positions
    return score.item()


def information_gap_position(n_attributes, n_values, dataset, sender, device):
    attributes, strings, _meanings = ask_sender(
        n_attributes, n_values, dataset, sender, device
    )
    return information_gap_representation(attributes, strings)


def histogram(strings, vocab_size):
    batch_size = strings.size(0)

    histogram = torch.zeros(batch_size, vocab_size, device=strings.device)

    for v in range(vocab_size):
        histogram[:, v] = strings.eq(v).sum(dim=-1)

    return histogram


def information_gap_vocab(n_attributes, n_values, dataset, sender, device, vocab_size):
    attributes, strings, _meanings = ask_sender(
        n_attributes, n_values, dataset, sender, device
    )

    histograms = histogram(strings, vocab_size)
    return information_gap_representation(attributes, histograms[:, 1:])


def edit_dist(_list):
    distances = []
    count = 0
    for i, el1 in enumerate(_list[:-1]):
        for j, el2 in enumerate(_list[i + 1 :]):
            count += 1
            # Normalized edit distance (same in our case as length is fixed)
            distances.append(editdistance.eval(el1, el2) / len(el1))
    return distances


def cosine_dist(_list):
    distances = []
    for i, el1 in enumerate(_list[:-1]):
        for j, el2 in enumerate(_list[i + 1 :]):
            distances.append(spatial.distance.cosine(el1, el2))
    return distances


def topographic_similarity(n_attributes, n_values, dataset, sender, device):
    _attributes, strings, meanings = ask_sender(
        n_attributes, n_values, dataset, sender, device
    )
    list_string = []
    for s in strings:
        list_string.append([x.item() for x in s])
    distance_messages = edit_dist(list_string)
    distance_inputs = cosine_dist(meanings.cpu().numpy())

    corr = spearmanr(distance_messages, distance_inputs).correlation
    return corr
