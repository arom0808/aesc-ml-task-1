import numpy as np


def prod_non_zero_diag(x):
    return np.prod(np.diag(x)[np.diag(x) != 0])


def are_multisets_equal(x, y):
    return np.array_equal(np.sort(x), np.sort(y))


def max_after_zero(x):
    return np.max(x[1:][(x == 0)[:-1]])


def convert_image(img, coefs):
    return np.dot(img, coefs).astype(np.uint8)


def run_length_encoding(x):
    if x.size == 0:
        return np.array([]), np.array([])
    start_ids = np.concatenate(([0], np.where(x[:-1] != x[1:])[0] + 1))
    run_lengths = np.diff(np.concatenate((start_ids, [x.size])))
    return x[start_ids], run_lengths


def pairwise_distance(X, Y):
    return np.sqrt(np.sum((X[:, np.newaxis] - Y) ** 2, axis=-1))
