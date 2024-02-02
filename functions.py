from collections import Counter
import math


def prod_non_zero_diag(x):
    ans = 1
    for i in range(min(len(x), len(x[0]))):
        if x[i][i] != 0:
            ans *= x[i][i]
    return ans


def are_multisets_equal(x, y):
    return Counter(x) == Counter(y)


def max_after_zero(x):
    return max(x[i] for i in range(1, len(x)) if x[i - 1] == 0)


def convert_image(img, coefs):
    return [[sum([int(coefs[i] * pix[i]) for i in range(3)]) for pix in row] for row in img]


def run_length_encoding(x):
    a, b = [], []
    if len(x) == 0:
        return a, b
    a.append(x[0])
    b.append(1)
    for i in range(1, len(x)):
        if x[i] != x[i - 1]:
            a.append(x[i])
            b.append(1)
        else:
            b[-1] += 1
    return a, b


def pairwise_distance(X, Y):
    m, n = len(X), len(Y)
    distances = [[0] * n for _ in range(m)]
    for i in range(m):
        for j in range(n):
            distances[i][j] = math.sqrt(sum((x - y) ** 2 for x, y in zip(X[i], Y[j])))
    return distances
