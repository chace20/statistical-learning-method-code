import numpy as np

data = np.array([[1, 'S', -1], [1, 'M', -1], [1, 'M', 1], [1, 'S', 1], [1, 'S', -1],
          [2, 'S', -1], [2, 'M', -1], [2, 'M', 1], [2, 'L', 1], [2, 'L', 1],
          [3, 'L', 1], [3, 'M', 1], [3, 'M', 1], [3, 'L', 1], [3, 'L', -1]])

X = [2, 'S']


def fit(data):
    pre = []
    pre.append(data[:, 2].count(1)/len(data))
    pre.append(data[:, 2].count(-1)/len(data))
    matrix = np.zeros([2, 2])


