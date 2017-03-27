import numpy as np

patterns = []

target = np.ones((6,6), dtype=bool)
target[1:5, 1:5] = 0
patterns.append(target)

patterns.append(np.array([
    [ 0, 0, 0, 0, 0, 0],
    [ 0, 1, 0, 0, 1, 0],
    [ 0, 0, 0, 0, 0, 0],
    [ 0, 1, 0, 0, 1, 0],
    [ 0, 1, 1, 1, 1, 0],
    [ 0, 0, 0, 0, 0, 0]
], dtype=bool))

patterns.append(np.array([
    [1, 0, 1, 0, 1, 0],
    [0, 0, 1, 0, 0, 0],
    [1, 0, 0, 1, 1, 1],
    [0, 1, 0, 0, 1, 1],
    [0, 1, 1, 0, 0, 1],
    [0, 0, 1, 0, 1, 1]
], dtype=bool))
