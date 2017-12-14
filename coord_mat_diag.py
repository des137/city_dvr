import numpy as np
import numpy.linalg

dim = 12
test_mat = np.zeros((dim, dim))

for m in range(dim):
    for n in range(dim):
        if m == n - 1 or m == n + 1:
            test_mat[m, n] = 0.5
print(test_mat)
EV = np.linalg.eigh(test_mat)[0]
print(np.diff(np.arccos(EV)))