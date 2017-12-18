import numpy as np
from scipy.sparse import kronsum

dim = 2
a = np.zeros((dim, dim))
b = np.zeros((dim, dim))

for i in range(dim):
    for ip in range(dim):
        if i == ip + 1:
            a[i, ip] = -1
            b[i, ip] = -1
        if i == ip - 1:
            a[i, ip] = +1
            b[i, ip] = +1
        if i == ip:
            a[i, ip] = 1.1
            b[i, ip] = 1.1

full2 = kronsum(a, a).todense()
full3 = kronsum(full2, a).todense()
print(full2)
print(full3)