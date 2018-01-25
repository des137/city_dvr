from contextlib import contextmanager
from itertools import product
import numpy as np
import time
from mpi4py import MPI


@contextmanager
def benchmark(name):
    start = time.time()
    yield
    end = time.time()

    print('{} took {:.2f} ms'.format(name, (end - start) * 1000.0))


comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()

dim = int(1e1 + 1)
dv = np.arange(dim)

dvini = rank * int(dim / size)
if rank == size - 1:
    dvend = dim
else:
    dvend = dvini + int(dim / size)
print('cpu%d: %d %d' % (rank, dvini, dvend))

mat = np.zeros((dim, dim))
vec = np.zeros(dim**2)

#with benchmark("cpu%d: array looping" % rank):
#    r = 0
#    for m in dv[dvini:dvend]:
#        c = 0
#        for n in dv:
#            mat[c, r] = m * n
#            c += 1
#        r += 1
#    #print(mat)

dvini = rank * int(dim**2 / size)
if rank == size - 1:
    dvend = dim**2
else:
    dvend = dvini + int(dim**2 / size)
print('cpu%d: %d %d' % (rank, dvini, dvend))
with benchmark("cpu%d: vector looping" % rank):
    c = 0
    for m in list(product(dv, dv))[dvini:dvend]:
        vec[c] = m[0] * m[1]
        c += 1
    #print(np.reshape(vec, (dim, dim)))
