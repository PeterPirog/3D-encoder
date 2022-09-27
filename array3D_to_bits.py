import tensorflow as tf
import numpy as np
from numpy.random import choice

# number of X
i = 3
# number of Y
j = 4
# number of Z
k = 5
#percentage of zeros
p=0.5

if __name__ == "__main__":
    N=i*j*k

    array_1d = choice([0, 1], N, p=[1 - p, p])
    print(array_1d)

    array_3d = array_1d.reshape((i, j,k))
    print(array_3d)

    vectorC=np.ndarray.flatten(array_3d,order='C')
    print(f'vector:{vectorC}')

    vectorF=np.ndarray.flatten(array_3d,order='F')
    print(f'vector:{vectorF}')
    vectorA=np.ndarray.flatten(array_3d,order='A')
    print(f'vector:{vectorA}')

    