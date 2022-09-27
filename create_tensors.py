
import numpy as np
from numpy.random import choice

# number of X
i = 3
# number of Y
j = 4
# number of Z
k = 5
#percentage of zeros
p=0.2

def convert_3d_array_2_vector(input_array):
    sum_Z=np.sum(array_3d,axis=0)
    sum_Z = np.sum(sum_Z, axis=0)
    #print(f'sum_Z:{sum_Z}')

    sum_X=np.sum(array_3d,axis=1)
    sum_X = np.sum(sum_X, axis=1)
    #print(f'sum_X:{sum_X}')

    sum_Y=np.sum(array_3d,axis=2)
    sum_Y = np.sum(sum_Y, axis=0)
    #print(f'sum_Y:{sum_Y}')

    output_vector=np.concatenate((sum_X,sum_Y,sum_Z),axis=0)
    #print(sum)
    return output_vector


if __name__ == "__main__":
    N=i*j*k

    array_1d = choice([0, 1], N, p=[1 - p, p])
    print(array_1d)

    array_3d = array_1d.reshape((i, j,k))
    print(array_3d)

    output_vector=convert_3d_array_2_vector(array_3d)
    print(f'output_vector:{output_vector}')

    print(f'onez indices:{np.argwhere(array_3d)}')