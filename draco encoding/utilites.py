

import numpy as np
from numpy.random import choice

def create_dense_3D_cloud(x_size,y_size,z_size,p_of_zeros=0.5):
    p=p_of_zeros
    N=x_size*y_size*z_size
    array_1d = choice([0, 1], N, p=[1 - p, p])
    array_3d = array_1d.reshape((x_size, y_size,z_size))
    return array_3d

def convert_dense_cloud_2_sparse_cloud(dense_cloud,normalized_indexes=True,expand_to_number_of_points=None):
    (x_size, y_size, z_size) = np.shape(dense_cloud)
    (x_indexes,y_indexes,z_indexes) = np.where(dense_cloud == 1)
    number_of_points=len(x_indexes)
    print(number_of_points)
    if normalized_indexes:
        x_indexes=x_indexes/x_size
        y_indexes = y_indexes / y_size
        z_indexes = z_indexes / z_size
    sparse_cloud=np.vstack((x_indexes,y_indexes,z_indexes)).T
    if expand_to_number_of_points is not None:
        if expand_to_number_of_points>number_of_points:
            expanded_array=np.zeros((expand_to_number_of_points-number_of_points,3))
            print(expanded_array)
            sparse_cloud=np.vstack((sparse_cloud,expanded_array))
        else:
            print('Number of points in the cloud is greater than expanded number of points')
    return sparse_cloud
