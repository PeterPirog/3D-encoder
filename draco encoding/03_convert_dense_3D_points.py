import numpy as np
import open3d
from numpy.random import choice

# number of X
i = 30
# number of Y
j = 40
# number of Z
k = 50
#percentage of zeros
p=0.01


def create_dense_3D_cloud(x_size,y_size,z_size,p_of_zeros=0.5):
    p=p_of_zeros
    N=x_size*y_size*z_size
    array_1d = choice([0, 1], N, p=[1 - p, p])
    array_3d = array_1d.reshape((i, j,k))
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
            sparse_cloud=np.vstack((expanded_array,sparse_cloud))
        else:
            print('Number of points in the cloud is greater than expanded number of points')
    return sparse_cloud



if __name__ == "__main__":
    # create dense 3D point cloud where 1=point, 0=empty space
    dense_cloud=create_dense_3D_cloud(x_size=i,y_size=j,z_size=k,p_of_zeros=p)
    print(dense_cloud)
    # convert to sparse cloud where each point has x,y,z normalized index
    sparse_cloud=convert_dense_cloud_2_sparse_cloud(dense_cloud,normalized_indexes=False,expand_to_number_of_points=625)
    print(sparse_cloud)


    #visualize point cloud
    point_cloud = open3d.geometry.PointCloud()
    # From numpy to Open3D
    point_cloud.points = open3d.utility.Vector3dVector(sparse_cloud)
    print(point_cloud.points)
    open3d.visualization.draw_geometries([point_cloud])

    # Draco encoding
    import DracoPy
    # https://github.com/seung-lab/DracoPy/blob/master/src/DracoPy.pyx#L130
    binary = DracoPy.encode(point_cloud.points,faces=None,
        quantization_bits=11, compression_level=1,
        quantization_range=-1, quantization_origin=None,
        create_metadata=False, preserve_order=False,
        colors=None)
    #print(binary)

    mesh = DracoPy.decode(binary)
    #print(mesh.points, np.shape(mesh.points))