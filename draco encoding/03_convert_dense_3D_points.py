import numpy as np
import open3d
from utilites import create_dense_3D_cloud,convert_dense_cloud_2_sparse_cloud


# number of X
i = 30
# number of Y
j = 40
# number of Z
k = 50
#percentage of zeros
p=0.01


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
    """
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
    """