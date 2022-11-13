import numpy as np
import open3d

points = np.random.rand(10, 3)


print(points,np.shape(points))
point_cloud = open3d.geometry.PointCloud()
# From numpy to Open3D
point_cloud.points = open3d.utility.Vector3dVector(points)
open3d.visualization.draw_geometries([point_cloud])

"""
# From numpy to Open3D
pcd.points = open3d.utility.Vector3dVector(np_points)

# From Open3D to numpy
np_points = np.asarray(pcd.points)

"""