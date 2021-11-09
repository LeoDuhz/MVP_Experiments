import open3d as o3d
import teaserpp_python
import numpy as np
import copy
from helpers import *
import h5py
from scipy.spatial.transform import Rotation as R
from icecream import ic as print


filename = "/home/leodu/h52ply/MVP_Benchmark/Registration/MVP_Test_RG.h5"


def visualize(pointcloud, pointcloud2=None, num=0):
    point_cloud = o3d.geometry.PointCloud()
    # point_cloud.points = open3d.utility.Vector3dVector(pointcloud[:,0:3].reshape(-1,3))
    point_cloud.points = o3d.utility.Vector3dVector(pointcloud.points)
    point_cloud.paint_uniform_color([1, 0, 0])
    # if(pointcloud2 != None):
    point_cloud2 = o3d.geometry.PointCloud()
    # point_cloud.points = open3d.utility.Vector3dVector(pointcloud[:,0:3].reshape(-1,3))
    point_cloud2.points = o3d.utility.Vector3dVector(pointcloud2.points)
    point_cloud2.paint_uniform_color([0, 1, 0])

    o3d.visualization.draw_geometries(
        [point_cloud, point_cloud2], width=800, height=600)
    # else:
    #     open3d.visualization.draw_geometries([point_cloud],width=800,height=600)
    return point_cloud


def generate_transformation(zyz, translation):
    rotation = R.from_euler('zyz', zyz, degrees=True)
    rotation_matrix = np.asarray(rotation.as_matrix())
    translation_matrix = np.asarray(translation)
    tranformation_matrix = np.empty((4, 4))
    tranformation_matrix[:3, :3] = rotation_matrix
    tranformation_matrix[:3, 3] = translation_matrix
    tranformation_matrix[3, :] = [0, 0, 0, 1]
    return tranformation_matrix


class readFile:
    src = None
    tgt = None
    complete = None
    local = None
    sdf = None
    pc_from_mesh = None

    def __init__(self, filename):
        if readFile.src is None:
            readFile.src = self.readPCL(filename, "src")
            readFile.tgt = self.readPCL(filename, "tgt")
            readFile.complete = self.readPCL(filename, "complete")
            # readFile.local = o3d.io.read_point_cloud('./data/particles/sun3d-hotel_uc-scan3/cloud_bin_2.ply')
            readFile.sdf, readFile.pc_from_mesh = None, None
            # readFile.sdf, readFile.pc_from_mesh = load_sdf_pcd('../data/particles/mesh/cat.obj')

    def readPCL(self, filename, dataType):
        f = h5py.File(filename, 'r')
        if dataType == "src":
            pointclouds = np.array(f['src'][:].astype('float32'))
        elif dataType == "tgt":
            pointclouds = np.array(f['tgt'][:].astype('float32'))
        elif dataType == "complete":
            pointclouds = np.array(f['complete'][:].astype('float32'))
        return pointclouds


def get_angular_error(R_exp, R_est):
    """
    Calculate angular error
    """
    return abs(np.arccos(min(max(((np.matmul(R_exp.T, R_est)).trace() - 1) / 2, -1.0), 1.0)))


pointcloud_read = readFile(filename)
src_cloud = pointcloud_read.src
tgt_cloud = pointcloud_read.tgt
complete_cloud = pointcloud_read.complete

random_pair_num = np.random.randint(0, 1000)
cloud = o3d.geometry.PointCloud()
cloud.points = o3d.utility.Vector3dVector(copy.deepcopy(
    src_cloud[random_pair_num, :, :].reshape(-1, 3)))
transformed_cloud = o3d.geometry.PointCloud()
transformed_cloud.points = o3d.utility.Vector3dVector(
    copy.deepcopy(tgt_cloud[random_pair_num, :, :].reshape(-1, 3)))
'''To move the center of the input pointcloud to the center of coordinate'''
center_of_cloud = np.array([(np.max(np.asarray(cloud.points)[:, 0])+np.min(np.asarray(cloud.points)[:, 0]))/2,
                            (np.max(np.asarray(cloud.points)[
                             :, 1])+np.min(np.asarray(cloud.points)[:, 1]))/2,
                            (np.max(np.asarray(cloud.points)[:, 2])+np.min(np.asarray(cloud.points)[:, 2]))/2])
cloud = copy.deepcopy(cloud).translate(-center_of_cloud)
transformed_cloud = copy.deepcopy(
    transformed_cloud).translate(-center_of_cloud)

print('length of src: ', len(cloud.points))
print('length of src: ', len(transformed_cloud.points))

'''
generate the rotation and translation randomly
'''
zyz = np.random.randint(0, 120, 3).tolist()
# translation = np.random.randint(-1,1,3).tolist()
# np.array([0,0,0]).tolist()
translation = np.random.uniform(-1, 1, [1, 3]).tolist()
scale = 1
T = generate_transformation(zyz, translation)
pcd_transformed = transformed_cloud
pcd_transformed.transform(T)
transformed_pointcloud_array = np.asarray(pcd_transformed.points)

VOXEL_SIZE = 0.5
VISUALIZE = True

# Load and visualize two point clouds from 3DMatch dataset
A_pcd_raw = cloud
B_pcd_raw = transformed_cloud
A_pcd_raw.paint_uniform_color([0.0, 0.0, 1.0])  # show A_pcd in blue
B_pcd_raw.paint_uniform_color([1.0, 0.0, 0.0])  # show B_pcd in red
if VISUALIZE:
    o3d.visualization.draw_geometries([A_pcd_raw, B_pcd_raw])  # plot A and B

# voxel downsample both clouds
A_pcd = A_pcd_raw.voxel_down_sample(voxel_size=VOXEL_SIZE)
B_pcd = B_pcd_raw.voxel_down_sample(voxel_size=VOXEL_SIZE)

print('length of src: ', len(A_pcd.points))
print('length of src: ', len(B_pcd.points))

# A_pcd = A_pcd_raw
# B_pcd = B_pcd_raw

if VISUALIZE:
    o3d.visualization.draw_geometries(
        [A_pcd, B_pcd])  # plot downsampled A and B

A_xyz = pcd2xyz(A_pcd)  # np array of size 3 by N
B_xyz = pcd2xyz(B_pcd)  # np array of size 3 by M

# extract FPFH features
A_feats = extract_fpfh(A_pcd, VOXEL_SIZE)
B_feats = extract_fpfh(B_pcd, VOXEL_SIZE)

# establish correspondences by nearest neighbour search in feature space
# corrs_A, corrs_B = find_correspondences(
#     A_feats, B_feats, mutual_filter=True)
# corrs_A = np.asarray(np.ones(A_xyz.shape[1]*B_xyz.shape[1]))
# corrs_B = np.asarray(np.ones(A_xyz.shape[1]*B_xyz.shape[1]))

corrs_A = []
corrs_B = []

for i in range(A_xyz.shape[1]):
    for j in range(B_xyz.shape[1]):
        # corrs_A[i*B_xyz.shape[1]+j] = i
        # corrs_B[i*B_xyz.shape[1]+j] = j

        corrs_A.append(i)
        corrs_B.append(j)


A_corr = A_xyz[:, corrs_A]  # np array of size 3 by num_corrs
B_corr = B_xyz[:, corrs_B]  # np array of size 3 by num_corrs

num_corrs = A_corr.shape[1]
print(f'Brute-Force generates {num_corrs} putative correspondences.')

# visualize the point clouds together with feature correspondences
points = np.concatenate((A_corr.T, B_corr.T), axis=0)
lines = []
for i in range(num_corrs):
    lines.append([i, i+num_corrs])
colors = [[0, 1, 0] for i in range(len(lines))]  # lines are shown in green
line_set = o3d.geometry.LineSet(
    points=o3d.utility.Vector3dVector(points),
    lines=o3d.utility.Vector2iVector(lines),
)
line_set.colors = o3d.utility.Vector3dVector(colors)
# o3d.visualization.draw_geometries([A_pcd, B_pcd, line_set])

# robust global registration using TEASER++
NOISE_BOUND = VOXEL_SIZE
teaser_solver = get_teaser_solver(NOISE_BOUND)
teaser_solver.solve(A_corr, B_corr)
solution = teaser_solver.getSolution()
R_teaser = solution.rotation
t_teaser = solution.translation
T_teaser = Rt2T(R_teaser, t_teaser)

# Visualize the registration results
A_pcd_T_teaser = copy.deepcopy(A_pcd).transform(T_teaser)
o3d.visualization.draw_geometries([A_pcd_T_teaser, B_pcd])

# local refinement using ICP
icp_sol = o3d.pipelines.registration.registration_icp(
    A_pcd, B_pcd, NOISE_BOUND, T_teaser,
    o3d.pipelines.registration.TransformationEstimationPointToPoint(),
    o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=100))
T_icp = icp_sol.transformation

# visualize the registration after ICP refinement
A_pcd_T_icp = copy.deepcopy(A_pcd).transform(T_icp)
o3d.visualization.draw_geometries([A_pcd_T_icp, B_pcd])

print("Expected rotation: ")
print(T[:3, :3])
print("Estimated rotation: ")
print(solution.rotation)
print("Error (rad): ")
print(get_angular_error(T[:3, :3], T_icp[:3, :3]))


print("Expected translation: ")
print(T[:3, 3])
print("Estimated translation: ")
print(solution.translation)
print("Error (m): ")
print(np.linalg.norm(T[:3, 3] - T_icp[:3, 3]))

# print("Number of correspondences: ", N)
# print("Number of outliers: ", N_OUTLIERS)
# print("Time taken (s): ", end - start)
