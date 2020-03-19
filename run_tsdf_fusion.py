#!/scratch_net/nudel/esandstroem/venvs/tsdf_fusion_env/bin/python
import os
app_path = '/scratch_net/nudel/esandstroem/venvs/tsdf_fusion_env/bin'
os.environ["PATH"] = app_path + os.pathsep + os.environ["PATH"]

from TSDFHandle import *
import numpy as np
import cv2
from utils import extract_mesh_marching_cubes
from visualization import plot_mesh
import plyfile
from sys import argv
import pathlib

if (len(argv) < 3):
	print('Usage: {0} <name of depth directory> <save mode>'.format(argv[0]))
	exit(0)


CURRENT_DIR = str(pathlib.Path().absolute())
depth_path = CURRENT_DIR + '/' + argv[1]
campose_path = CURRENT_DIR + '/' + 'left_camera_matrix'


box = np.array([[-4,4],[-4,4],[-4,4]])  # each cell depicts the interval where we will reconstruct the shape i.e.
# [[-xmin,xmax],[-ymin,ymax],[-zmin,zmax]]
tsdf = TSDF(bbox=box, resolution=0.025, resolution_factor=1)

depth_dir = os.listdir(depth_path)
sortOrder_depth = [int(x[:-4]) for x in depth_dir]
depth_dir = [x for _, x in sorted(zip(sortOrder_depth, depth_dir))]

campose_dir = os.listdir(campose_path)
sortOrder_pose = [int(x[:-4]) for x in campose_dir]
campose_dir = [x for _, x in sorted(zip(sortOrder_pose, campose_dir))]

camera_intrinsics = np.array([[256, 0, 256], [0, 256, 256], [0, 0, 1]]).astype(np.float32)
# apparently, the tsdf fusion code expects that the camera coordinate system is such that z is in the
# camera viewing direction, y is down and x is to the right. This is achieved by a serie of rotations
rot_180_around_y = np.array([[-1, 0, 0], [0, 1, 0], [0, 0, -1]]).astype(np.float32)
rot_180_around_z = np.array([[-1, 0, 0], [0, -1, 0], [0, 0, 1]]).astype(np.float32)
rotation = np.matmul(rot_180_around_z, rot_180_around_y)

for i in range(len(depth_dir)):
	depth = cv2.imread(depth_path + '/' + depth_dir[i], -1)
	depth = depth / 1000
	weight_map = np.ones(depth.shape)
	campose = np.linalg.inv(np.loadtxt(campose_path + '/' + campose_dir[i]).astype(np.float32))
	campose = np.matmul(camera_intrinsics, np.matmul(rotation,campose[0:3, 0:4]))
	tsdf.fuse(campose, depth.astype(np.float32), weight_map.astype(np.float32))


mesh = extract_mesh_marching_cubes(tsdf.get_volume()[:, :, :, 0])
if argv[2]:
	mesh.write('tsdf_fusion_' + argv[1] + '.ply')
plot_mesh(mesh)
