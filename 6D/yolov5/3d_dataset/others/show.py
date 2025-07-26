import json
import math
import random
import shutil
import sys
import cv2
import pickle as pkl
import time
from tqdm import tqdm
from transforms3d.quaternions import mat2quat


def read_pickle(pkl_path):
	with open(pkl_path, "rb") as f:
		return pkl.load(f)


# coor_load = read_pickle('../dataset/lm/imgn/benchvise/000025_0-coor.pkl')
# c = cv2.imread('../dataset/lm/imgn/ape/000005_0-color.png')
# # e = cv2.imread('../dataset/lm/imgn/ape/000005_0-label.png')
# # d = c[coor_load['u']:coor_load['u']+coor_load['h'],coor_load['l']:coor_load['l']+coor_load['w'],:]
# # cv2.namedWindow('a', cv2.WINDOW_NORMAL)
# # cv2.namedWindow('d', cv2.WINDOW_NORMAL)
# # cv2.namedWindow('e', cv2.WINDOW_NORMAL)
# a = coor_load['coor']
# cv2.imshow('a', a)
# # cv2.imshow('d', d)
# # cv2.imshow('e', e*255.)
# cv2.waitKey()


import numpy as np
from PIL import Image

# ==============================================================================
#                                                                   SCALE_TO_255
# ==============================================================================

from plyfile import PlyData
import glob
import os
from scipy.spatial import cKDTree
from sklearn.neighbors import KDTree
from scipy.spatial import ConvexHull


def load_ply_vtx(pth):
	"""
    load object vertices
    :param pth: str
    :return: pts: (N, 3)
    """
	ply = PlyData.read(pth)
	vtx = ply['vertex']
	pts = np.stack([vtx['x'], vtx['y'], vtx['z']], axis=-1)
	ptsx = np.stack([vtx['x'], vtx['x']], axis=-1)
	ptsy = np.stack([vtx['y'], vtx['y']], axis=-1)
	ptsz = np.stack([vtx['z'], vtx['z']], axis=-1)
	print('min_x: ', 1000 * ptsx.min())
	print('min_y: ', 1000 * ptsy.min())
	print('min_z: ', 1000 * ptsz.min())
	print('size_x: ', 1000 * (ptsx.max() - ptsx.min()))
	print('size_y: ', 1000 * (ptsy.max() - ptsy.min()))
	print('size_z: ', 1000 * (ptsz.max() - ptsz.min()))

	def compute_diameter(point_cloud):
		# 计算点集凸包
		hull = ConvexHull(point_cloud)
		# 从凸包中选择两个最远的点
		max_dist = 0
		for i in range(len(hull.vertices)):
			for j in range(i + 1, len(hull.vertices)):
				dist = np.linalg.norm(point_cloud[hull.vertices[i]] - point_cloud[hull.vertices[j]])
				if dist > max_dist:
					max_dist = dist
		return max_dist

	print(compute_diameter(pts))

	return pts


# obj = 'squirrel'
# load_ply_vtx('E:\DL_Projects\EPro_PnP_main\EPro_PnP_6DoF\dataset\lm\models\\' + obj + '\\' + obj + '.ply')

def get_model_info():
	objs = ['ape', 'benchvise', 'camera', 'can', 'cat',
	        'driller', 'duck', 'eggbox', 'glue', 'holepuncher',
	        'iron', 'lamp', 'phone', 'AR_QC', 'squirrel']
	for obj in objs:
		print(obj)
		load_ply_vtx('E:\DL_Projects\EPro_PnP_main\EPro_PnP_6DoF\dataset\lm\models\\' + obj + '\\' + obj + '.ply')


# get_model_info()
# allFiles = ['ape', 'benchvise', 'camera', 'can', 'cat',
#             'driller', 'duck', 'eggbox', 'glue', 'holepuncher',
#             'iron', 'lamp', 'phone', 'AR_QC']
#
# all_obj = sorted(glob.glob('E:\DL_Projects\EPro_PnP_main\EPro_PnP_6DoF\dataset\lm\models\*\*.ply'),
#                  key=lambda filename: allFiles.index(os.path.splitext(os.path.basename(filename))[0]))
# with open('E:\DL_Projects\EPro_PnP_main\EPro_PnP_6DoF\dataset\lm\models\models_info.txt', encoding='utf-8') as file:
#     model_info = file.read()
# #     file.close()
#
# objs = []
# for i, j in zip(all_obj, range(len(all_obj))):
#     if j < 13:
#         objs.append(load_ply_vtx(i) * 1000)
#     else:
#         objs.append(load_ply_vtx(i))
#
# # x3d = np.ones((4, point_cloud.shape[0]), dtype=np.float32)
# # x3d[0, :] = point_cloud[:, 0]
# # x3d[1, :] = point_cloud[:, 1]
# # x3d[2, :] = point_cloud[:, 2]
#
# # point_cloud = np.matmul(pose3, x3d)
# # out = point_project(point_cloud, para=[0, 0, 1, 0]
#
# pointcloud2image(objs[13])

import open3d as o3d
import copy


def pose_show(pose=None, obj='benchvise', id='0003'):
	obj = 'AR_QC'

	point_cloud_file = "E:\DL_Projects\EPro_PnP_main\EPro_PnP_6DoF\dataset\lm\models\\" + obj + "\\" + obj + ".ply"
	pcd = o3d.io.read_point_cloud(point_cloud_file)

	# point_cloud_file = "E:\DL_Projects\EPro_PnP_main\EPro_PnP_6DoF\dataset\lm\models\\" + obj + "\\" + obj + ".obj"
	# pcd = o3d.io.read_triangle_mesh(point_cloud_file)

	# point_cloud_file = "C:\\Users\\win10\\Desktop\\AR_QC.ply"
	# pcd = o3d.io.read_point_cloud(point_cloud_file)

	# aabb = pcd.get_axis_aligned_bounding_box()
	# print(aabb)
	# pcd.transform([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])
	# pcd.transform([[0, 1, 0, 0], [-1, 0, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])
	# pcd.paint_uniform_color([0.1, 0.1, 0.7])
	if pose is None:
		with open(
				'E:\DL_Projects\EPro_PnP_main\EPro_PnP_6DoF\dataset\lm\\imgn\\' + obj + '\\00' + id + '-pose.txt') as posefile:
			# with open(r'C:\Users\win10\Desktop\p2\output\9.txt') as posefile:
			pose_content = posefile.read()
			posefile.close()
		pose_gt = pose_content.split()
		for i, j in zip(pose_gt, range(len(pose_gt))):
			pose_gt[j] = float(i)
		if len(pose_gt) == 12:
			# pose_gt = pose.reshape(3, 4)
			# pose_gt = np.concatenate((pose_gt, [[0, 0, 0, 1]]), axis=0)
			pose_gt.append(0.0)
			pose_gt.append(0.0)
			pose_gt.append(0.0)
			pose_gt.append(1.0)
		pose_gt = np.array(pose_gt).reshape(4, 4)
	else:
		pose_gt = pose.reshape(3, 4)
		pose_gt = np.concatenate((pose_gt, [[0, 0, 0, 1]]), axis=0)

	# def convert_to_right_handed(RT_matrix):
	#     # 沿z轴反转平移向量
	#     RT_matrix[0, 3] *= -1
	#     RT_matrix[1, 3] *= -1
	#     RT_matrix[2, 3] *= -1
	#
	#     # 将旋转矩阵中的第三列向量取反
	#     RT_matrix[0:3, 2] *= -1
	#
	#     # 反转第三行，平移向量和旋转矩阵中的第三列向量的符号均取反
	#     RT_matrix[2, 0] *= -1
	#     RT_matrix[2, 1] *= -1
	#     RT_matrix[2, 2] *= -1
	#     RT_matrix[2, 3] *= -1
	#
	#     return RT_matrix
	# pose_gt = convert_to_right_handed(pose_gt)

	# pose_gt[0, 3] *= 1000
	# pose_gt[1, 3] *= 1000
	# pose_gt[2, 3] *= 1000
	#     [0.0, 0.0, 0.0, 1.0]]
	# x90 = np.array([[1, 0, 0, 0],
	#                 [0, 0, -1, 0],
	#                 [0, 1, 0, 0],
	#                 [0, 0, 0, 1]])
	# y90 = np.array([[0, 0, 1, 0],
	#                 [0, 1, 0, 0],
	#                 [-1, 0, 0, 0],
	#                 [0, 0, 0, 1]])
	# z90 = np.array([[0, -1, 0, 0],
	#                 [1, 0, 0, 0],
	#                 [0, 0, 1, 0],
	#                 [0, 0, 0, 1]])
	# pcd_T = copy.deepcopy(pcd)
	# pcd_T.transform(pose_gt)
	# pcd_T.transform(x90)
	# pcd_T.transform(x90)
	# pcd_T.paint_uniform_color([0, 1, 0])
	# print(pcd.get_center())
	# print(pcd_T.get_center())

	# a = cv2.imread('E:\DL_Projects\EPro_PnP_main\EPro_PnP_6DoF\dataset\lm\\imgn\\' + obj + '\\00' + id + '-color.png')
	# cv2.imshow('a', a)
	load_view_point(pcd, pose_gt, id)


# cv2.waitKey()
# o3d.visualization.draw_geometries(geometry_list=[pcd, pcd_T], window_name=obj, width=640, height=480)


def load_view_point(pcd, pose_gt, s):
	vis = o3d.visualization.Visualizer()
	vis.create_window(window_name='pose_est', width=640, height=480)
	vis.add_geometry(pcd)

	ctr = vis.get_view_control()
	# get default camera parameters
	param = o3d.camera.PinholeCameraParameters()
	param.intrinsic.intrinsic_matrix = [[572.4114, 0, 325.2611], [0, 573.57043, 242.04899], [0, 0, 1]]
	# param.intrinsic.intrinsic_matrix = [[943.2, 0, 639.5], [0, 943.2, 359.5], [0, 0, 1]]

	render_option = vis.get_render_option()
	render_option.background_color = np.array([0, 0, 0])  # 设置背景为黑色
	render_option.point_size = 2.5

	param.extrinsic = pose_gt

	# set param to current camera control view
	ctr.convert_from_pinhole_camera_parameters(param, allow_arbitrary=True)
	vis.update_geometry(pcd)
	vis.poll_events()
	vis.update_renderer()

	# vis.capture_screen_image('./AR_QC/' + s + '-color.png')
	# vis.capture_screen_image('E:\DL_Projects\EPro_PnP_main\EPro_PnP_6DoF\dataset\lm\\imgn\\' + 'AR_QC_data_applend' + '\\00' + s + '-color.png')

	vis.run()


# vis.destroy_window()


from shutil import copyfile


def render_data():
	obj = 'AR_QC'
	point_cloud_file = "E:\DL_Projects\EPro_PnP_main\EPro_PnP_6DoF\dataset\lm\models\\" + obj + "\\" + obj + ".ply"
	pcd = o3d.io.read_point_cloud(point_cloud_file)
	ids = glob.glob("E:\DL_Projects\EPro_PnP_main\EPro_PnP_6DoF\dataset\lm\\imgn\\duck\\*-pose.txt")
	for each in ids:
		copyfile(each,
		         'E:\DL_Projects\EPro_PnP_main\EPro_PnP_6DoF\dataset\lm\\imgn\\' + 'AR_QC_data_applend' + '\\00' + str(
			         9718 + int(each[-15:-11]) + int(each[-10:-9])) + '-pose.txt')
		with open(each, 'r') as posefile:
			pose_content = posefile.read()
			posefile.close()
		pose_gt = pose_content.split()
		for i, j in zip(pose_gt, range(len(pose_gt))):
			pose_gt[j] = float(i)
		pose_gt.append(0.0)
		pose_gt.append(0.0)
		pose_gt.append(0.0)
		pose_gt.append(1.0)
		pose_gt = np.array(pose_gt).reshape(4, 4)
		load_view_point(pcd, pose_gt, str(9718 + int(each[-15:-11]) + int(each[-10:-9])))
		print(each)


def mask_find_bboxs(mask):
	retval, labels, stats, centimgds = cv2.connectedComponentsWithStats(mask, connectivity=8)  # connectivity参数的默认值为8
	stats = stats[stats[:, 4].argsort()]
	return stats[:-1]  # 排除最外层的连通图


def get_box():
	ids = glob.glob(r'E:\DL_Projects\EPro_PnP_main\EPro_PnP_6DoF\dataset\lm\real_train\AR_QC\*-color.png')
	imgs = glob.glob(r'E:\DL_Projects\EPro_PnP_main\EPro_PnP_6DoF\dataset\lm\imgn\AR_QC\*-color.png')
	imgs.extend(glob.glob(r'E:\DL_Projects\EPro_PnP_main\EPro_PnP_6DoF\dataset\lm\imgn\AR_QC_data_applend\*-color.png'))
	cnt = 0
	for each in ids:
		id = each[-16:-10]
		print(cnt)
		cnt += 1
		print(id)
		a = cv2.imread(r'E:\DL_Projects\EPro_PnP_main\EPro_PnP_6DoF\dataset\lm\imgn\AR_QC\\' + id + each[-10:])
		if a is None:
			a = cv2.imread(
				r'E:\DL_Projects\EPro_PnP_main\EPro_PnP_6DoF\dataset\lm\imgn\AR_QC_data_applend\\' + id + each[-10:])
		mask = np.zeros((480, 640))
		for i in range(480):
			for j in range(640):
				if a[i, j, 0] != 0 or a[i, j, 1] != 0 or a[i, j, 2] != 0:
					mask[i, j] = 1


# cv2.imwrite(each.replace('-color.png', '-label.png'), np.uint8(mask))
# cv2.imshow('a', a)
# cv2.imshow('mask', mask)
# cv2.waitKey(1)
# blocksize = 31
# C = -mask.mean()
# mask = cv2.adaptiveThreshold(mask, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, blocksize, C)
# bboxs = mask_find_bboxs(mask.astype(np.uint8))
#
# if len(bboxs)>1:
#     bboxs = [bboxs[-1]]
# # print(len(bboxs))
# for b in bboxs:
#     x0, y0 = b[0], b[1]
#     x1 = b[0] + b[2]
#     y1 = b[1] + b[3]
#
#     w, h = x1 - x0, y1 - y0
#
#     Note = open(each[:-9] +'box.txt', mode='w')
#     Note.write(str(x0) + '\n')
#     Note.write(str(y0) + '\n')
#     Note.write(str(w) + '\n')
#     Note.write(str(h) + '\n')
#     Note.close()
#     start_point, end_point = (x0, y0), (x1, y1)
#     color = (0, 0, 255)  # 边框颜色红
#     thickness = 3  # 边框厚度1
#     a = np.ascontiguousarray(a)
#     a = cv2.rectangle(a, start_point, end_point, color, thickness)
#     cv2.imshow('a', a)
#     cv2.waitKey(1)


def rpkl():
	pkl_path = 'E:\DL_Projects\EPro_PnP_main\EPro_PnP_6DoF\dataset\lm\imgn\\ape\\000005_5-coor.pkl'
	data_list = read_pickle(pkl_path)
	print(data_list)


import joblib


def genpkl():
	ids = glob.glob('E:\DL_Projects\EPro_PnP_main\EPro_PnP_6DoF\dataset\lm\imgn\AR_QC_data_applend\*-color.png')
	cnt = 0
	for each in ids:
		# 先声明
		gws = []
		# 第一个参数，要生成的类，第二个存放pkl路径
		joblib.dump(gws, each[:-9] + 'coor.pkl')
		print(cnt)
		cnt += 1
	return "pkl 生成完成"


def data_aug():
	cnt = 0
	imgs = glob.glob(r'E:\DL_Projects\EPro_PnP_main\EPro_PnP_6DoF\dataset\lm\real_test\1\*-color.png')
	boxes = glob.glob(r'E:\DL_Projects\EPro_PnP_main\EPro_PnP_6DoF\dataset\lm\real_test\1\*-box.txt')
	for each, each2 in zip(imgs, boxes):
		with open(each2) as file:
			box_content = file.read()
			box_content = list(box_content.split('\n'))
			x0, y0, w, h = float(box_content[0]), float(box_content[1]), float(box_content[2]), float(box_content[3])
			file.close()
		cnt += 1
		print(cnt, each[:-9] + 'box_fasterrcnn.txt')
		Note = open(each[:-9] + 'box_fasterrcnn.txt', mode='w')
		Note.write(str(x0) + '\n')
		Note.write(str(y0) + '\n')
		Note.write(str(w + x0) + '\n')
		Note.write(str(h + y0))
		Note.close()


def filt():
	all_pics = glob.glob(r'E:\DL_Projects\EPro_PnP_main\EPro_PnP_6DoF\dataset\lm\real_test\AR_QC\*-color.png')
	all_box = glob.glob(r'E:\DL_Projects\EPro_PnP_main\EPro_PnP_6DoF\dataset\lm\real_test\AR_QC\*-box.txt')
	all_box2 = glob.glob(r'E:\DL_Projects\EPro_PnP_main\EPro_PnP_6DoF\dataset\lm\real_test\AR_QC\*-box_fasterrcnn.txt')
	all_pose = glob.glob(r'E:\DL_Projects\EPro_PnP_main\EPro_PnP_6DoF\dataset\lm\real_test\AR_QC\*-pose.txt')
	all_coor = glob.glob(r'E:\DL_Projects\EPro_PnP_main\EPro_PnP_6DoF\dataset\lm\real_test\AR_QC\*-coor.pkl')
	all_names = []
	for i in all_pics:
		all_names.append(i[-16:-10])
	print(len(all_names))
	for j in all_box2:
		if j[-25:-19] not in all_names:
			os.remove(j)


def remove_data(i):
	os.remove(i)
	os.remove(i.replace('-color.png', '-pose.txt'))
	os.remove(i.replace('-color.png', '-box.txt'))
	# os.remove(i.replace('-color.png', '-box_fasterrcnn.txt'))
	os.remove(i.replace('-color.png', '-label.png'))
	os.remove(i.replace('-color.png', '-coor.pkl'))
	print('delete done')
	return


def check_data():
	imgs = glob.glob(r'E:\DL_Projects\EPro_PnP_main\EPro_PnP_6DoF\dataset\lm\imgn\AR_QC\*-color.png')
	masks = glob.glob(r'E:\DL_Projects\EPro_PnP_main\EPro_PnP_6DoF\dataset\lm\imgn\AR_QC\*-label.png')
	boxes = glob.glob(r'E:\DL_Projects\EPro_PnP_main\EPro_PnP_6DoF\dataset\lm\imgn\AR_QC\*-box.txt')
	coors = glob.glob(r'E:\DL_Projects\EPro_PnP_main\EPro_PnP_6DoF\dataset\lm\imgn\AR_QC\*-coor.pkl')
	poses = glob.glob(r'E:\DL_Projects\EPro_PnP_main\EPro_PnP_6DoF\dataset\lm\imgn\AR_QC\*-pose.txt')
	for i, j, k, l, m in zip(imgs[-1::-1], boxes[-1::-1], masks[-1::-1], poses[-1::-1], coors[-1::-1]):
		# print(i)
		q = i[:66] + '0' + i[67:]
		s = i[:66] + '1' + i[67:]
		e = i[:66] + '2' + i[67:]
		with open(j) as file:
			box_content = file.read()
			box_content = list(box_content.split('\n'))
			x0, y0, w, h = float(box_content[0]), float(box_content[1]), float(box_content[2]), float(box_content[3])
			file.close()

		sp, ep = (int(x0), int(y0)), (int(x0 + w), int(y0 + h))
		color = (0, 0, 255)  # 边框颜色红
		thickness = 3  # 边框厚度1
		a = cv2.imread(i)
		m = cv2.imread(k)
		a = np.ascontiguousarray(a)
		a = cv2.rectangle(a, sp, ep, color, thickness)
		cv2.imshow('a', a)
		cv2.imshow('m', np.uint8(m * 255))
		if cv2.waitKey() == 32:
			remove_data(i)
		# remove_data(q)
		# remove_data(s)
		# remove_data(e)
		else:
			print('retain')


import scipy.linalg as linalg


def rotate_mat(axis, radian):
	rot_matrix = linalg.expm(np.cross(np.eye(3), axis / linalg.norm(axis) * radian))
	return rot_matrix


def add_data():
	flag = False
	cnt = 0
	obj = 'AR_QC'
	point_cloud_file = "E:\DL_Projects\EPro_PnP_main\EPro_PnP_6DoF\dataset\lm\models\\" + obj + "\\" + obj + ".ply"
	pcd = o3d.io.read_point_cloud(point_cloud_file)

	vis = o3d.visualization.Visualizer()
	vis.create_window(window_name='pose_est', width=640, height=480, visible=False)
	vis.add_geometry(pcd)

	ctr = vis.get_view_control()
	# get default camera parameters
	param = o3d.camera.PinholeCameraParameters()
	param.intrinsic.intrinsic_matrix = [[572.4114, 0, 325.2611], [0, 573.57043, 242.04899], [0, 0, 1]]
	# param.intrinsic.intrinsic_matrix = [[943.2, 0, 639.5], [0, 943.2, 359.5], [0, 0, 1]]

	render_option = vis.get_render_option()
	render_option.background_color = np.array([0, 0, 0])  # 设置背景为黑色
	render_option.point_size = 2.5
	poses = glob.glob(r'E:\DL_Projects\EPro_PnP_main\EPro_PnP_6DoF\dataset\lm\imgn\AR_QC\*-pose.txt')
	for each in poses:
		# img = cv2.imread(each.replace('-pose.txt', '-color.png'))
		# cv2.imshow('img', img)
		for ggg in ['1', '2']:
			id = ggg + each[-14:-9]
			with open(each, 'r') as posefile:
				pose_content = posefile.read()
				posefile.close()
			pose_gt = pose_content.split()
			for i, j in zip(pose_gt, range(len(pose_gt))):
				pose_gt[j] = float(i)
			pose_gt.append(0.0)
			pose_gt.append(0.0)
			pose_gt.append(0.0)
			pose_gt.append(1.0)
			pose_gt = np.array(pose_gt).reshape(4, 4)

			if ggg != '00':
				rx = rotate_mat(axis=[1, 0, 0], radian=math.pi * np.random.randint(-10, 10) / 100)
				ry = rotate_mat(axis=[0, 1, 0], radian=math.pi * np.random.randint(-10, 10) / 100)
				rz = rotate_mat(axis=[0, 0, 1], radian=math.pi * np.random.randint(-10, 10) / 100)
				t = np.dot(np.dot(rx, ry), rz)
				pose_gt[0:3, 0:3] = np.dot(pose_gt[0:3, 0:3], t)

				tx, ty, tz = np.random.randint(-10, 11) / 100, np.random.randint(-10, 11) / 100, np.random.randint(-10,
				                                                                                                   11) / 100
				pose_gt[0, 3] = pose_gt[0, 3] + tx
				pose_gt[1, 3] = pose_gt[1, 3] + ty
				pose_gt[2, 3] = pose_gt[2, 3] + tz

			# pose
			# file = open(each[:-15] + id + "-pose.txt", 'w')
			# for i in [0, 1, 2]:
			#     for j in [0, 1, 2, 3]:
			#         file.write(str(pose_gt[i][j]) + ' ')
			#     file.write('\n')
			# file.close()

			param.extrinsic = pose_gt

			# set param to current camera control view
			ctr.convert_from_pinhole_camera_parameters(param, allow_arbitrary=True)
			vis.update_geometry(pcd)
			vis.poll_events()
			vis.update_renderer()

			# png
			# vis.capture_screen_image(each[:-15] + id + '-color.png')

			res = vis.capture_screen_float_buffer()
			# 将渲染图像从Open3D格式转换为OpenCV格式
			res = np.asarray(res)
			res = (res * 255).astype(np.uint8)
			res = cv2.cvtColor(res, cv2.COLOR_RGB2BGR)

			# mask
			mask = np.zeros((480, 640))
			mask[(res[:, :, 0] != 0) | (res[:, :, 1] != 0) | (res[:, :, 2] != 0)] = 1
			# cv2.imshow('mask', mask)
			# cv2.imwrite(each[:-15] + id + '-label.png', np.uint8(mask))

			# box
			bboxs = mask_find_bboxs(mask.astype(np.uint8))
			if len(bboxs) > 1:
				bboxs = [bboxs[-1]]
			for b in bboxs:
				x0, y0 = b[0], b[1]
				x1 = b[0] + b[2]
				y1 = b[1] + b[3]
				w, h = x1 - x0, y1 - y0
				if x0 > 0 and y0 > 0 and x1 < 640 and y1 < 480:
					flag = True
				if flag:
					Note = open(each[:-15] + id + '-box.txt', mode='w')
					Note.write(str(x0) + '\n')
					Note.write(str(y0) + '\n')
					Note.write(str(w) + '\n')
					Note.write(str(h) + '\n')
					Note.close()
					file = open(each[:-15] + id + "-pose.txt", 'w')
					for i in [0, 1, 2]:
						for j in [0, 1, 2, 3]:
							file.write(str(pose_gt[i][j]) + ' ')
						file.write('\n')
					file.close()
					vis.capture_screen_image(each[:-15] + id + '-color.png')
					cv2.imwrite(each[:-15] + id + '-label.png', np.uint8(mask))
					joblib.dump([], each[:-15] + id + '-coor.pkl')
					flag = False
			# coor
			# joblib.dump([], each[:-15] + id + '-coor.pkl')
			print(cnt)
			cnt += 1


# cv2.waitKey()


def change_data():
	cnt = 0
	obj = 'AR_QC'
	point_cloud_file = "E:\DL_Projects\EPro_PnP_main\EPro_PnP_6DoF\dataset\lm\models\\" + obj + "\\" + obj + ".ply"
	pcd = o3d.io.read_point_cloud(point_cloud_file)

	vis = o3d.visualization.Visualizer()
	vis.create_window(window_name='pose_est', width=640, height=480, visible=False)
	vis.add_geometry(pcd)

	ctr = vis.get_view_control()
	# get default camera parameters
	param = o3d.camera.PinholeCameraParameters()
	param.intrinsic.intrinsic_matrix = [[572.4114, 0, 325.2611], [0, 573.57043, 242.04899], [0, 0, 1]]
	# param.intrinsic.intrinsic_matrix = [[943.2, 0, 639.5], [0, 943.2, 359.5], [0, 0, 1]]

	render_option = vis.get_render_option()
	render_option.background_color = np.array([0, 0, 0])  # 设置背景为黑色
	render_option.point_size = 2.5
	poses = glob.glob(r'E:\DL_Projects\EPro_PnP_main\EPro_PnP_6DoF\dataset\lm\imgn\AR_QC\*-pose.txt')
	poses.extend(glob.glob(r'E:\DL_Projects\EPro_PnP_main\EPro_PnP_6DoF\dataset\lm\real_test\AR_QC\*-pose.txt'))
	for each in poses:
		with open(each, 'r') as posefile:
			pose_content = posefile.read()
			posefile.close()
		pose_gt = pose_content.split()
		for i, j in zip(pose_gt, range(len(pose_gt))):
			pose_gt[j] = float(i)
		pose_gt.append(0.0)
		pose_gt.append(0.0)
		pose_gt.append(0.0)
		pose_gt.append(1.0)
		pose_gt = np.array(pose_gt).reshape(4, 4)
		param.extrinsic = pose_gt

		# set param to current camera control view
		ctr.convert_from_pinhole_camera_parameters(param, allow_arbitrary=True)
		vis.update_geometry(pcd)
		vis.poll_events()
		vis.update_renderer()

		# png
		vis.capture_screen_image(each.replace('-pose.txt', '-color.png'))

		res = vis.capture_screen_float_buffer()
		# 将渲染图像从Open3D格式转换为OpenCV格式
		res = np.asarray(res)
		res = (res * 255).astype(np.uint8)
		res = cv2.cvtColor(res, cv2.COLOR_RGB2BGR)

		# mask
		mask = np.zeros((480, 640))
		mask[(res[:, :, 0] != 0) | (res[:, :, 1] != 0) | (res[:, :, 2] != 0)] = 1
		cv2.imwrite(each.replace('-pose.txt', '-label.png'), np.uint8(mask))

		# box
		bboxs = mask_find_bboxs(mask.astype(np.uint8))
		if len(bboxs) > 1:
			bboxs = [bboxs[-1]]
		for b in bboxs:
			x0, y0 = b[0], b[1]
			x1 = b[0] + b[2]
			y1 = b[1] + b[3]
			w, h = x1 - x0, y1 - y0
			os.remove(each.replace('-pose.txt', '-box.txt'))
			Note = open(each.replace('-pose.txt', '-box.txt'), mode='w')
			Note.write(str(x0) + '\n')
			Note.write(str(y0) + '\n')
			Note.write(str(w) + '\n')
			Note.write(str(h) + '\n')
			Note.close()
		print(cnt)
		cnt += 1


def eulerAngles2rotationMat(theta, format='degree'):
	"""
    Calculates Rotation Matrix given euler angles.
    :param theta: 1-by-3 list [rx, ry, rz] angle in degree
    :return:
    RPY角，是ZYX欧拉角，依次 绕定轴XYZ转动[rx, ry, rz]
    """
	if format is 'degree':
		theta = [i * math.pi / 180.0 for i in theta]

	R_x = np.array([[1, 0, 0],
	                [0, math.cos(theta[0]), -math.sin(theta[0])],
	                [0, math.sin(theta[0]), math.cos(theta[0])]
	                ])

	R_y = np.array([[math.cos(theta[1]), 0, math.sin(theta[1])],
	                [0, 1, 0],
	                [-math.sin(theta[1]), 0, math.cos(theta[1])]
	                ])

	R_z = np.array([[math.cos(theta[2]), -math.sin(theta[2]), 0],
	                [math.sin(theta[2]), math.cos(theta[2]), 0],
	                [0, 0, 1]
	                ])
	R = np.dot(R_z, np.dot(R_y, R_x))
	return R


def prepare_pose_data(cls='arqc'):
	H, W = 640, 640
	obj = cls
	point_cloud_file = "./models/" + obj + "/" + obj + ".ply"
	pcd = o3d.io.read_point_cloud(point_cloud_file)
	# pcd.paint_uniform_color([0.8705, 0.1608, 0.0627])

	vis = o3d.visualization.Visualizer()
	vis.create_window(window_name='pose_est', width=W, height=H, visible=True)
	vis.add_geometry(pcd)

	ctr = vis.get_view_control()
	# get default camera parameters
	param = o3d.camera.PinholeCameraParameters()
	# param.intrinsic.intrinsic_matrix = [[572.4114, 0, 325.2611], [0, 573.57043, 242.04899], [0, 0, 1]]#EPro-PnP
	param.intrinsic.intrinsic_matrix = [[524.7917885754071, 0, 332.5213232846151],
	                                    [0, 489.3563960810721, 281.2339855172282],
	                                    [0, 0, 1]]  # DeepIM
	# param.intrinsic.intrinsic_matrix = [[640, 0, 320], [0, 640, 320], [0, 0, 1]]

	render_option = vis.get_render_option()
	render_option.background_color = np.array([0, 0, 0])  # 设置背景为黑色
	render_option.point_size = 2.5

	poses_all = "./pose_standard.txt"
	poses = {}
	bgs = glob.glob(r'/home/yxh/文档/datasets/coco/val2017/*.jpg')
	with open(poses_all, 'r') as posefile:
		pose_content = posefile.read()
		posefile.close()
		pose_content = pose_content.split()
		for each in range(0, 144, 3):
			poses[str(each // 3)] = pose_content[each] + ',' + pose_content[each + 1] + ',' + pose_content[each + 2]

	for each in poses:
		a = poses[each].split(',', -1)
		for i, j in enumerate(a):
			a[i] = float(j)
		poses[each] = a

	for iter in range(4800):
		pose_cls = iter // 100
		pose_base = copy.deepcopy(poses)[str(pose_cls)]
		pose_base.append(0.0)
		pose_base.append(0.0)
		pose_base.append(0.0)
		pose_base.append(1.0)
		pose_base = np.array(pose_base).reshape(4, 4)

		rot = False
		trans = False
		if rot:
			rx = rotate_mat(axis=[1, 0, 0], radian=math.pi * np.random.randint(-10, 10) / 100)
			ry = rotate_mat(axis=[0, 1, 0], radian=math.pi * np.random.randint(-10, 10) / 100)
			rz = rotate_mat(axis=[0, 0, 1], radian=math.pi * np.random.randint(-10, 10) / 100)
			t = np.dot(np.dot(rx, ry), rz)
			pose_base[0:3, 0:3] = np.dot(pose_base[0:3, 0:3], t)
		if trans:
			tx, ty, tz = np.random.randint(-10, 11) / 100, np.random.randint(-10, 11) / 100, np.random.randint(-10,
			                                                                                                   11) / 100
			pose_base[0, 3] = pose_base[0, 3] + tx
			pose_base[1, 3] = pose_base[1, 3] + ty
			pose_base[2, 3] = pose_base[2, 3] + tz

		tx, ty, tz = 0, 0, 0
		pose_base[0, 3] = pose_base[0, 3] + tx
		pose_base[1, 3] = pose_base[1, 3] + ty
		pose_base[2, 3] = pose_base[2, 3] + tz

		param.extrinsic = pose_base

		# set param to current camera control view
		ctr.convert_from_pinhole_camera_parameters(param, allow_arbitrary=True)
		vis.update_geometry()
		vis.poll_events()
		vis.update_renderer()

		# vis.run()
		# vis.destroy_window()

		# png
		# vis.capture_screen_image('..\\train\\images\\' + str(iter) + '.png')

		res = vis.capture_screen_float_buffer()
		# 将渲染图像从Open3D格式转换为OpenCV格式
		res = np.asarray(res)
		res = (res * 255).astype(np.uint8)
		res = cv2.cvtColor(res, cv2.COLOR_RGB2BGR)

		# mask
		mask = np.zeros((H, W, 3))
		mask[(res[:, :, 0] != 0) | (res[:, :, 1] != 0) | (res[:, :, 2] != 0)] = 1

		bg = cv2.imread(bgs[np.random.randint(0, len(bgs))])
		bg = cv2.resize(bg, dsize=(W, H))
		# cv2.imwrite('..\\' + obj + '\\val\\images\\' + str(iter) + '.png', bg * (1 - mask) + res)
		# box
		bboxs = mask_find_bboxs(mask[:, :, 0].astype(np.uint8))
		if len(bboxs) > 1:
			bboxs = [bboxs[-1]]
		for b in bboxs:
			x0, y0 = b[0], b[1]
			x1 = b[0] + b[2]
			y1 = b[1] + b[3]
			w, h = x1 - x0, y1 - y0
			xcs, ycs, hs, ws = ((x0 + w / 2) / W), ((y0 + h / 2) / H), h / H, w / W
		# print('\"'+str(iter)+'\":', w*h,',')
		# print('\"'+str(iter)+'\":', mat2quat(pose_base[:3, :3]), ',')
		# Note = open('..\\' + obj + '\\val\\labels\\' + str(iter) + '.txt', mode='w')
		# Note.write(str(pose_cls) + ' ')
		# Note.write(str(xcs) + ' ')
		# Note.write(str(ycs) + ' ')
		# Note.write(str(ws) + ' ')
		# Note.write(str(hs) + '\n')
		# Note.close()
		# print('tz:', pose_base[2, 3], 'pred_z:', z0 * sqrt_S0 / math.sqrt(h * w))

# prepare_pose_data()


def prepare_yolo_data(cls='AR_QC'):
	pics_all = glob.glob("..\\dataset\\lm\\imgn\\" + cls + "\\*-color.png")
	pics_all.extend(glob.glob("..\\dataset\\lm\\real_test\\" + cls + "\\*-color.png"))
	bgs = glob.glob(r'..\dataset\bg_images\VOC2012\JPEGImages\*.jpg')
	cnt = 0
	for each in pics_all:
		if 'imgn' in each:
			pic = cv2.imread(each)
			label = cv2.imread(each.replace('-color.png', '-label.png'))
			bg = cv2.imread(bgs[np.random.randint(0, len(bgs), None)])
			bg = cv2.resize(bg, dsize=(640, 480))
			label = label / 17
			color = bg * (1 - label) + pic * label
			cv2.imwrite(each.replace(cls, 'train\\images'), color)
			box_path = each.replace('-color.png', '-box.txt')
			target_box_path = box_path.replace(cls, 'train\\labels')

		elif 'real_test' in each:
			shutil.copy(each, each.replace(cls, 'val\\images').replace('real_test', 'imgn'))
			box_path = each.replace('-color.png', '-box_fasterrcnn.txt')
			target_box_path = each.replace('-color.png', '-box.txt').replace(cls, 'val\\labels').replace('real_test',
			                                                                                             'imgn')

		with open(box_path, 'r') as boxfile:
			box_content = boxfile.read()
			box_gt = box_content.split()
			for i, j in zip(box_gt, range(len(box_gt))):
				box_gt[j] = float(i)  # x0,y0,w,h
			x0, y0, w, h = box_gt
			if 'real_test' in each:
				x0, y0, x1, y1 = box_gt
				w, h = x1 - x0, y1 - y0
			box_gt[0] = (x0 + w / 2) / 640
			box_gt[1] = (y0 + h / 2) / 480
			box_gt[2] = (w) / 640
			box_gt[3] = (h) / 480
			boxfile.close()
			Note = open(target_box_path, mode='w')
			Note.write(str(0) + ' ')
			Note.write(str(box_gt[0]) + ' ')
			Note.write(str(box_gt[1]) + ' ')
			Note.write(str(box_gt[2]) + ' ')
			Note.write(str(box_gt[3]) + ' ')
			Note.close()
		print(cnt)
		cnt += 1


def prepare_coco_data(cls='squirrel'):
	import pandas as pd
	def gen_coor(binary):
		contours, heriachy = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
		print(len(contours))
		for i, contour in enumerate(contours):
			if len(contour) < 20:
				continue
			num = len(contour[:, 0, 0])  # 个数
			if num > 500:
				hundred = num // 100  # 步长
				tem = contour[:, 0][::hundred]
				return tem, 1
			if 1 <= num <= 10:
				print('num<10')
				return contour[:, 0], 0
			else:
				return contour[:, 0], 1

	name_l = []
	w_list = []
	h_list = []
	contours_list = []
	h, w = 480, 640
	pics_all = glob.glob("..\\dataset\\lm\\imgn\\" + cls + "\\*-color.png")
	pics_all.extend(glob.glob("..\\dataset\\lm\\real_test\\" + cls + "\\*-color.png"))
	bgs = glob.glob(r'..\dataset\bg_images\VOC2012\JPEGImages\*.jpg')
	for each in pics_all:
		if 'imgn' in each:
			binary = cv2.imread(each.replace('-color.png', '-label.png'))
			pic = cv2.imread(each)
			bg = cv2.imread(bgs[np.random.randint(0, len(bgs), None)])
			bg = cv2.resize(bg, dsize=(640, 480))
			color = bg * (1 - binary) + pic * binary
			cv2.imwrite(each.replace(cls, 'train2014'), color)
			binary = binary * 255
			contours, flag = gen_coor(binary=binary[:, :, 0])
			if flag == 0:
				continue
			name = each.split('squirrel\\')[1]
			print(name)
			name_l.append(name)
			w_list.append(w)
			h_list.append(h)
			contours_list.append(contours)
	res_pd = pd.DataFrame()
	res_pd['imagePath'] = name_l
	res_pd['imageWidth'] = w_list
	res_pd['imageHeight'] = h_list
	res_pd['points'] = contours_list
	res_pd.to_csv(r'..\dataset\lm\imgn\result.csv', index=False)

	def get_coor(sppoint):
		tem_list = []
		for i in range(len(sppoint)):
			sp = sppoint[i].replace('[', '')
			sp1 = sp.replace(']', '')
			sp2 = sp1.strip()
			num = sp2.count(' ')
			strs = ''
			for n in range(num):
				strs = strs + ' '
				sp3 = sp2.replace(strs, ',')  # sppoint[i][-6:-5]
			x = int(sp3.split(',')[0])
			y = int(sp3.split(',')[-1])
			coor = [x, y]
			tem_list.append(coor)
		return tem_list

	def process_text_to_json(name, h, w, coor):
		dict = {}
		dict["version"] = "5.0.1"
		dict["flags"] = {}
		dict["shapes"] = []
		dict["shapes"].append(
			{"label": cls, "points": coor, "group_id": "null", "shape_type": "polygon", "flags": {}})
		dict["imagePath"] = name
		dict["imageData"] = "none"
		dict["imageHeight"] = h
		dict["imageWidth"] = w
		# location_data = {"location_data": location_data}
		return json.dumps(dict, indent=4)

	coor_df = pd.read_csv(r'..\dataset\lm\imgn\result.csv')

	imagePath_l = []
	imageWidth_l = []
	imageHeight_l = []

	print(coor_df['points'])

	for i, imagePath in enumerate(coor_df['imagePath']):
		imagePath_l.append(imagePath)

	for i, imageWidth in enumerate(coor_df['imageWidth']):
		imageWidth_l.append(imageWidth)

	for i, imageHeight in enumerate(coor_df['imageHeight']):
		imageHeight_l.append(imageHeight)

	for i, point in enumerate(coor_df['points']):
		print(imagePath_l[i])
		sppoint = point.split('\n ')
		coor = get_coor(sppoint)
		x = process_text_to_json(imagePath_l[i], imageHeight_l[i], imageWidth_l[i], coor)
		# 保存本地json文件
		fileObject = open(r'..\SSD\datasets\train2014\{}.json'.format(imagePath_l[i][-18:-4]), 'w')
		fileObject.write(x)
		fileObject.close()


def camera(save_path):
	cap = cv2.VideoCapture(0)  # 打开摄像头
	if not cap.isOpened():
		print("未打开摄像头!")

	else:
		print("已打开摄像头，任务开始!")
		frame_index = 1  # 图片计数

		retval, frame = cap.read()  # 这里是读取视频帧，第一个参数输出是否识别到，第二个参数输出识别到的图像帧
		while retval:  # 当读到图像帧时
			if cv2.waitKey(1) & 0xFF == ord('q'):  # 中断采集
				break
			if frame_index == 501:  # 拍摄多少张停止
				break
			print(frame_index)
			cv2.imshow("camera_frame", frame)  # 显示视频帧
			cv2.waitKey(100)  # 间隔100ms拍摄一张图片

			save1 = str(save_path)  # jpg格式图片存放文件夹的路径
			isExists = os.path.exists(save1)
			if not isExists:  # 判断如果文件不存在,则创建
				os.makedirs(save1)
			frame_name1 = f"{frame_index}-color.png"
			save1 = save1 + frame_name1
			retval, frame = cap.read()
			frame = cv2.resize(frame, dsize=(640, 480))
			cv2.imwrite(save1, frame)  # 保存拍摄的图片
			frame_index += 1

	cv2.destroyAllWindows()
	cap.release()


def change_yolo_data():
	boxes = glob.glob(r'E:\DL_Projects\EPro_PnP_main\EPro_PnP_6DoF\dataset\lm\imgn\train\labels\*-color.txt')
	for box_path in boxes:
		with open(box_path, 'r') as boxfile:
			box_content = boxfile.read()
			box_gt = box_content.split()
			for i, j in zip(box_gt, range(len(box_gt))):
				box_gt[j] = float(i)  # x0,y0,w,h
				cls, x0, y0, w, h = box_gt
		os.remove(box_path)
		Note = open(box_path.replace('color', 'box'), mode='w')
		Note.write(str(0) + ' ')
		Note.write(str(x0) + ' ')
		Note.write(str(y0) + ' ')
		Note.write(str(w) + ' ')
		Note.write(str(h) + ' ')
		Note.close()


def box():
	obj = 'squirrel'
	pcd = o3d.io.read_point_cloud("./models/" + obj + "/" + obj + ".ply")

	# 获取AABB包围盒
	aabb = pcd.get_axis_aligned_bounding_box()
	aabb.color = (0, 0, 0)

	o3d.visualization.draw_geometries([pcd, aabb], window_name="bouding box")
