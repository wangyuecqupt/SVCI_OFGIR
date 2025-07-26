import open3d as o3d
import numpy as np
import scipy.linalg as linalg
import torch
import torch.nn as nn
import cv2
import random
import os


def visualize(inps, pre, label):
    inp_show = inps[0, 3:, :, :].permute(1, 2, 0).detach().cpu().numpy().astype(np.uint8)
    pre_show = inps[pre + 1, 3:, :, :].permute(1, 2, 0).detach().cpu().numpy().astype(np.uint8)
    gt_show = inps[label + 1, 3:, :, :].permute(1, 2, 0).detach().cpu().numpy().astype(np.uint8)
    inp_show = cv2.putText(np.ascontiguousarray(inp_show), "test", (0, 20),
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3)
    pre_show = cv2.putText(np.ascontiguousarray(pre_show), "pre: " + str(pre), (0, 20),
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3)
    gt_show = cv2.putText(np.ascontiguousarray(gt_show), "gt:" + str(label), (0, 20),
                          cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3)
    shows = np.hstack((pre_show, inp_show, gt_show))
    cv2.imshow('shows', shows)
    cv2.waitKey()

def seed_torch(seed=3407):
    np.random.seed(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)  # 为了禁止hash随机化，使得实验可复现
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.enabled = False
    os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"


def get_test_sample(path):
    realimg = cv2.imread(path)
    H, W = realimg.shape[:2]
    with open(path.replace('images', 'labels').replace('.png', '.txt'), 'r') as f:
        cls_box = f.readlines()[0].split(' ')
        xc = int(float(cls_box[1]) * W)
        yc = int(float(cls_box[2]) * H)
        w = int(float(cls_box[3]) * W)
        h = int(float(cls_box[4][:-2]) * H)
        f.close()
    x0, y0, x1, y1 = xc-w//2, yc-h//2, xc + w//2, yc + h//2
    if h >= w:
        x0 = x0 - (h - w) // 2
        x1 = x1 + (h - w) // 2
    else:
        y0 = y0 - (w - h) // 2
        y1 = y1 + (w - h) // 2
    realimg = cv2.resize(realimg[y0:y1, x0:x1, :], dsize=(128, 128))
    return realimg


def conv(batchNorm, in_planes, out_planes, kernel_size=3, stride=1):
    if batchNorm:
        return nn.Sequential(
            nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=(kernel_size - 1) // 2,
                      bias=False),
            nn.BatchNorm2d(out_planes),
            nn.LeakyReLU(0.1, inplace=True),
        )
    else:
        return nn.Sequential(
            nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=(kernel_size - 1) // 2,
                      bias=True),
            nn.LeakyReLU(0.1, inplace=True),
        )


def rotate_mat(axis, radian):
    rot_matrix = linalg.expm(np.cross(np.eye(3), axis / linalg.norm(axis) * radian))
    return rot_matrix


class SerializableMesh:
    def __init__(self, mesh: o3d.geometry.TriangleMesh):
        self.vertices = np.asarray(mesh.vertices)
        self.triangles = np.asarray(mesh.triangles)
        self.vertex_colors = np.asarray(mesh.vertex_colors)

    def to_open3d(self) -> o3d.geometry.TriangleMesh:
        src_obj = o3d.geometry.TriangleMesh(
            vertices=o3d.utility.Vector3dVector(self.vertices),
            triangles=o3d.utility.Vector3iVector(self.triangles),
        )
        src_obj.vertex_colors = o3d.utility.Vector3dVector(self.vertex_colors)
        return src_obj


class SerializablePointCloud:
    def __init__(self, pointcloud: o3d.geometry.PointCloud):
        self.points = np.array(pointcloud.points)
        self.colors = np.array(pointcloud.colors)
        self.normals = np.array(pointcloud.normals)

    def to_open3d(self) -> o3d.geometry.PointCloud:
        pointcloud = o3d.geometry.PointCloud()
        pointcloud.points = o3d.utility.Vector3dVector(self.points)
        pointcloud.colors = o3d.utility.Vector3dVector(self.colors)
        pointcloud.normals = o3d.utility.Vector3dVector(self.normals)
        return pointcloud


def acc(pres, labels):
    """用来计算正确率的"""
    result = pres == labels
    result = list(result.detach().cpu())
    t = result.count(True)
    return t / len(labels)
