from augument import DiffDataset
import glob
from tqdm import tqdm
import numpy as np
import open3d as o3d
import torch
import copy
import cv2
import math
from util import rotate_mat, SerializableMesh, SerializablePointCloud


class ClsSet(DiffDataset):
    def __init__(self, classes=['benchvise', 'camera', 'cat', 'driller', 'duck', 'iron', 'lamp', 'phone'], logger=None,
                 H=640, W=640,
                 val=0, show=False):

        self.angles_light = [i for i in range(-15, 16)]
        self.angles_violent = [i for i in range(-35, -19)] + [i for i in range(20, 36)]
        self.trans_level = [1, 1, 11] if H != 640 else [5, 5, 55]
        self.H, self.W = H, W
        self.val = val
        self.show = show
        self.rot, self.trans = True, True
        self.bgs = glob.glob(r'/home/yxh/文档/datasets/coco/val2017/*.jpg')
        poses_all = "/home/yxh/文档/yolov5/3d_dataset/others/pose_standard.txt"
        self.poses = {}

        with open(poses_all, 'r') as posefile:
            pose_content = posefile.read()
            posefile.close()
            pose_content = pose_content.split()
            for each in range(0, 144, 3):
                self.poses[each // 3] = pose_content[each] + ',' + pose_content[each + 1] + ',' + pose_content[
                    each + 2]

        for each in self.poses:
            a = self.poses[each].split(',', -1)
            for i, j in enumerate(a):
                a[i] = float(j)
            a.extend([0.0, 0.0, 0.0, 1.0])
            a = np.array(a).reshape(4, 4)
            self.poses[each] = a

        self.pcds = []
        self.refs = []
        self.poses_of_all_objs = []
        self.classes = classes

        for obj in tqdm(classes):
            if obj == 'squirrel':
                p = o3d.io.read_point_cloud("/media/yxh/Document/yolo_v5/3d_dataset/others/models/" + obj + "/squirrel.ply")
                p = SerializablePointCloud(p)
            else:
                p = o3d.io.read_triangle_mesh("E:/yolo_v5/3d_dataset/others/models/" + obj + "/textured.obj")
                p = SerializableMesh(p)
            self.pcds.append(p)

            ref_imgs = glob.glob('/media/yxh/Document/yolo_v5/3d_dataset/linemod_sym/' + obj + '/*.png')
            ref_imgs = sorted(ref_imgs, key=lambda x: int(x.split('/')[-1][:-4]))

            self.pose_ides = []
            for each in ref_imgs:
                self.pose_ides.append(int(each.split('/')[-1][:-4]))
            self.poses_of_all_objs.append(self.pose_ides)

            refs = []
            for each in ref_imgs:
                refs.append(torch.from_numpy(np.ascontiguousarray(cv2.imread(each))))
            self.refs.append(torch.cat(refs, dim=2).permute(2, 0, 1))
        if logger is not None:
            logger.info("load {} models successfuly: {}.".format(len(classes), classes))
        else:
            print("load {} models successfuly.".format(len(classes)))

        self.render_initialized = False

    def init_render(self):
        self.vis = o3d.visualization.Visualizer()
        self.vis.create_window(window_name='pose_est', width=self.W, height=self.H, visible=False)
        self.param = o3d.camera.PinholeCameraParameters()
        self.param.intrinsic.intrinsic_matrix = [[524.7917885754071, 0, 332.5213232846151],
                                                 [0, 489.3563960810721, 281.2339855172282],
                                                 [0, 0, 1]]
        self.render_option = self.vis.get_render_option()
        self.render_option.background_color = np.array([0, 0, 0])  # 设置背景为黑色
        self.render_option.point_size = 3.5
        self.render_initialized = True

    def __len__(self):
        return len(self.bgs) if self.val == 0 else self.val

    def render_step1(self, obj_id):
        if hasattr(self, "pcd"):
            self.vis.remove_geometry(self.pcd)
        self.pcd = self.pcds[obj_id].to_open3d()
        self.vis.add_geometry(self.pcd)
        self.ctr = self.vis.get_view_control()

    def render_step2(self, angles, trans, pose_id):
        self.pose_base = copy.deepcopy(self.poses)[pose_id]
        if self.rot:
            rx = rotate_mat(axis=[1, 0, 0], radian=math.pi * np.random.choice(angles) / 100)
            ry = rotate_mat(axis=[0, 1, 0], radian=math.pi * np.random.choice(angles) / 100)
            rz = rotate_mat(axis=[0, 0, 1], radian=math.pi * np.random.choice(angles) / 100)
            t = np.dot(np.dot(rx, ry), rz)
            self.pose_base[0:3, 0:3] = np.dot(self.pose_base[0:3, 0:3], t)
        if self.trans:
            tx, ty, tz = np.random.randint(-trans[0], trans[0] + 1) / 100, \
                         np.random.randint(-trans[1], trans[1] + 1) / 100, \
                         np.random.randint(-trans[2], -5) / 100
            self.pose_base[0, 3] = self.pose_base[0, 3] + tx
            self.pose_base[1, 3] = self.pose_base[1, 3] + ty
            self.pose_base[2, 3] = self.pose_base[2, 3] + tz
            self.render_option.point_size = 5 / self.pose_base[2, 3]

    def render_step3(self):
        self.param.extrinsic = self.pose_base
        self.ctr.convert_from_pinhole_camera_parameters(self.param, allow_arbitrary=True)
        self.vis.update_geometry()
        self.vis.poll_events()
        self.vis.update_renderer()
        # png
        res = self.vis.capture_screen_float_buffer()
        # # 将渲染图像从Open3D格式转换为OpenCV格式
        res = np.asarray(res)
        res = (res * 255).astype(np.uint8)
        res = cv2.cvtColor(res, cv2.COLOR_RGB2BGR)
        return res

    def get_rendered_data(self, obj_id, pose_id, angles, trans):
        self.render_step1(obj_id)
        if len(angles) == 2:
            res = []
            self.render_step2(angles[0], trans, pose_id)
            res.append(self.render_step3())
            self.render_step2(angles[1], trans, pose_id)
            res.append(self.render_step3())
        else:
            self.render_step2(angles, trans, pose_id)
            res = self.render_step3()
        return res

    def occlude(self, b, mask):
        x0, y0, w, h = b[0], b[1], b[2], b[3]
        if np.random.randint(0, 2) == 1:
            aa, bb = np.random.randint(0, 3, 2)
            mask[y0 + bb * h // 3:y0 + (bb + 1) * h // 3, x0 + aa * w // 3:x0 + (aa + 1) * w // 3] = 0
        return mask

    def augs(self, img):
        if np.random.randint(0, 3) == 2:
            dx, dy = np.random.randint(low=-5, high=6, size=2)
            img = self.random_jitter(img, dx, dy)
        if np.random.randint(0, 5) == 1:
            img = self.brightness_adjustment(img)
        if np.random.randint(0, 10) == 1:
            img = self.noisy(noise_typ='gauss', image=img)
        elif np.random.randint(0, 10) >= 0:
            img = self.add_noise(img)
        if np.random.randint(0, 3) == 1:
            img = self.virtual_light(img)
        if np.random.randint(0, 3) == 1:
            img = self.add_time_noise(img)
        return img

    def get_bgs(self, numbers_bg):
        bg_ids = np.random.choice(range(len(self.bgs)), numbers_bg)
        bgs = []
        for i, bg_id in enumerate(bg_ids):
            bg = cv2.imread(self.bgs[bg_id])
            if i == 2:
                bg = cv2.resize(bg, dsize=(128, 128))
            else:
                bg = cv2.resize(bg, dsize=(self.H, self.W))
            bgs.append(bg)
        return bgs

    def add_bg(self, fg, bg, crop=False):
        mask = np.zeros((fg.shape[0], fg.shape[1]))
        mask[np.any(fg != 0, axis=2)] = 1
        contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        max_contour = max(contours, key=cv2.contourArea)
        box = cv2.boundingRect(max_contour)
        [x, y, w, h] = box
        mask = self.occlude(box, mask)
        mask = np.repeat(mask[:, :, np.newaxis], 3, axis=2)
        mask = cv2.GaussianBlur(mask, (5, 5), 1, 2)
        res = fg * mask + (1 - mask) * bg
        if not crop:
            return res
        else:
            x0, y0, x1, y1, xc, yc = x, y, x + w, y + h, x + w / 2, y + h / 2
            if h >= w:
                x0 = x0 - (h - w) // 2
                x1 = x1 + (h - w) // 2
                if x0 < 0:
                    delatw = - x0
                    x0 = 0
                    x1 = x1 + delatw
                if x1 > self.W:
                    delatw = x1 - self.W
                    x0 = x0 - delatw
                    x1 = self.W
            else:
                y0 = y0 - (w - h) // 2
                y1 = y1 + (w - h) // 2
                if y0 < 0:
                    delath = - y0
                    y0 = 0
                    y1 = y1 + delath
                if y1 > self.H:
                    delath = y1 - self.H
                    y0 = y0 - delath
                    y1 = self.H
            return res[y0:y1, x0:x1, :]

    def getvalitem(self, index):
        obj_id = int(np.random.choice(range(len(self.classes)), 1))
        self.poses_of_cur_obj = self.poses_of_all_objs[obj_id]
        sample_pose_index = np.random.randint(0, len(self.poses_of_cur_obj))
        pose_id = self.poses_of_cur_obj[sample_pose_index]

        fg = self.get_rendered_data(obj_id, pose_id, angles=self.angles_light, trans=self.trans_level)
        bg = self.get_bgs(numbers_bg=1)[0]
        inp = self.add_bg(fg, bg, crop=True)

        if inp.shape[:2] != (128, 128):
            inp = cv2.resize(inp, dsize=(128, 128))

        inp = self.augs(inp)
        inp = torch.from_numpy(np.ascontiguousarray(inp)).permute(2, 0, 1)
        inps = torch.cat((inp, self.refs[obj_id]), dim=0)

        if self.show:
            a = inp.permute(1, 2, 0).numpy()
            b = self.refs[obj_id][sample_pose_index * 3:sample_pose_index * 3 + 3, :, :].permute(1, 2, 0).numpy()
            cv2.imshow("a", np.hstack((a, b)).astype(np.uint8))
            cv2.waitKey(1)

        return inps.float(), torch.from_numpy(np.ascontiguousarray(pose_id))

    def __getitem__(self, index):

        if not self.render_initialized:
            self.init_render()

        if self.val:
            return self.getvalitem(index)

        obj_id = int(np.random.choice(range(len(self.classes)), 1))
        self.poses_of_cur_obj = self.poses_of_all_objs[obj_id]

        sample_pose_index = np.random.randint(0, len(self.poses_of_cur_obj))
        pose_id = self.poses_of_cur_obj[sample_pose_index]

        fg_light, fg_violent = self.get_rendered_data(obj_id, pose_id,
                                                      angles=[self.angles_light, self.angles_violent],
                                                      trans=self.trans_level)

        bg_light, bg_violent, bg_std = self.get_bgs(numbers_bg=3)

        inp_light = self.add_bg(fg_light, bg_light, crop=True)
        if inp_light.shape[:2] != (128, 128):
            inp_light = cv2.resize(inp_light, dsize=(128, 128))

        inp_light = self.augs(inp_light)
        inp_light = torch.from_numpy(np.ascontiguousarray(inp_light)).permute(2, 0, 1)

        inp_violent = self.add_bg(fg_violent, bg_violent, crop=True)
        if inp_violent.shape[:2] != (128, 128):
            inp_violent = cv2.resize(inp_violent, dsize=(128, 128))

        inp_violent = self.augs(inp_violent)
        inp_violent = torch.from_numpy(np.ascontiguousarray(inp_violent)).permute(2, 0, 1)

        inp_std = self.refs[obj_id][sample_pose_index * 3:sample_pose_index * 3 + 3, :, :].permute(1, 2, 0).numpy()
        inp_std = self.add_bg(inp_std, bg_std)
        if inp_std.shape[:2] != (128, 128):
            inp_std = cv2.resize(inp_std, dsize=(128, 128))
        inp_std = self.augs(inp_std)
        inp_std = torch.from_numpy(np.ascontiguousarray(inp_std)).permute(2, 0, 1)

        if self.show:
            a = inp_light.permute(1, 2, 0).numpy()
            b = inp_violent.permute(1, 2, 0).numpy()
            c = inp_std.permute(1, 2, 0).numpy()
            cv2.imshow("a", np.hstack((a, c, b)).astype(np.uint8))
            cv2.waitKey(1)

        return inp_std.float(), inp_light.float(), inp_violent.float()


if __name__ == '__main__':
    from torch.utils.data import DataLoader

    TrainSet = ClsSet(classes=[
        # 'benchvise',
        # 'camera',
        # 'cat', 'driller', 'duck', 'iron', 'lamp', 'phone'
        'squirrel'
    ], logger=None, show=True, H=640, W=640, val=1000)
    TrainLoader = DataLoader(dataset=TrainSet, batch_size=1, shuffle=True, num_workers=0)
    for bid, (_) in enumerate(TrainLoader):
        pass
