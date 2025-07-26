import PIL.Image
import torch
import numpy as np
import cv2
import copy
import random
import argparse
import glob
from torch.utils.data import Dataset, DataLoader
import json
from PIL import Image

from PIL import Image, ImageDraw
import random


class DiffDataset(Dataset):
    def __init__(self, background_path, cfg):

        self.cfg = cfg
        self.background_images = glob.glob(background_path + '*.jpg')
        self.background_images.extend(glob.glob(background_path + '*.png'))
        self.background_images.extend(glob.glob(background_path + '*.bmp'))

        L = len(self.background_images)
        print('train on {} background pictures'.format(L))

    def rotation(self, img, angle):
        rows = img.shape[0]
        cols = img.shape[1]
        M = cv2.getRotationMatrix2D((cols / 2, rows / 2), angle=angle, scale=1)  # 向左旋转angle度并缩放为原来的scale倍
        img = cv2.warpAffine(img, M, (cols, rows), cv2.INTER_NEAREST)  # 第三个参数是输出图像的尺寸中心
        return img

    def random_jitter(self, input, dx, dy):
        H = np.float32([[1, 0, dx], [0, 1, dy]])  # 定义平移矩阵
        rows, cols = input.shape[:2]  # 获取图像高宽(行列数)
        res = cv2.warpAffine(input, H, (cols, rows), cv2.INTER_NEAREST)
        return res

    def black_edge_crop(self, img, H, W):
        return cv2.resize(img[30:H - 30, 30:W - 30, :], dsize=(W, H))

    def random_h(self, img):
        img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        region_mask1 = np.where(img[:, :, 0] < 175, 1, 0)
        region_mask2 = np.where(img[:, :, 0] > 5, 1, 0)
        region_mask = np.bitwise_and(region_mask1, region_mask2)
        hue_t = np.ones_like(img[:, :, 0]) * np.random.randint(-5, 6) * region_mask
        img[:, :, 0] = img[:, :, 0] + hue_t
        return cv2.cvtColor(img, cv2.COLOR_HSV2BGR)

    def brightness_adjustment(self, img):
        blank = np.zeros_like(img)
        c = (1 + np.random.randint(low=-1, high=2, size=None, dtype='l') / 10.)
        img = cv2.addWeighted(img, c, blank, 1 - c, 0)
        img[img > 255] = 255
        return img

    def change_channel(self, img, sd):
        b, g, r = cv2.split(img)
        if sd == 0:
            out = cv2.merge([b, r, g])
        elif sd == 1:
            out = cv2.merge([b, g, r])
        elif sd == 2:
            out = cv2.merge([r, g, b])
        elif sd == 3:
            out = cv2.merge([r, b, g])
        elif sd == 4:
            out = cv2.merge([g, b, r])
        elif sd == 5:
            out = cv2.merge([g, r, b])
        return out

    def warmcolor(self, img):
        factorR = np.random.choice((1.0, 2.0), 1)[0]
        factorG = np.random.choice((0.5, 1.5), 1)[0]
        img[:, :, 2] = np.clip(img[:, :, 2] * factorR, 0, 255)
        img[:, :, 1] = np.clip(img[:, :, 1] * factorG, 0, 255)
        return img

    def coldcolor(self, img):
        factorB = np.random.choice((1.0, 2.0), 1)[0]
        factorG = np.random.choice((0.5, 1.5), 1)[0]
        img[:, :, 0] = np.clip(img[:, :, 0] * factorB, 0, 255)
        img[:, :, 1] = np.clip(img[:, :, 1] * factorG, 0, 255)
        return img

    def random_flip(self, input, flag):
        if flag == 1:
            return np.fliplr(input)
        elif flag == 2:
            return np.flipud(input)
        elif flag == 3:
            return np.flipud(np.fliplr(input))
        else:
            return input

    def get_hull(self, axis_list):
        hull = cv2.convexHull(axis_list, clockwise=True, returnPoints=True)
        return hull

    def gen_diffs_src(self, ex, mask, diff_src, num_diff, H, W):
        big_cnt = 0
        for b in range(num_diff):
            # 随机确定左上角的点和长宽
            if np.random.randint(0, 30) == 0 and big_cnt == 0:
                x_l, y_l, w, h = np.random.randint(low=99, high=W - 300, size=None, dtype='l'), \
                    np.random.randint(low=99, high=H - 300, size=None, dtype='l'), \
                    np.random.randint(low=100, high=300, size=None, dtype='l'), \
                    np.random.randint(low=100, high=300, size=None, dtype='l')
                big_cnt = 1
                big_flag = 1
            else:
                x_l, y_l, w, h = np.random.randint(low=49, high=W - 50, size=None, dtype='l'), \
                    np.random.randint(low=49, high=H - 50, size=None, dtype='l'), \
                    np.random.randint(low=10, high=50, size=None, dtype='l'), \
                    np.random.randint(low=10, high=50, size=None, dtype='l')
                big_flag = 0

            # 在此方形区域内生成差异， 同时将空mask图片中的对应区域变为1（白色）作为标签
            points = [[x_l + w // 2, y_l + h // 2]]
            for i in range(x_l, x_l + w):
                for j in range(y_l, y_l + h):
                    if big_flag == 1:
                        if np.random.randint(0, 5000) == 1:
                            points.append([i, j])
                    else:
                        # if np.random.randint(0, 100) == 1:
                        if np.random.randint(0, 10) == 1:
                            points.append([i, j])
            random.shuffle(points)
            pts = np.asarray([points], dtype=np.int32)
            hull = self.get_hull(pts).transpose(1, 0, 2)
            mask = cv2.fillPoly(mask.copy(), hull, color=1)
            ex = cv2.fillPoly(ex.copy(), hull, color=(0, 0, 0))

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        mask = cv2.dilate(mask.copy(), kernel, 1)
        mask_filled = np.stack((mask, mask, mask), axis=2)
        mask_filled = cv2.GaussianBlur(mask_filled, (3, 3), 0, 0)
        ex = (255 * ((ex / 255) * (1 - mask_filled) + (diff_src / 255) * mask_filled)).astype(np.uint8)

        return ex, mask

    def gen_diffs(self, ex, mask, diff_src, num_diff, H, W):
        big_cnt = 0
        for b in range(num_diff):
            # 随机确定左上角的点和长宽
            if np.random.randint(0, 30) == 0 and big_cnt == 0:
                x_c, y_c, wd2, hd2 = np.random.randint(low=0 + 50, high=W - 1 - 50, size=None, dtype='l'), \
                    np.random.randint(low=0 + 50, high=H - 1 - 50, size=None, dtype='l'), \
                    np.random.randint(low=25, high=150, size=None, dtype='l'), \
                    np.random.randint(low=25, high=150, size=None, dtype='l')
                big_cnt = 1
                big_flag = 1
            else:
                x_c, y_c, wd2, hd2 = np.random.randint(low=0 + 5, high=W - 1 - 5, size=None, dtype='l'), \
                    np.random.randint(low=0 + 5, high=H - 1 - 5, size=None, dtype='l'), \
                    np.random.randint(low=5, high=25, size=None, dtype='l'), \
                    np.random.randint(low=5, high=25, size=None, dtype='l')
                big_flag = 0

            # 计算矩形区域的坐标范围
            x_range = np.arange(x_c - wd2, x_c + wd2)
            y_range = np.arange(y_c - hd2, y_c + hd2)

            # 创建网格坐标
            grid_coords = np.array(np.meshgrid(x_range, y_range)).T.reshape(-1, 2)

            # 计算选择的概率
            if big_flag == 1:
                # 如果 big_flag 为 1，则使用 1/5000 的概率
                select_prob = 1 / 5000
            else:
                # 否则使用 1/10 的概率
                select_prob = 1 / 50

                # 生成一个均匀分布的随机数数组
            rand_vals = np.random.rand(len(grid_coords))

            # 根据阈值筛选点
            threshold = select_prob
            selected_indices = np.where(rand_vals < threshold)[0]

            # 根据选中的索引提取坐标
            points = grid_coords[selected_indices]

            # 如果需要保证至少有一个点被选中，则手动添加一个点
            if not points.size:
                points = [grid_coords[0]]  # 或者其他合理的默认点

            random.shuffle(points)
            pts = np.asarray([points], dtype=np.int32)
            hull = self.get_hull(pts).transpose(1, 0, 2)
            mask = cv2.fillPoly(mask.copy(), hull, color=1)
            ex = cv2.fillPoly(ex.copy(), hull, color=(0, 0, 0))

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        mask = cv2.dilate(mask.copy(), kernel, 1)
        mask_filled = np.stack((mask, mask, mask), axis=2)
        mask_filled = cv2.GaussianBlur(mask_filled, (3, 3), 0, 0)
        ex = (255 * ((ex / 255) * (1.0 - mask_filled) + (diff_src / 255) * mask_filled)).astype(np.uint8)

        return ex, mask

    def noisy(self, noise_typ, image):
        if noise_typ == "gauss":
            row, col, ch = image.shape
            mean = 0
            var = 0.001
            sigma = var ** 0.5
            gauss = np.random.normal(mean, sigma, (row, col, ch))
            gauss = gauss.reshape(row, col, ch)
            noisy = (image / 255 + gauss) if np.random.randint(0, 2) == 1 else (image / 255 - gauss)
            noisy = cv2.normalize(noisy, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
            noisy = np.uint8(noisy * 255)
            return noisy
        elif noise_typ == "s&p":
            row, col, ch = image.shape
            s_vs_p = 0.5
            amount = 0.004
            out = np.copy(image)
            # Salt mode
            num_salt = np.ceil(amount * image.size * s_vs_p)
            coords = [np.random.randint(0, i - 1, int(num_salt))
                      for i in image.shape[:2]]
            out[:, :, 0:1][tuple(coords)] = 255
            out[:, :, 1:2][tuple(coords)] = 255
            out[:, :, 2:3][tuple(coords)] = 255
            # Pepper mode
            num_pepper = np.ceil(amount * image.size * (1. - s_vs_p))
            coords = [np.random.randint(0, i - 1, int(num_pepper))
                      for i in image.shape[:2]]
            out[:, :, 0:1][tuple(coords)] = 0
            out[:, :, 1:2][tuple(coords)] = 0
            out[:, :, 2:3][tuple(coords)] = 0
            return out

    def generate_curve(self, img, control_points, color, thickness):
        # 使用贝塞尔曲线拟合头发弯曲线条
        curve_points = []
        num_points = 100  # 调整生成的曲线上的点的数量
        for t in np.linspace(0, 1, num_points):
            curve_point = np.power(1 - t, 3) * control_points[0] + 3 * np.power(1 - t, 2) * t * control_points[
                1] + 3 * (
                                  1 - t) * np.power(t, 2) * control_points[2] + np.power(t, 3) * control_points[3]
            curve_points.append(curve_point)
        curve_points = np.array(curve_points, dtype=np.int32)
        # 绘制头发弯曲线条
        cv2.polylines(img, [curve_points], isClosed=False, color=color, thickness=thickness)
        return img

    def random_gray(self, em):
        if em >= 128:
            B = np.random.randint(0, 30)
            G = np.random.randint(0, 30)
            R = np.random.randint(0, 30)
        else:
            B = np.random.randint(196, 226)
            G = np.random.randint(196, 226)
            R = np.random.randint(196, 226)
        return (B, G, R)

    def one_curve(self, ex, mask, H, W):
        x1 = np.random.randint(20, W - 20)
        y1 = np.random.randint(20, H - 20)
        x2 = max(min(x1 + np.random.randint(-20, 20), W - 1), 0)
        y2 = max(min(y1 + np.random.randint(-20, 20), H - 1), 0)
        x3 = max(min(x2 + np.random.randint(-40, 40), W - 1), 0)
        y3 = max(min(y2 + np.random.randint(-40, 40), H - 1), 0)
        x4 = max(min(x3 + np.random.randint(-80, 80), W - 1), 0)
        y4 = max(min(y3 + np.random.randint(-80, 80), H - 1), 0)
        control_points = np.array([[x1, y1], [x2, y2], [x3, y3], [x4, y4]], dtype=np.int32)
        thickness = np.random.randint(1, 3)
        mask_for_curve = np.zeros_like(mask)
        mask_for_curve = self.generate_curve(mask_for_curve, control_points, 1, thickness)
        mask = self.generate_curve(mask, control_points, 1, thickness)
        ex_mean = ex[mask_for_curve == 1].sum() / np.sum(mask_for_curve == 1)
        ex = self.generate_curve(ex.copy(), control_points, self.random_gray(ex_mean / 3), thickness)
        return ex, mask

    def virtual_light(self, img):
        # 获取图像行和列
        rows, cols = img.shape[:2]
        # 设置中心点和光照半径
        centerX = np.random.randint(50, cols - 50)
        centerY = np.random.randint(50, rows - 50)
        radius = min(centerX, centerY)
        # radius = np.random.randint(50, cols // 4)
        # 设置光照强度
        strength = 0 + np.random.randint(-20, 20)
        x = 1 if np.random.randint(0, 2) == 1 else -1
        # 新建目标图像
        distance = (centerY - np.arange(rows)[:, np.newaxis] - 0.5) ** 2 + \
                   (centerX - np.arange(cols)[np.newaxis, :] - 0.5) ** 2

        # 计算结果矩阵
        result = strength * (1 - np.sqrt(distance) / radius)
        result[distance >= radius ** 2] = 0
        result = np.clip(result * x, -255, 255).astype("int32")

        # 添加结果
        dst = np.clip(img + result[..., np.newaxis], 0, 255).astype("uint8")

        return dst

    def add_noise(self, image):
        # random number
        r = np.random.rand(1)
        # gaussian noise
        if r < 0.9:
            row, col, ch = image.shape
            mean = 0
            var = np.random.rand(1) * 0.3 * 256
            sigma = var ** 0.5
            gauss = sigma * np.random.randn(row, col) + mean
            gauss = np.repeat(gauss[:, :, np.newaxis], ch, axis=2)
            noisy = image + gauss
            noisy = np.clip(noisy, 0, 255)
        else:
            # motion blur
            sizes = [3]
            size = sizes[int(np.random.randint(len(sizes), size=1))]
            kernel_motion_blur = np.zeros((size, size))
            if np.random.rand(1) < 0.5:
                kernel_motion_blur[int((size - 1) / 2), :] = np.ones(size)
            else:
                kernel_motion_blur[:, int((size - 1) / 2)] = np.ones(size)
            kernel_motion_blur = kernel_motion_blur / size
            noisy = cv2.filter2D(image, -1, kernel_motion_blur)

        return noisy

    def add_time_noise(self, image):
        mu = np.random.randint(0, 30) / 10.
        sigma = np.random.randint(0, 30) / 10.
        noise = np.random.normal(mu, sigma, image.shape)
        noisy_image = (image + noise) if np.random.randint(0, 2) == 0 else (image - noise)
        noisy_image = np.clip(noisy_image, 0, 255).astype(np.uint8)
        return noisy_image

    def random_resize(self, img, right, up, fg):
        H, W = img.shape[0], img.shape[1]
        if fg == 0:
            if len(img.shape) == 3:
                img_croped = img[20:H - 20, 20:W - 20, :]
            else:
                img_croped = img[20:H - 20, 20:W - 20]
            img_resized = cv2.resize(img_croped, dsize=(W, H), interpolation=cv2.INTER_NEAREST)
        elif fg == 1:
            img_resized = img
        else:
            img_paded = cv2.copyMakeBorder(img, int(up), int(up), int(right), int(right),
                                           cv2.BORDER_CONSTANT, value=0)
            img_resized = cv2.resize(img_paded, dsize=(W, H), interpolation=cv2.INTER_NEAREST)
        return img_resized.copy()

    def rgb_norm_torch(self, image):
        mean = torch.mean(image, dim=(2, 3), keepdim=True)
        std = torch.std(image, dim=(2, 3), keepdim=True)
        normalized_tensor = (image - mean) / (std + 1e-7)
        return normalized_tensor

    def augumentations(self, input_image, template_image, mask):

        if self.cfg['rot']:
            if np.random.randint(0, 3) == 2:
                angle = np.random.randint(low=-10, high=11) / 4
                mask = self.rotation(img=mask, angle=angle)
                input_image = self.rotation(img=input_image, angle=angle)
            elif np.random.randint(0, 3) == 1:
                angle = np.random.randint(low=-10, high=11) / 4
                template_image = self.rotation(img=template_image, angle=angle)
        if self.cfg['jit']:
            if np.random.randint(0, 3) == 2:
                dx, dy = np.random.randint(low=-5, high=6, size=2)
                input_image = self.random_jitter(input_image, dx, dy)
                mask = self.random_jitter(mask, dx, dy)
            elif np.random.randint(0, 3) == 1:
                dx, dy = np.random.randint(low=-5, high=6, size=2)
                template_image = self.random_jitter(template_image, dx, dy)
        if self.cfg['crop']:
            template_image = self.black_edge_crop(template_image, self.cfg['H'], self.cfg['W'])
            input_image = self.black_edge_crop(input_image, self.cfg['H'], self.cfg['W'])
        if self.cfg['hue']:
            if np.random.randint(0, 5) == 1:
                input_image = self.random_h(input_image)
        if self.cfg['bright']:
            if np.random.randint(0, 5) == 1:
                input_image = self.brightness_adjustment(input_image)
        if self.cfg['chan_change']:
            if np.random.randint(0, 2) == 1:
                sd = np.random.randint(low=0, high=6)
                input_image = self.change_channel(input_image, sd=sd)
                template_image = self.change_channel(template_image, sd=sd)

        if self.cfg['flip']:
            if np.random.randint(0, 2) == 1:
                sd = np.random.randint(low=0, high=6)
                input_image = self.random_flip(input_image, sd)
                template_image = self.random_flip(template_image, sd)
        # diff_image = cv2.imread(self.background_images[np.random.randint(0, len(self.background_images))])
        diff_image = np.asarray(
            Image.open(self.background_images[np.random.randint(0, len(self.background_images))]).convert('RGB'))[:, :,
                     (2, 1, 0)]

        diff_image = cv2.resize(diff_image, dsize=(self.cfg['W'], self.cfg['H']))
        input_image, mask = self.gen_diffs(input_image, mask, diff_image, self.num_diff, self.cfg['H'], self.cfg['W'])

        if self.cfg['curve']:
            input_image, mask = self.one_curve(input_image, mask, self.cfg['H'], self.cfg['W'])
        if self.cfg['label_smoothing']:
            mask[mask == 1] = 0.95
            mask[mask == 0] = 0.05
        if self.cfg['noise']:
            if np.random.randint(0, 10) == 1:
                input_image = self.noisy(noise_typ='gauss', image=input_image)
            elif np.random.randint(0, 10) == 12:
                input_image = self.noisy(noise_typ='s&p', image=input_image)
            elif np.random.randint(0, 10) >= 0:
                input_image = self.add_noise(input_image)
                template_image = self.add_noise(template_image)
        if self.cfg['blur']:
            k = np.random.randint(1, 3) * 2 + 1
            if np.random.randint(0, 4) == 1:
                input_image = cv2.GaussianBlur(input_image, (k, k), 0)
        if self.cfg['light']:
            if np.random.randint(0, 3) == 1:
                input_image = self.virtual_light(input_image)
        fg = np.random.randint(0, 2)
        right = np.random.randint(10, 15)
        up = np.random.randint(10, 15)
        input_image = self.random_resize(input_image, right, up, fg)
        template_image = self.random_resize(template_image, right, up, fg)
        mask = self.random_resize(mask, right, up, fg)

        if self.cfg['time_noise']:
            if np.random.randint(0, 3) == 1:
                input_image = self.add_time_noise(input_image)

        if self.cfg['gray']:
            # if np.random.randint(0, 3) == 1:
            input_image = np.mean(input_image, axis=2, keepdims=True)
            template_image = np.mean(template_image, axis=2, keepdims=True)
            # input_image = cv2.cvtColor(input_image, cv2.COLOR_GRAY2BGR)
            # template_image = cv2.cvtColor(template_image, cv2.COLOR_GRAY2BGR)

        if self.cfg['input_shuffle']:
            if np.random.randint(0, 2) == 0:
                data1 = np.transpose(input_image, axes=(2, 0, 1))
                data2 = np.transpose(template_image, axes=(2, 0, 1))
            else:
                data2 = np.transpose(input_image, axes=(2, 0, 1))
                data1 = np.transpose(template_image, axes=(2, 0, 1))
        else:
            data1 = np.transpose(input_image, axes=(2, 0, 1))
            data2 = np.transpose(template_image, axes=(2, 0, 1))

        if self.cfg['show_input']:
            cv2.imshow('input_image', input_image.astype(np.uint8))
            cv2.imshow('template_image', template_image.astype(np.uint8))
            cv2.imshow('diff', (abs(input_image - template_image)).astype(np.uint8))
            cv2.imshow('mask', mask)
            cv2.waitKey()

        mask = np.transpose(mask[:, :, np.newaxis], axes=(2, 0, 1))
        return np.concatenate((data1, data2), axis=0), mask

    def __len__(self):
        return len(self.background_images)

    def __getitem__(self, index):

        mask = np.zeros((self.cfg['H'], self.cfg['W']))

        # template_image = cv2.imread(self.background_images[index])
        template_image = np.asarray(Image.open(self.background_images[index]).convert('RGB'))[:, :, (2, 1, 0)]

        template_image = cv2.resize(template_image, dsize=(self.cfg['W'], self.cfg['H']))
        input_image = copy.deepcopy(template_image)
        self.num_diff = np.random.randint(low=self.cfg['num_diff'][0], high=self.cfg['num_diff'][1], size=None,
                                          dtype='l')  # 生成差异的数量
        image, label = self.augumentations(input_image, template_image, mask)

        return (torch.from_numpy(np.ascontiguousarray(image)).float(),
                torch.from_numpy(np.ascontiguousarray(label)).float())


if __name__ == '__main__':
    path_setjson = open('./train_config.json', 'r')
    cfg = json.load(path_setjson)
    path_setjson.close()
    train_data = DiffDataset(background_path=r'E:\\yxh\\train2017\\train2017\\', cfg=cfg)
    train_loader = DataLoader(dataset=train_data, batch_size=8, shuffle=True)
    for i, (images, labels) in enumerate(train_loader):
        print(i)
