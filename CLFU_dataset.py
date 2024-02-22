# Dataset for Clean_Fusion Training

import torch
import cv2
import os
import random
from torch.utils import data as data
from torchvision.transforms.functional import normalize
from basicsr.data.transforms import augment, paired_random_crop, random_augmentation
from basicsr.utils import img2tensor, padding, imfrombytes, FileClient
from basicsr.utils.img_util import img2tensory


class TrainDataset(data.Dataset):
    def __init__(self, train_root_dir, scale, gt_size, patch_size, mean, std, transform, geometric_augs=False, is_train=True):
        super().__init__()
        self.train_root_dir = train_root_dir
        self.visible_dir = os.path.join(self.train_root_dir, 'Vis')
        self.infrared_dir = os.path.join(self.train_root_dir, 'Inf')
        self.vis_input_folder = os.path.join(self.visible_dir, 'input')
        self.vis_target_folder = os.path.join(self.visible_dir, 'target')
        self.inf_input_folder = os.path.join(self.infrared_dir, 'input')
        # self.inf_target_folder = os.path.join(self.infrared_dir, 'target')

        self.vis_input = os.listdir(self.vis_input_folder)
        self.vis_target = os.listdir(self.vis_target_folder)
        self.inf_input = os.listdir(self.inf_input_folder)
        # self.inf_target = os.listdir(self.inf_target_folder)
        # self.inf = os.listdir(self.infrared_dir)

        self.scale = scale
        self.gt_size = gt_size
        self.patch_size = patch_size
        self.geometric_augs = geometric_augs
        self.mean = mean
        self.std = std
        self.is_train = is_train
        self.file_client = None
        self.transform = transform
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def __len__(self):
        assert len(self.vis_input) == len(self.inf_input)
        return len(self.vis_input)

    def __getitem__(self, index):
        if self.file_client is None:
            self.file_client = FileClient(backend='disk')
        vis_input_path = os.path.join(self.vis_input_folder, self.vis_input[index])
        vis_target_path = os.path.join(self.vis_target_folder, self.vis_target[index])
        inf_input_path = os.path.join(self.inf_input_folder, self.inf_input[index])
        # inf_target_path = os.path.join(self.inf_target_folder, self.inf_target[index])

        img_bytes = self.file_client.get(vis_input_path, 'vis_input')
        try:
            vis_input = imfrombytes(img_bytes, float32=True)
        except:
            raise Exception("gt path {} not working".format(vis_input_path))

        vis_input_1 = vis_input.copy()

        img_bytes = self.file_client.get(vis_target_path, 'vis_target')
        try:
            vis_target = imfrombytes(img_bytes, float32=True)
        except:
            raise Exception("gt path {} not working".format(vis_target_path))

        inf_input = cv2.imread(inf_input_path)
        inf_input = cv2.cvtColor(inf_input, cv2.COLOR_BGR2YCrCb)

        # augmentation setting
        if self.is_train:
            # padding
            vis_target, vis_input = padding(vis_target, vis_input, self.gt_size)

            # random crop
            vis_target, vis_input = paired_random_crop(vis_target, vis_input, self.gt_size, self.scale, vis_target_path)

            # flip, rotation augmentations
            if self.geometric_augs:
                vis_input, vis_target = random_augmentation(vis_input, vis_target)

        # inf_target, inf_input = self.get_patch(inf_target, inf_input)
        inf_input = self.get_patch(inf_input)

        # BGR to RGB, HWC to CHW, numpy to tensor
        vis_target, vis_input = img2tensor([vis_target, vis_input], bgr2rgb=True, float32=True)
        # inf_target, inf_input = img2tensory([inf_target, inf_input], bgr2ycbcr=True, float32=True)
        vis_input_1 = img2tensor(vis_input_1, bgr2rgb=True, float32=True)
        # inf_input = img2tensor(inf_input, bgr2rgb=True, float32=True)

        # normalize
        if self.mean is not None and self.std is not None:
            normalize(vis_target, self.mean, self.std, inplace=True)
            normalize(vis_input, self.mean, self.std, inplace=True)

        if self.transform is not None:
            inf_input = self.transform(inf_input)

        # return vis_input, vis_target, inf_input, inf_target

        return {'vis_input': vis_input,  'vis_target': vis_target,
                'vis_input_path': vis_input_path, 'vis_target_path': vis_target_path,
                'inf_input': inf_input, #'inf_target': inf_target,
                'vis_input_1': vis_input_1}


    # def get_patch(self, target_img, input_img):
    #     h, w = target_img.shape[:2]
    #     stride = self.patch_size
    #
    #     x = random.randint(0, w - stride)
    #     y = random.randint(0, h - stride)
    #
    #     target_img = target_img[y:y + stride, x:x + stride, :]
    #     input_img = input_img[y:y + stride, x:x + stride, :]
    #
    #     return target_img, input_img

    def get_patch(self, input_img):
        h, w = input_img.shape[:2]
        stride = self.patch_size

        x = random.randint(0, w - stride)
        y = random.randint(0, h - stride)

        input_img = input_img[y:y + stride, x:x + stride, :]

        return input_img

class CombineDataset(data.Dataset):
    def __init__(self, train_root_dir, scale, gt_size, patch_size, mean, std, geometric_augs=False, is_train=True):
        super().__init__()
        self.train_root_dir = train_root_dir
        self.visible_dir = os.path.join(self.train_root_dir, 'Vis')
        self.infrared_dir = os.path.join(self.train_root_dir, 'Inf')
        self.vis_input_folder = os.path.join(self.visible_dir, 'input')
        self.vis_target_folder = os.path.join(self.visible_dir, 'target')
        self.inf_input_folder = os.path.join(self.infrared_dir, 'input')

        self.vis_input = os.listdir(self.vis_input_folder)
        self.vis_target = os.listdir(self.vis_target_folder)
        self.inf_input = os.listdir(self.inf_input_folder)
        # self.inf = os.listdir(self.infrared_dir)

        self.scale = scale
        self.gt_size = gt_size
        self.patch_size = patch_size
        self.geometric_augs = geometric_augs
        self.mean = mean
        self.std = std
        self.is_train = is_train
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def __len__(self):
        assert len(self.vis_input) == len(self.inf_input)
        return len(self.vis_input)

    def __getitem__(self, index):
        vis_input_path = os.path.join(self.vis_input_folder, self.vis_input[index])
        vis_target_path = os.path.join(self.vis_target_folder, self.vis_target[index])
        inf_input_path = os.path.join(self.inf_input_folder, self.inf_input[index])

        vis_input = cv2.imread(vis_input_path)
        vis_target = cv2.imread(vis_target_path)
        inf_input = cv2.imread(inf_input_path)
        # augmentation setting
        if self.is_train:
            # padding
            vis_target, vis_input = padding(vis_target, vis_input, self.gt_size)

            # random crop
            vis_target, vis_input = paired_random_crop(vis_target, vis_input, self.gt_size, self.scale, vis_target_path)

            # flip, rotation augmentations
            if self.geometric_augs:
                vis_input, vis_target = random_augmentation(vis_input, vis_target)

            inf_input = self.get_patch(inf_input)

        # BGR to RGB, HWC to CHW, numpy to tensor
        vis_target, vis_input = img2tensor([vis_target, vis_input], bgr2rgb=True, float32=True)
        inf_input = img2tensor([inf_input], bgr2rgb=True, float32=True)

        # normalize
        if self.mean is not None and self.std is not None:
            normalize(vis_target, self.mean, self.std, inplace=True)
            normalize(vis_input, self.mean, self.std, inplace=True)
            normalize(inf_input, self.mean, self.std, inplace=True)
        data_task1 = {
            'vis_input': vis_input,
            'vis_target': vis_target,
        }

        data_task2 = {
            'vis_derained': vis_input,  # Use the derained image from task 1 as input for task 2
            'inf_input': inf_input,
        }

        return data_task1, data_task2

    def get_patch(self, input_img):
        h, w = input_img.shape[:2]
        stride = self.patch_size

        x = random.randint(0, w - stride)
        y = random.randint(0, h - stride)

        input_img = input_img[y:y + stride, x:x + stride, :]

        return input_img


class TestDataset(data.Dataset):
    def __init__(self, test_root_dir, mean, std, transform):
        super().__init__()
        self.test_root_dir = test_root_dir
        self.mean = mean
        self.std = std
        self.file_client = None
        self.transform = transform

        self.visible_dir = os.path.join(self.test_root_dir, 'Vis')
        self.infrared_dir = os.path.join(self.test_root_dir, 'Inf')
        self.vis_input_folder = os.path.join(self.visible_dir, 'input')
        self.vis_target_folder = os.path.join(self.visible_dir, 'target')
        self.inf_input_folder = os.path.join(self.infrared_dir, 'input')
        # self.inf_target_folder = os.path.join(self.infrared_dir, 'target')

        self.vis_input = os.listdir(self.vis_input_folder)
        self.vis_target = os.listdir(self.vis_target_folder)
        self.inf_input = os.listdir(self.inf_input_folder)
        # self.inf_target = os.listdir(self.inf_target_folder)

    def __len__(self):
        assert len(self.vis_input) == len(self.inf_input)
        return len(self.vis_input)

    def __getitem__(self, index):
        if self.file_client is None:
            self.file_client = FileClient(backend='disk')

        vis_input_path = os.path.join(self.vis_input_folder, self.vis_input[index])
        vis_target_path = os.path.join(self.vis_target_folder, self.vis_target[index])
        inf_input_path = os.path.join(self.inf_input_folder, self.inf_input[index])
        # inf_target_path = os.path.join(self.inf_target_folder, self.inf_target[index])

        img_bytes = self.file_client.get(vis_input_path, 'vis_input')
        try:
            vis_input = imfrombytes(img_bytes, float32=True)
        except:
            raise Exception("gt path {} not working".format(vis_input_path))

        img_bytes = self.file_client.get(vis_target_path, 'vis_target')
        try:
            vis_target = imfrombytes(img_bytes, float32=True)
        except:
            raise Exception("gt path {} not working".format(vis_target_path))

        inf_input = cv2.imread(inf_input_path)
        inf_input = cv2.cvtColor(inf_input, cv2.COLOR_BGR2YCrCb)

        # img_bytes = self.file_client.get(inf_target_path, 'inf_target')
        # try:
        #     inf_target = imfrombytes(img_bytes, float32=True)
        # except:
        #     raise Exception("gt path {} not working".format(inf_target_path))

        # BGR to RGB, HWC to CHW, numpy to tensor
        vis_target, vis_input = img2tensor([vis_target, vis_input], bgr2rgb=True, float32=True)
        # inf_target, inf_input = img2tensory([inf_target, inf_input], bgr2ycbcr=True, float32=True)

        # normalize
        if self.mean is not None and self.std is not None:
            normalize(vis_target, self.mean, self.std, inplace=True)
            normalize(vis_input, self.mean, self.std, inplace=True)
        if self.transform:
            inf_input = self.transform(inf_input)

        # return vis_input, vis_target, inf_input, inf_target

        return {'vis_input': vis_input, 'vis_target': vis_target,
                'vis_input_path': vis_input_path, 'vis_target_path': vis_target_path,
                'inf_input': inf_input#, 'inf_target': inf_target
                }

    def get_filenames(self):
        return self.vis_input      # 返回RGB图片文件名