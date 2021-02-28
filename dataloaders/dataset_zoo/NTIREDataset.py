import os
import glob
import random
import numpy as np
import torch
from torchvision import transforms
from torchvision.transforms import functional as tvF

import PIL.Image as Image
import cv2
from scipy.ndimage import gaussian_filter

from dataloaders.IDataset import IDataset

class NTIREDataset(IDataset):
    def __init__(self, args, train=False):
        super(NTIREDataset, self).__init__()

        self.train = train
        self.input_shape = args.input_shape if hasattr(args, 'input_shape') else [256, 256]
        self.normalize = args.normalize if hasattr(args, 'normalize') else 'between1-1'

        self.image_paths = args.train_set_paths if train else args.test_set_paths

        self.image_files = []
        self.reference_files = []

        self.random_num = random.random()

        self.extractImageFiles()

    def extractImageFiles(self):
        for path in self.image_paths:
            self.image_files.extend(glob.glob(path + 'hazy/*.*'))
        for index, image_path in enumerate(self.image_files):
            path_parts = os.path.split(image_path)
            gt_path = os.path.join(os.path.split(path_parts[0])[0], 'GT', path_parts[-1])
            if os.path.exists(gt_path):
                self.reference_files.append(gt_path)
            else:
                del self.image_files[index]

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, item: int) -> dict:
        im_ref = Image.open(self.image_files[item]).convert('RGB')
        im_haze = Image.open(self.reference_files[item]).convert('RGB')

        resize = transforms.Resize(size=self.input_shape)
        im_haze = resize(im_haze)
        im_ref = resize(im_ref)

        # # random crop
        # crop = transforms.RandomCrop(size=self.input_shape)
        # i, j, h, w = crop.get_params(im_ref, self.input_shape)
        # im_ref = tvF.crop(im_ref, i, j, h, w)
        # im_haze = tvF.crop(im_haze, i, j, h, w)

        return self.fillOutputDataDict(*self.transform_multi(im_haze, im_ref))

    def transform_multi(self, im_haze, im_ref):
        if im_haze.size[0] < self.input_shape[0] or im_haze.size[1] < self.input_shape[1]:
            resize = transforms.Resize(size=self.input_shape)
            im_haze = resize(im_haze)
            im_ref = resize(im_ref)

        self.random_num = random.random()

        if not im_ref is None:
            haze_diff = (np.array(im_haze) / 255 - np.array(im_ref) / 255).astype(np.float32)
            return self.transform(im_haze), self.transform(im_ref), self.transform(haze_diff, normalize=False)

        return self.transform(im_haze), torch.tensor([]), torch.tensor([])

    def transform(self, image, normalize=True):
        torch_image = tvF.to_tensor(image)
        if normalize:
            if self.normalize == 'between1-1':
                torch_image = torch_image * 2 - 1

        # random horizontal flip
        if self.random_num > 0.7:
            torch_image = tvF.hflip(torch_image)

        # random vertical flip
        if self.random_num < 0.3:
            torch_image = tvF.vflip(torch_image)

        return torch_image


# from utils.configParser import options
# import matplotlib.pyplot as plt
#
# plt.ion()
#
# all_args = options('../../configs/EncoderDecoder_v01.ini')
# all_args.argsDataset.train_set_paths = ['D:\Image_Datasets\dehaze/NTIRE2021_Train/',
#                                         'D:\Image_Datasets\dehaze\Dense_Haze_NTIRE19/',
#                                         'D:\Image_Datasets\dehaze\O-HAZE/']
# all_args.argsDataset.input_shape = [1024, 1024]
# ntire = NTIREDataset(all_args.argsDataset, train=True)
#
# print(len(ntire))
#
# data = ntire.__getitem__(3)
#
# input_im = ((data['inputs'] + 1) * 0.5 * 255).numpy().transpose((1, 2, 0)).astype(np.uint8)
# gts_im = ((data['gts'] + 1) * 0.5 * 255).numpy().transpose((1, 2, 0)).astype(np.uint8)
# reference_im = ((data['original'] + 1) * 0.5 * 255).numpy().transpose((1, 2, 0)).astype(np.uint8)
#
# plt.figure()
# plt.imshow(Image.fromarray(input_im).convert('RGB'))
# plt.waitforbuttonpress()
# plt.figure()
# plt.imshow(Image.fromarray(gts_im).convert('RGB'))
# plt.waitforbuttonpress()
# plt.figure()
# plt.imshow(Image.fromarray(reference_im).convert('RGB'))
# plt.waitforbuttonpress()
#
# tmp = 0