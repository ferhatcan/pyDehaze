import os
import glob
import numpy as np
import torch
from torchvision import transforms
from torchvision.transforms import functional as tvF

import PIL.Image as Image
import cv2
from scipy.ndimage import gaussian_filter

from dataloaders.IDataset import IDataset


class DIV2KDataset(IDataset):
    def __init__(self, args, train=False):
        super(DIV2KDataset, self).__init__()

        self.train = train
        self.input_shape = args.input_shape if hasattr(args, 'input_shape') else [256, 256]
        self.normalize = args.normalize if hasattr(args, 'normalize') else 'between1-1'

        self.image_paths = args.train_set_paths if train else args.test_set_paths

        self.image_files = []

        self.extractImageFiles()

    def extractImageFiles(self):
        self.image_files = glob.glob(self.image_paths[0] + '*.png')

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, item: int) -> dict:
        im_ref = Image.open(self.image_files[item]).convert('RGB')
        im_haze = self.construct_haze_image(im_ref)
        # random crop
        crop = transforms.RandomCrop(size=self.input_shape)
        i, j, h, w = crop.get_params(im_ref, self.input_shape)
        im_ref = tvF.crop(im_ref, i, j, h, w)
        im_haze = tvF.crop(im_haze, i, j, h, w)

        return self.fillOutputDataDict(*self.transform_multi(im_haze, im_ref))

    def construct_haze_image(self, im_ref):
        image_shape = np.array(im_ref).shape

        transmission_pathes = np.array((30, 30, 1))
        transmission_shape = image_shape // transmission_pathes
        transmission_map = np.random.randint(256, size=transmission_shape[:2]) / 255

        transmission_map = gaussian_filter(transmission_map, sigma=3)
        transmission_map = transmission_map.repeat(transmission_pathes[0], axis=0).repeat(transmission_pathes[1],
                                                                                          axis=1)
        transmission_map = gaussian_filter(transmission_map, sigma=7)
        transmission_map = cv2.resize(transmission_map, dsize=(image_shape[1], image_shape[0]),
                                      interpolation=cv2.INTER_NEAREST)

        transmission_map = self._normalize_image(transmission_map)
        # transmission_map[transmission_map > 0.85] = 1
        transmission_map[transmission_map < 0.2] = 0.15

        transmission_map = np.stack((transmission_map, transmission_map, transmission_map), axis=-1)

        A = 1
        hazy_im = (np.array(im_ref) / 255) * transmission_map + A * (1 - transmission_map)
        im_haze = Image.fromarray((hazy_im * 255).astype(np.uint8)).convert('RGB')

        return im_haze

    @staticmethod
    def _normalize_image(img):
        max_value, min_value = img.max(), img.min()
        return (img - min_value) / (max_value - min_value)

    def transform_multi(self, im_haze, im_ref):
        if im_haze.size[0] < self.input_shape[0] or im_haze.size[1] < self.input_shape[1]:
            resize = transforms.Resize(size=self.input_shape)
            im_haze = resize(im_haze)
            im_ref = resize(im_ref)

        if not im_ref is None:
            haze_diff = (np.array(im_haze) / 255 - np.array(im_ref) / 255).astype(np.float32)
            return self.transform(im_haze), self.transform(im_ref), self.transform(haze_diff, normalize=False)

        return self.transform(im_haze), torch.tensor([]), torch.tensor([])

    def transform(self, image, normalize=True):
        torch_image = tvF.to_tensor(image)
        if normalize:
            if self.normalize == 'between1-1':
                torch_image = torch_image * 2 - 1

        return torch_image

#
# from utils.configParser import options
# import matplotlib.pyplot as plt
#
# plt.ion()
#
# all_args = options('../../configs/EncoderDecoder_v01.ini')
# all_args.argsDataset.train_set_paths = ['D:/Image_Datasets/div2k/images/train/DIV2K_train_HR/']
# all_args.argsDataset.input_shape = [1024, 1024]
# div2k = DIV2KDataset(all_args.argsDataset, train=True)
#
# print(len(div2k))
#
# data = div2k.__getitem__(3)
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