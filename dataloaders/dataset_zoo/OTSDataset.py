import os
import glob
import numpy as np
import torch
from torchvision import transforms
from torchvision.transforms import functional as tvF

import PIL.Image as Image

from dataloaders.IDataset import IDataset


class OTSDataset(IDataset):
    def __init__(self, args, train=False):
        super(OTSDataset, self).__init__()

        self.train = train
        self.input_shape = args.input_shape if hasattr(args, 'input_shape') else [256, 256]
        self.normalize = args.normalize if hasattr(args, 'normalize') else 'between1-1'

        self.image_paths = args.train_set_paths if train else args.test_set_paths
        self.extensions = ["jpg", "jpeg", "png"]

        self.image_files = []
        self.reference_files = []

        self.extractImageFiles()

    def extractImageFiles(self):
        # reference image parameters
        A = 1
        beta = 0.04

        self.image_files = glob.glob(self.image_paths[0] + '*.jpg')
        for image_path in self.image_files:
            self.reference_files.append(os.path.join(image_path.split('\\')[0],
                                                     image_path.split('\\')[-1].split('_')[0] + '_' + str(A) + '_' + str(beta) + '.jpg'))

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, item: int) -> dict:
        im_haze = Image.open(self.image_files[item]).convert('RGB')
        if self.train:
            im_ref = Image.open(self.reference_files[item]).convert('RGB')
        else:
            im_ref = None

        return self.fillOutputDataDict(*self.transform_multi(im_haze, im_ref))

    def transform_multi(self, im_haze, im_ref):
        if im_haze.size[0] < self.input_shape[0] or im_haze.size[1] < self.input_shape[1]:
            resize = transforms.Resize(size=self.input_shape)
            im_haze = resize(im_haze)
            im_ref = resize(im_ref)

        # random crop
        crop = transforms.RandomCrop(size=self.input_shape)
        i, j, h, w = crop.get_params(im_haze, self.input_shape)
        haze_im = tvF.crop(im_haze, i, j, h, w)
        if not im_ref is None:
            ref_im = tvF.crop(im_ref, i, j, h, w)
            haze_diff = (np.array(haze_im) / 255 - np.array(ref_im) / 255).astype(np.float32)
            return self.transform(haze_im), self.transform(ref_im), self.transform(haze_diff, normalize=False)

        return self.transform(haze_im), torch.tensor([]), torch.tensor([])

    def transform(self, image, normalize=True):
        torch_image = tvF.to_tensor(image)
        if normalize:
            if self.normalize == 'between1-1':
                torch_image = torch_image * 2 - 1

        return torch_image


# from utils.configParser import options
# import matplotlib.pyplot as plt
#
# plt.ion()
#
# all_args = options('../../configs/EncoderDecoder_v01.ini')
# ots = OTSDataset(all_args.argsDataset, train=True)
#
# print(len(ots))
#
# data = ots.__getitem__(313000)
#
# input_im = ((data['inputs'] + 1) * 0.5).numpy().transpose((1, 2, 0))
# gts_im = ((data['gts'] + 1) * 0.5).numpy().transpose((1, 2, 0))
# reference_im = ((data['original'] + 1) * 0.5).numpy().transpose((1, 2, 0))
#
# plt.figure()
# plt.imshow(input_im)
# plt.waitforbuttonpress()
# plt.figure()
# plt.imshow(gts_im)
# plt.waitforbuttonpress()
# plt.figure()
# plt.imshow(reference_im)
# plt.waitforbuttonpress()
#
# tmp = 0