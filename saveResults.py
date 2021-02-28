import os
import glob

import PIL.Image as Image
import cv2
import numpy as np
import matplotlib.pyplot as plt

from assemblers.assemblerGetter import getExperimentWithDesiredAssembler

CONFIG_FILE_NAME = './configs/EncoderDecoderResnet_v01.ini'
CHALLENGE_VALID_PATH = 'D:/Image_Datasets/dehaze/NTIRE2021_Valid_Hazy'
CHALLENGE_TRAIN_PATH = 'D:/Image_Datasets/dehaze/NTIRE2021_Train_Hazy'

SAVE_PATH = './OUTPUTS/'

def main():
    experiment = getExperimentWithDesiredAssembler(CONFIG_FILE_NAME)
    experiment.load('model_best')

    valid_image_files = glob.glob(CHALLENGE_VALID_PATH + '/*.png')
    train_image_files = glob.glob(CHALLENGE_TRAIN_PATH + '/*.png')

    os.makedirs(os.path.join(SAVE_PATH, 'VALID'), exist_ok=True)
    os.makedirs(os.path.join(SAVE_PATH, 'TRAIN'), exist_ok=True)

    for valid_image_file in valid_image_files:
        img = np.array(Image.open(valid_image_file).convert('RGB'))
        orig_shape = img.shape
        img, padding_dims = experiment.arrange_input_image(img)
        out = experiment.inference(img)

        cropped_out = out[padding_dims[0][0]:orig_shape[0] + padding_dims[0][0],
                          padding_dims[1][0]:orig_shape[1] + padding_dims[0][0],
                          padding_dims[2][0]:orig_shape[2] + padding_dims[0][0]]

        out_pil = Image.fromarray((cropped_out * 255).astype(np.uint8), mode='RGB')
        out_save_path = os.path.join(SAVE_PATH, 'VALID', os.path.split(valid_image_file)[-1])

        out_pil.save(out_save_path)

    for train_image_file in train_image_files:
        img = np.array(Image.open(train_image_file).convert('RGB'))
        orig_shape = img.shape
        img, padding_dims = experiment.arrange_input_image(img)
        out = experiment.inference(img)

        cropped_out = out[padding_dims[0][0]:orig_shape[0] - 1 * padding_dims[0][1],
                          padding_dims[1][0]:orig_shape[1] - 1 * padding_dims[1][1],
                          padding_dims[2][0]:orig_shape[2] - 1 * padding_dims[2][1]]

        out_pil = Image.fromarray((cropped_out * 255).astype(np.uint8), mode='RGB')
        out_save_path = os.path.join(SAVE_PATH, 'TRAIN', os.path.split(train_image_file)[-1])

        out_pil.save(out_save_path)

        # plt.figure()
        # plt.imshow(out_pil)
        # plt.title('Generated Image')
        # plt.waitforbuttonpress()
        #
        # plt.figure()
        # plt.imshow(Image.fromarray(img, mode='RGB'))
        # plt.title('Input Image')
        # plt.waitforbuttonpress()

    tmp = 0

if __name__ == '__main__':
    main()