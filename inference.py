import PIL.Image as Image
import cv2
import numpy as np
import matplotlib.pyplot as plt

from assemblers.assemblerGetter import getExperimentWithDesiredAssembler

CONFIG_FILE_NAME = './configs/EncoderDecoderResnet_v01-DIV2K.ini'

def rgb2gray(rgb):
    return np.dot(rgb[...,:3], [0.2989, 0.5870, 0.1140])

def main():
    experiment = getExperimentWithDesiredAssembler(CONFIG_FILE_NAME)
    experiment.load('model_best')

    # img_path = 'D:/Image_Datasets/dehaze/RTTS/RTTS/JPEGImages/BD_Baidu_256.png'
    # img_path = 'D:/Image_Datasets/dehaze/OTS/OTS/0001_0.95_0.16.jpg'
    img_path = 'D:\Image_Datasets\dehaze/NTIRE2021_Train_Hazy/01.png'

    img = np.array(Image.open(img_path).convert('RGB').convert('RGB').resize((512, 512)))

    img, _ = experiment.arrange_input_image(img)


    out = experiment.inference(img)
    # gt_haze_diff = (img / 255 - ref_img / 255)

    out_pil = Image.fromarray((out * 255).astype(np.uint8), mode='RGB').convert('RGB')

    # haze_diff = (haze_diff * 255).astype(np.uint8)
    result = ((img/255 - out) + 1) * 0.5

    # ref_img_path = 'D:/Image_Datasets/dehaze/OTS/OTS/0001_1_0.04.jpg'
    ref_img_path = 'D:\Image_Datasets\dehaze/NTIRE2021_Train_HazeFree/01.png'
    ref_img = np.array(Image.open(ref_img_path).convert('RGB').resize((512, 512)))
    ref_img, _ = experiment.arrange_input_image(ref_img)
    mse = np.mean((ref_img.astype(np.float) - np.array(out_pil).astype(np.float)) ** 2)
    psnr = 20 * np.log10(255.0 / np.sqrt(mse))
    print("MSE is {:.4f}".format(mse / (255**2)))
    print("PSNR is {:.4f}".format(psnr))

    # input_img = Image.open(img_path).convert('RGB').convert('HSV')
    # h = np.array(input_img)[:, :, 0]
    # s = np.array(input_img)[:, :, 1]
    # v = np.array(input_img)[:, :, 2]
    # reference_img = Image.open(ref_img_path).convert('RGB').convert('HSV')
    # ref_h = np.array(reference_img)[:, :, 0]
    # ref_s = np.array(reference_img)[:, :, 1]
    # ref_v = np.array(reference_img)[:, :, 2]
    #
    # diff_h = ref_h.astype(np.float) - h.astype(np.float)
    # diff_s = ref_s.astype(np.float) - s.astype(np.float)
    # diff_v = ref_v.astype(np.float) - v.astype(np.float)
    #
    # Lab = cv2.cvtColor(np.array(input_img), cv2.COLOR_RGB2LAB)
    # ref_Lab = cv2.cvtColor(np.array(reference_img), cv2.COLOR_RGB2LAB)
    #
    # diff_L = Lab[:, :, 0].astype(np.float) - ref_Lab[:, :, 0].astype(np.float)
    # diff_a = Lab[:, :, 1].astype(np.float) - ref_Lab[:, :, 1].astype(np.float)
    # diff_b = Lab[:, :, 2].astype(np.float) - ref_Lab[:, :, 2].astype(np.float)

    plt.figure()
    plt.imshow(ref_img)
    plt.title('Reference Image')
    plt.waitforbuttonpress()


    plt.figure()
    plt.imshow(out_pil)
    plt.title('Generated Image')
    plt.waitforbuttonpress()

    plt.figure()
    plt.imshow(result, cmap='gray')
    plt.title('Difference Image')
    plt.waitforbuttonpress()

    plt.figure()
    plt.imshow(((((img.astype(np.float) - ref_img.astype(np.float))) / 255 + 1) * 0.5), cmap='gray')
    plt.title('Reference Difference Image')
    plt.waitforbuttonpress()

    plt.figure()
    plt.imshow(Image.fromarray(img, mode='RGB').convert('RGB'))
    plt.title('Input Image')
    plt.waitforbuttonpress()

    tmp = 0

if __name__ == '__main__':
    main()