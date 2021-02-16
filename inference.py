import PIL.Image as Image
import numpy as np
import matplotlib.pyplot as plt

from assemblers.assemblerGetter import getExperimentWithDesiredAssembler

CONFIG_FILE_NAME = './configs/EncoderDecoder_v01.ini'

def main():
    experiment = getExperimentWithDesiredAssembler(CONFIG_FILE_NAME)
    experiment.load('model_best')

    img_path = 'D:/Image_Datasets/dehaze/RTTS/RTTS/JPEGImages/BD_Baidu_256.png'
    # ref_img_path = 'D:/Image_Datasets/dehaze/OTS/OTS/0001_1_0.04.jpg'
    img = np.array(Image.open(img_path).convert('RGB'))
    # ref_img = np.array(Image.open(ref_img_path).convert('RGB'))
    img = experiment.arrange_input_image(img)
    # ref_img = experiment.arrange_input_image(ref_img)

    haze_diff = experiment.inference(img)
    # gt_haze_diff = (img / 255 - ref_img / 255)

    # mse = np.mean((gt_haze_diff - haze_diff) ** 2)

    # haze_diff = (haze_diff * 255).astype(np.uint8)
    result = ((img / 255 - haze_diff) * 255).astype(np.uint8)

    plt.figure()
    plt.imshow(haze_diff)
    plt.waitforbuttonpress()

    plt.figure()
    plt.imshow(result.astype(np.uint8))
    plt.waitforbuttonpress()

    plt.figure()
    plt.imshow(img)
    plt.waitforbuttonpress()

    tmp = 0

if __name__ == '__main__':
    main()