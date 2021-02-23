import PIL.Image as Image
import numpy as np
import matplotlib.pyplot as plt

from assemblers.assemblerGetter import getExperimentWithDesiredAssembler

CONFIG_FILE_NAME = './configs/EncoderDecoder_v02-includeInput.ini'

def rgb2gray(rgb):
    return np.dot(rgb[...,:3], [0.2989, 0.5870, 0.1140])

def main():
    experiment = getExperimentWithDesiredAssembler(CONFIG_FILE_NAME)
    experiment.load('model_best')

    # img_path = 'D:/Image_Datasets/dehaze/RTTS/RTTS/JPEGImages/BD_Baidu_256.png'
    img_path = 'D:/Image_Datasets/dehaze/OTS/OTS/0001_0.95_0.16.jpg'

    img = np.array(Image.open(img_path).convert('RGB'))

    img = experiment.arrange_input_image(img)


    out = experiment.inference(img)
    # gt_haze_diff = (img / 255 - ref_img / 255)



    # haze_diff = (haze_diff * 255).astype(np.uint8)
    result = ((img/255 - out) * 255).astype(np.uint8)

    ref_img_path = 'D:/Image_Datasets/dehaze/OTS/OTS/0001_1_0.04.jpg'
    ref_img = np.array(Image.open(ref_img_path).convert('RGB'))
    ref_img = experiment.arrange_input_image(ref_img)
    mse = np.mean((ref_img - out * 255) ** 2)
    psnr = 20 * np.log10(255.0 / np.sqrt(mse))
    print("MSE is {:.4f}".format(mse / 255**2))
    print("PSNR is {:.4f}".format(psnr))

    plt.figure()
    plt.imshow(ref_img)
    plt.title('Reference Image')
    plt.waitforbuttonpress()


    plt.figure()
    plt.imshow(out)
    plt.title('Generated Image')
    plt.waitforbuttonpress()

    plt.figure()
    plt.imshow(rgb2gray(result.astype(np.uint8)), cmap='gray')
    plt.title('Difference Image')
    plt.waitforbuttonpress()

    plt.figure()
    plt.imshow(img)
    plt.title('Input Image')
    plt.waitforbuttonpress()

    tmp = 0

if __name__ == '__main__':
    main()