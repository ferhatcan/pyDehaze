import os

import numpy as np
import cv2
from PIL import Image
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter

# np.random.seed(4)

def gaussian_filter_3d(inp, sigma=5):
    result = []
    for i in range(3):
        result.append(gaussian_filter(inp[:, :, i], sigma=sigma))
    return np.stack(result, axis=-1)

def normalize_image(img):
    max_value, min_value = img.max(), img.min()
    return (img - min_value) / (max_value - min_value)


haze_free_im_path = 'D:\Image_Datasets\div2k\images/train\DIV2K_train_HR/0007.png'
haze_free_im = Image.open(haze_free_im_path).convert('RGB')

image_shape = np.array(haze_free_im).shape

transmission_pathes = np.array((16, 16, 1))
transmission_shape = image_shape // transmission_pathes
transmission_map = np.random.randint(256, size=transmission_shape[:2]) / 255

transmission_map = gaussian_filter(transmission_map, sigma=3)
transmission_map = transmission_map.repeat(transmission_pathes[0], axis=0).repeat(transmission_pathes[1], axis=1)
transmission_map = gaussian_filter(transmission_map, sigma=7)
transmission_map = cv2.resize(transmission_map, dsize=(image_shape[1], image_shape[0]), interpolation=cv2.INTER_NEAREST)


transmission_map = normalize_image(transmission_map)
# transmission_map[transmission_map > 0.85] = 1
transmission_map[transmission_map < 0.2] = 0.15
# transmission_map[np.logical_and(0.5 < transmission_map, transmission_map < 0.7)] = 1
# transmission_map[np.logical_and(0.3 < transmission_map, transmission_map < 0.5)] = 0.5
# transmission_map[np.logical_and(0.0 < transmission_map, transmission_map < 0.3)] = 0.2

transmission_map = np.stack((transmission_map, transmission_map, transmission_map), axis=-1)

plt.figure()
plt.imshow(transmission_map[:,:,0], cmap='gray')

plt.waitforbuttonpress()

A = 1
hazy_im = (np.array(haze_free_im) / 255) * transmission_map +  A * (1 - transmission_map)
# hazy_im = (np.array(haze_free_im) / 255) * 0.5 +  0.5 * (transmission_map * (np.array(haze_free_im) / 255))

plt.figure()
plt.title('Generated Haze Image')
plt.imshow(hazy_im)
plt.waitforbuttonpress()

plt.figure()
plt.title('Haze-Free Image')
plt.imshow(haze_free_im)
plt.waitforbuttonpress()

plt.figure()
plt.title('Difference Image')
plt.imshow(((np.array(hazy_im).astype(np.float) - np.array(haze_free_im).astype(np.float))))
plt.waitforbuttonpress()

tmp = 0