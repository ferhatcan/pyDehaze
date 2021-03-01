import os
import cv2 as cv
import matplotlib.pyplot as plt
import datetime
for root, dirs, files in os.walk('/media/emre/DATA/Image_Datasets/div2k/DIV2K_train_HR/'):
    start_time = datetime.datetime.now().replace(second=0)
    for file in files:
        image = os.path.join(root, file)
        out = "./generated_images/" + image.split('/')[-1]
        os.system("cd /home/emre/Desktop/example_folder/")
        os.system("gmic -input " + image + " -jeje_clouds 85,0.5 -output " + out)
    print(datetime.datetime.now().replace(second=0) - start_time)
