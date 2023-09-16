import os
from PIL import Image
import numpy as np
data_path = 'D:\\DeepLearning\\Binary_Classification\\dataset'

for phase in os.listdir(data_path):
    phase_list = os.path.join(data_path,phase)
    for image in os.listdir(phase_list):
        img = Image.open(os.path.join(phase_list,image))
        if np.shape(img)[2] != 3:
            print(f"{phase} | {image}'s size is {image}")
    
    print(f"{phase} end")