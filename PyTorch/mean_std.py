import os
import numpy as np
from numpy.core.fromnumeric import mean, std
from tqdm import tqdm
from scipy.io import loadmat
from tqdm.utils import Comparable

data_path = ''

def compute(data_path):
    image_path = os.path.join(data_path, 'feature')
    image_list = [os.path.join(image_path,i) for i in os.listdir(image_path)]
    per_image_mean = []
    per_image_std = []
    for item in tqdm(image_list):
        image = loadmat(item)['feature']/255
        per_image_mean.append(np.mean(image))
        per_image_std.append(np.std(image))
    mean = np.mean(per_image_mean)
    std = np.mean(per_image_std)
    
    return mean, std

compute(data_path)