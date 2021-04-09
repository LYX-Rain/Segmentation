import os
import shutil
from tqdm import tqdm

# 将 image 和 mask 数据分开

data_path = '../Data/suichang_round1_train_210120'
image_path = os.path.join(data_path, 'images')
mask_path = os.path.join(data_path, 'masks')

if not os.path.exists(image_path):
    os.makedirs(image_path)
if not os.path.exists(mask_path):
    os.makedirs(mask_path)

image_list = [os.path.join(data_path,i) for i in os.listdir(data_path) if os.path.splitext(i)[-1] == '.tif']
mask_list = [os.path.join(data_path,i) for i in os.listdir(data_path) if os.path.splitext(i)[-1] == '.png']
total = len(image_list)

image_name = [i for i in os.listdir(data_path) if os.path.splitext(i)[-1] == '.tif']
mask_name = [i for i in os.listdir(data_path) if os.path.splitext(i)[-1] == '.png']

for i in tqdm(total):
    shutil.move(os.path.join(data_path,image_name[i]), os.path.join(image_path,image_name[i]))
    shutil.move(os.path.join(data_path,mask_name[i]), os.path.join(mask_path,mask_name[i]))