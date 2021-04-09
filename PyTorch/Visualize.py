from torchvision.utils import draw_segmentation_masks
from PIL import Image
import os

data_path = ""
image_path = os.path.join(data_path, 'images')
output_path = os.path.join(data_path, 'results')
image_list = [os.path.join(image_path,i) for i in os.listdir(image_path)]
mask_list = [os.path.join(output_path,i) for i in os.listdir(output_path)]

for i in range(len(image_list)):
    result = draw_segmentation_masks(image_list[i], mask_list[i], alpha=0.2)