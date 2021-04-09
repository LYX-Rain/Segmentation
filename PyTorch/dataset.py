import os
from PIL import Image
from torch.utils.data import Dataset

class MyDataset(Dataset):
    def __init__(self, data_path, transform=None):
        image_path = os.path.join(data_path, 'images')
        mask_path = os.path.join(data_path, 'masks')
        self.image_list = [os.path.join(image_path,i) for i in os.listdir(image_path)]
        self.mask_list = [os.path.join(mask_path,i) for i in os.listdir(mask_path)]
        self.transform = transform

        # self.label_names = ['耕地', '林地', '草地', '道路', '城镇建设用地', '农村建设用地', '工业用地', '构筑物', '水域', '裸地']
        # self.label_to_index = dict((name, index) for index, name in enumerate(self.label_names))  # 为每个标签分配索引
    
    def __getitem__(self, idx):
        image = Image.open(self.image_list[idx])
        mask = Image.open(self.mask_list[idx])

        if self.transform is not None:
            image, mask = self.transform(image, mask)
        mask -= 1
        return image, mask
    
    def __len__(self):
        return len(self.image_list)