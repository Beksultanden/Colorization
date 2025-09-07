#!/usr/bin/env python
# coding: utf-8

# In[1]:


from os import listdir
from os.path import isfile, join

import torch
import torch.utils.data as Data
import torchvision.transforms as transforms
from image_util import read_to_pil  # убедись, что эта функция загружает изображения

class CocoFullDataset(Data.Dataset):
    def __init__(self, opt):
        self.IMAGE_DIR = opt.test_img_dir
        self.IMAGE_ID_LIST = [f for f in listdir(self.IMAGE_DIR) if isfile(join(self.IMAGE_DIR, f))]

        self.transforms = transforms.Compose([
            transforms.Resize((opt.fineSize, opt.fineSize), interpolation=2),
            transforms.ToTensor()
        ])

    def __getitem__(self, index):
        image_path = join(self.IMAGE_DIR, self.IMAGE_ID_LIST[index])
        img = read_to_pil(image_path)  # загружаем изображение
        img_tensor = self.transforms(img)

        output = {
            'full_img': img_tensor,
            'file_id': self.IMAGE_ID_LIST[index].split('.')[0]
        }
        return output

    def __len__(self):
        return len(self.IMAGE_ID_LIST)

