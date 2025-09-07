from os import listdir
from os.path import isfile, join
from random import sample

import numpy as np
import torch
import torch.utils.data as Data
import torchvision.transforms as transforms

from image_util import *



class Training_Full_Dataset(Data.Dataset):
    '''
    Training on COCOStuff dataset. [train2017.zip]
    
    Download the training set from https://github.com/nightrome/cocostuff
    '''
    def __init__(self, opt):
        self.IMAGE_DIR = opt.train_img_dir
        self.transforms = transforms.Compose([transforms.Resize((opt.fineSize, opt.fineSize), interpolation=2),
                                              transforms.ToTensor()])
        self.IMAGE_ID_LIST = [f for f in listdir(self.IMAGE_DIR) if isfile(join(self.IMAGE_DIR, f))]

    def __getitem__(self, index):
        output_image_path = join(self.IMAGE_DIR, self.IMAGE_ID_LIST[index])
        rgb_img, gray_img = gen_gray_color_pil(output_image_path)
        output = {}
        output['rgb_img'] = self.transforms(rgb_img)
        output['gray_img'] = self.transforms(gray_img)
        return output

    def __len__(self):
        return len(self.IMAGE_ID_LIST)


class Training_Instance_Dataset(Data.Dataset):
    '''
    Training on COCOStuff dataset. [train2017.zip]
    
    Download the training set from https://github.com/nightrome/cocostuff

    Make sure you've predicted all the images' bounding boxes using inference_bbox.py

    It would be better if you can filter out the images which don't have any box.
    '''
    def __init__(self, opt):
        self.PRED_BBOX_DIR = '{0}_bbox'.format(opt.train_img_dir)
        self.IMAGE_DIR = opt.train_img_dir
        self.IMAGE_ID_LIST = [f for f in listdir(self.IMAGE_DIR) if isfile(join(self.IMAGE_DIR, f))]
        self.transforms = transforms.Compose([
            transforms.Resize((opt.fineSize, opt.fineSize), interpolation=2),
            transforms.ToTensor()
        ])
    
    def __getitem__(self, index):
        pred_info_path = join(self.PRED_BBOX_DIR, self.IMAGE_ID_LIST[index].split('.')[0] + '.npz')
        output_image_path = join(self.IMAGE_DIR, self.IMAGE_ID_LIST[index])
        pred_bbox = gen_maskrcnn_bbox_fromPred(pred_info_path)

        rgb_img, gray_img = gen_gray_color_pil(output_image_path)

        index_list = range(len(pred_bbox))
        index_list = sample(index_list, 1)
        startx, starty, endx, endy = pred_bbox[index_list[0]]
        output = {}
        output['rgb_img'] = self.transforms(rgb_img.crop((startx, starty, endx, endy)))
        output['gray_img'] = self.transforms(gray_img.crop((startx, starty, endx, endy)))
        return output

    def __len__(self):
        return len(self.IMAGE_ID_LIST)





