#!/usr/bin/env python
# coding: utf-8

# In[ ]:

#!/usr/bin/env python

import os
from os.path import join
import torch
from tqdm import tqdm
from options.test_options import TestOptions  # Используем test_options.py
from models import create_model
from util import util
from coco_full_dataset import CocoFullDataset  # Убедитесь, что этот файл существует

# Используем первую доступную GPU
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
torch.backends.cudnn.benchmark = True

if __name__ == '__main__':
    # Загружаем параметры тестирования
    opt = TestOptions().parse()
    
    # Создаём папку для сохранения результатов, если её нет
    save_img_path = opt.results_img_dir
    os.makedirs(save_img_path, exist_ok=True)
    
    # Указываем размер batch'а (должен быть 1 для обработки изображений по одному)
    opt.batch_size = 1

    # Загружаем тестовый датасет
    dataset = CocoFullDataset(opt)
    dataset_loader = torch.utils.data.DataLoader(dataset, batch_size=1, num_workers=2)

    print(f'# Testing images = {len(dataset)}')

    # Создаём модель и загружаем веса первого этапа (Full Colorization)
    model = create_model(opt)
    model.setup_to_test('coco_full')  # Загрузка весов модели

    # Проходим по датасету и раскрашиваем изображения
    for data_raw in tqdm(dataset_loader, dynamic_ncols=True):
        # Подготовка данных
        img_data = util.get_colorization_data(data_raw['full_img'], opt, ab_thresh=0, p=opt.sample_p)
        
        # Установка входных данных и выполнение модели
        model.set_input(img_data)
        model.forward()
        
        # Сохранение результата
        model.save_current_imgs(join(save_img_path, data_raw['file_id'][0] + '.png'))

    print(f'Colorized images saved in {save_img_path}')

'''

import os
from os.path import join
import torch
from tqdm import tqdm
from options.test_options import TestOptions
from models import create_model
from util import util
from coco_full_dataset import CocoFullDataset

# Используем первую доступную GPU
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
torch.backends.cudnn.benchmark = True

if __name__ == '__main__':
    opt = TestOptions().parse()
    
    save_img_path = opt.results_img_dir
    os.makedirs(save_img_path, exist_ok=True)
    
    opt.batch_size = 1

    dataset = CocoFullDataset(opt)
    dataset_loader = torch.utils.data.DataLoader(dataset, batch_size=1, num_workers=2)

    print(f'# Testing images = {len(dataset)}')

    # **Измеряем память перед загрузкой модели**
    print(f"Memory allocated before model load: {torch.cuda.memory_allocated() / 1024**2:.2f} MB")
    print(f"Memory reserved before model load: {torch.cuda.memory_reserved() / 1024**2:.2f} MB")

    # Создаём модель и загружаем веса
    model = create_model(opt)
    model.setup_to_test('coco_full')  

    # **Измеряем память после загрузки модели**
    print(f"Memory allocated after model load: {torch.cuda.memory_allocated() / 1024**2:.2f} MB")
    print(f"Memory reserved after model load: {torch.cuda.memory_reserved() / 1024**2:.2f} MB")

    for data_raw in tqdm(dataset_loader, dynamic_ncols=True):
        img_data = util.get_colorization_data(data_raw['full_img'], opt, ab_thresh=0, p=opt.sample_p)
        
        model.set_input(img_data)

        # **Измеряем память перед инференсом**
        print(f"Memory allocated before inference: {torch.cuda.memory_allocated() / 1024**2:.2f} MB")
        print(f"Memory reserved before inference: {torch.cuda.memory_reserved() / 1024**2:.2f} MB")
        
        torch.cuda.empty_cache()

        model.forward()

        # **Измеряем память после инференса**
        print(f"Memory allocated after inference: {torch.cuda.memory_allocated() / 1024**2:.2f} MB")
        print(f"Memory reserved after inference: {torch.cuda.memory_reserved() / 1024**2:.2f} MB")

        model.save_current_imgs(join(save_img_path, data_raw['file_id'][0] + '.png'))

    print(f'Colorized images saved in {save_img_path}')
'''

