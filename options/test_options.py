#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import argparse
import os
import torch
from util import util

class TestOptions:
    def __init__(self):
        self.initialized = False

    def initialize(self, parser):
        # Основные параметры
        parser.add_argument('--batch_size', type=int, default=1, help='Размер батча')
        parser.add_argument('--loadSize', type=int, default=256, help='Масштабирование изображений до этого размера')
        parser.add_argument('--fineSize', type=int, default=256, help='Обрезка изображений до этого размера')
        parser.add_argument('--input_nc', type=int, default=1, help='Количество каналов входного изображения')
        parser.add_argument('--output_nc', type=int, default=2, help='Количество каналов выходного изображения')
        parser.add_argument('--ngf', type=int, default=64, help='Количество фильтров в первом слое генератора')
        parser.add_argument('--gpu_ids', type=str, default='0', help='ID GPU: например, 0 или 0,1,2. Используйте -1 для CPU')
        parser.add_argument('--dataset_mode', type=str, default='aligned', help='Режим загрузки данных: aligned, unaligned, single')
        parser.add_argument('--which_direction', type=str, default='AtoB', help='Направление: AtoB или BtoA')
        parser.add_argument('--nThreads', type=int, default=4, help='Количество потоков для загрузки данных')
        parser.add_argument('--checkpoints_dir', type=str, default='./checkpoints', help='Директория для сохранения моделей')
        parser.add_argument('--norm', type=str, default='batch', help='Тип нормализации: batch или instance')
        parser.add_argument('--serial_batches', action='store_true', help='Использовать ли последовательные батчи')
        parser.add_argument('--no_flip', action='store_true', help='Не отражать изображения для аугментации')
        parser.add_argument('--init_type', type=str, default='normal', help='Тип инициализации сети: normal, xavier, kaiming, orthogonal')
        parser.add_argument('--results_dir', type=str, default='./results/', help='Директория для сохранения результатов')
        parser.add_argument('--phase', type=str, default='test', help='Фаза: train, val, test')
        parser.add_argument('--which_epoch', type=str, default='latest', help='Эпоха для загрузки: latest или номер')
        parser.add_argument('--how_many', type=int, default=200, help='Количество тестовых изображений')
        parser.add_argument('--aspect_ratio', type=float, default=1.0, help='Соотношение сторон результата')
        parser.add_argument('--load_model', action='store_true', help='Загрузить последнюю модель')
        parser.add_argument('--name', type=str, default='coco_full', help='Имя эксперимента')
        parser.add_argument('--test_img_dir', type=str, default='example/', help='Директория с тестовыми изображениями')
        parser.add_argument('--results_img_dir', type=str, default='./results/', help='Директория для сохранения результатов')
        parser.add_argument('--suffix', type=str, default='', help='Суффикс для имени эксперимента')
        parser.add_argument('--model', type=str, default='coco_full', help='Название модели')
        parser.add_argument('--resize_or_crop', type=str, default='resize_and_crop',  help=' resize_and_crop, crop, scale_width, scale_width_and_crop')
        parser.add_argument('--no_dropout', action='store_true', help='Отключить dropout в генераторе')
        
        parser.add_argument('--classification', action='store_true', help='backprop trunk using classification, otherwise use regression')
       
       
        
        

        
        parser.add_argument('--half', action='store_true', help='half precision model')
                      


        # Параметры для цветности
        parser.add_argument('--ab_norm', type=float, default=110., help='Нормализация цветности')
        parser.add_argument('--ab_max', type=float, default=110., help='Максимальное значение ab')
        parser.add_argument('--ab_quant', type=float, default=10., help='Квантование цветности')
        parser.add_argument('--l_norm', type=float, default=100., help='Нормализация яркости')
        parser.add_argument('--l_cent', type=float, default=50., help='Центрирование яркости')
        parser.add_argument('--mask_cent', type=float, default=.5, help='Центрирование маски')
        parser.add_argument('--sample_p', type=float, default=1.0, help='Вероятность сэмплирования')
        parser.add_argument('--sample_Ps', type=int, nargs='+', default=[1, 2, 3, 4, 5, 6, 7, 8, 9], help='Размеры патчей')

        self.initialized = True
        return parser

    def gather_options(self):
        if not self.initialized:
            parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
            parser = self.initialize(parser)
        opt, _ = parser.parse_known_args()
        self.parser = parser
        return parser.parse_args()

    def print_options(self, opt):
        message = ''
        message += '----------------- Options ---------------\n'
        for k, v in sorted(vars(opt).items()):
            comment = ''
            default = self.parser.get_default(k)
            if v != default:
                comment = '\t[default: %s]' % str(default)
            message += '{:>25}: {:<30}{}\n'.format(str(k), str(v), comment)
        message += '----------------- End -------------------'
        print(message)

        # Сохранение в файл
        expr_dir = os.path.join(opt.checkpoints_dir, opt.name)
        util.mkdirs(expr_dir)
        file_name = os.path.join(expr_dir, 'opt.txt')
        with open(file_name, 'wt') as opt_file:
            opt_file.write(message)
            opt_file.write('\n')

    def parse(self):
        opt = self.gather_options()
        opt.isTrain = False  # Режим тестирования

        # Обработка суффикса
        if opt.suffix:
            suffix = ('_' + opt.suffix.format(**vars(opt))) if opt.suffix != '' else ''
            opt.name = opt.name + suffix

        # Установка GPU
        str_ids = opt.gpu_ids.split(',')
        opt.gpu_ids = []
        for str_id in str_ids:
            id = int(str_id)
            if id >= 0:
                opt.gpu_ids.append(id)
        if len(opt.gpu_ids) > 0:
            torch.cuda.set_device(opt.gpu_ids[0])

        # Параметры цветности
        opt.A = 2 * opt.ab_max / opt.ab_quant + 1
        opt.B = opt.A

        self.opt = opt
        return self.opt

