#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import os
import torch
from collections import OrderedDict
from util.image_pool import ImagePool
from util import util
from .base_model import BaseModel
from . import networks
import numpy as np
from skimage import io
from skimage import img_as_ubyte

import matplotlib.pyplot as plt
import math
from matplotlib import colors


class CocoFullModel(BaseModel):
    def name(self):
        return 'CocoFullModel'

    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        return parser

    def initialize(self, opt):
        BaseModel.initialize(self, opt)
        self.model_names = ['G']  # Только модель G (coco_full)

        # Загрузка/определение сети
        num_in = opt.input_nc + opt.output_nc + 1
        
        self.netG = networks.define_G(num_in, opt.output_nc, opt.ngf,
                                      'siggraph', opt.norm, not opt.no_dropout, opt.init_type, self.gpu_ids,
                                      use_tanh=True, classification=opt.classification)
        if not self.isTrain or opt.load_model:
            self.load_networks(opt.which_epoch)

    def set_input(self, input):
        AtoB = self.opt.which_direction == 'AtoB'
        self.real_A = input['A' if AtoB else 'B'].to(self.device)
        self.real_B = input['B' if AtoB else 'A'].to(self.device)
        self.hint_B = input['hint_B'].to(self.device)
        
        self.mask_B = input['mask_B'].to(self.device)
        self.mask_B_nc = self.mask_B + self.opt.mask_cent

        self.real_B_enc = util.encode_ab_ind(self.real_B[:, :, ::4, ::4], self.opt)

    def forward(self):
        # Прямой проход через сеть G
        (_, self.fake_B_reg) = self.netG(self.real_A, self.hint_B, self.mask_B)

    def optimize_parameters(self):
        # Оптимизация параметров (если используется обучение)
        self.forward()
        self.compute_losses_G()
        self.optimizer_G.zero_grad()
        self.loss_G.backward()
        self.optimizer_G.step()

    def save_current_imgs(self, path):
        # Сохранение текущих изображений
        out_img = torch.clamp(util.lab2rgb(torch.cat((self.real_A.type(torch.cuda.FloatTensor), self.fake_B_reg.type(torch.cuda.FloatTensor)), dim=1), self.opt), 0.0, 1.0)
        out_img = np.transpose(out_img.cpu().data.numpy()[0], (1, 2, 0))
        io.imsave(path, img_as_ubyte(out_img))

    def setup_to_test(self, weight_path):
        # Загрузка весов для тестирования
        G_path = 'checkpoints/{0}/latest_net_G.pth'.format(weight_path)
        print('Loading model from %s' % G_path)
        G_state_dict = torch.load(G_path)
        self.netG.module.load_state_dict(G_state_dict, strict=False)
        self.netG.eval()

