import os.path
from data.base_dataset import BaseDataset, get_transform
from data.image_folder import make_dataset
from PIL import Image


class SingleDataset(BaseDataset):

    @staticmethod
    def modify_commandline_options(parser, is_train):
        # стандартные параметры
        return parser
    def initialize(self, opt):
        # Сохраненение параметров
        self.opt = opt
        self.root = opt.dataroot
        self.dir_A = os.path.join(opt.dataroot)
        # Собираем изображения из папки который создали
        self.A_paths = make_dataset(self.dir_A)
        self.A_paths = sorted(self.A_paths)
        # Получаем preprocessing pipeline
        self.transform = get_transform(opt)

    def __getitem__(self, index):
        # Загрузка изображений
        A_path = self.A_paths[index]
        A_img = Image.open(A_path).convert('RGB')
        # Применил transform
        A = self.transform(A_img)
        # Определение количество входных каналов
        if self.opt.which_direction == 'BtoA':
            input_nc = self.opt.output_nc
        else:
            input_nc = self.opt.input_nc

        # Если модель ожидает 1 канал  переводим в grayscale
        if input_nc == 1:
            tmp = A[0, ...] * 0.299 + A[1, ...] * 0.587 + A[2, ...] * 0.114
            A = tmp.unsqueeze(0)

        return {
            'A': A,
            'A_paths': A_path
        }

    def __len__(self):
        # Размер датасета
        return len(self.A_paths)

    def name(self):
        return 'SingleImageDataset'
