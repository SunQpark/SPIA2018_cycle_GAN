import sys, os
import torch
from torch.utils.data import ConcatDataset
import numpy as np
from torchvision.datasets import ImageFolder
sys.path.append('./')
from datasets import CubDataset, CocoWrapper
from base import BaseDataLoader
from torchvision import transforms


def normalize_sizes(image):
    resize = transforms.Resize(192)
    if min(image.size) > 400:
        return resize(image)
    else:
        return image


class SketchDataLoader(BaseDataLoader):
    def __init__(self, data_dir, batch_size, validation_split=0.0, validation_fold=0, shuffle=True, num_workers=4):
        self.batch_size = batch_size
        trsfm = transforms.Compose([
            # transforms.ColorJitter(brightness=0.2),
            transforms.Grayscale(),
            # normalize_sizes,
            # transforms.Pad(50, padding_mode='edge'),
            # transforms.CenterCrop(256),
            # transforms.Resize(128),
            transforms.ToTensor(),
        ])
        self.dataset = ImageFolder(os.path.join(data_dir, 'aligned/sketch'), transform=trsfm)
        super(SketchDataLoader, self).__init__(self.dataset, self.batch_size, shuffle, validation_split, validation_fold, num_workers,)



class LfwDataLoader(BaseDataLoader):
    def __init__(self, data_dir, batch_size, validation_split=0.0, validation_fold=0, shuffle=True, num_workers=4):
        self.batch_size = batch_size
        trsfm = transforms.Compose([
            transforms.CenterCrop(256),
            # transforms.Resize(128),
            transforms.ToTensor(),
        ])
        
        self.dataset = ImageFolder('../data/lfw', transform=trsfm)
        super(LfwDataLoader, self).__init__(self.dataset, self.batch_size, shuffle, validation_split, validation_fold, num_workers)


class CelebADataLoader(BaseDataLoader):
    def __init__(self, data_dir, batch_size, validation_split=0.0, validation_fold=0, shuffle=True, num_workers=4):
        self.batch_size = batch_size
        trsfm = transforms.Compose([
            # transforms.CenterCrop(256),
            # transforms.Resize(128),
            transforms.ToTensor(),
        ])

        self.dataset = ImageFolder(os.path.join(data_dir, 'aligned/picture'), transform=trsfm)
        super(CelebADataLoader, self).__init__(self.dataset, self.batch_size, shuffle, validation_split, validation_fold, num_workers)


class CocoDataLoader(BaseDataLoader):
    def __init__(self, data_dir, batch_size, validation_split=0.0, validation_fold=0, shuffle=False, num_workers=4):
        self.batch_size = batch_size
            
        self.img_dir = os.path.join(data_dir, "images/train2014")
        self.ann_dir = os.path.join(data_dir, "annotations/captions_train2014.json")

        trsfm = transforms.Compose([
            transforms.CenterCrop(256),
            # transforms.Resize(64),
            transforms.ToTensor(),
        ])
        
        self.dataset = CocoWrapper(data_dir, transform=trsfm)
        super(CocoDataLoader, self).__init__(self.dataset, self.batch_size, shuffle, validation_split, validation_fold, num_workers, collate_fn=self._collate)

    def _collate(self, list_inputs):
        data = torch.cat([d.unsqueeze(0) for d, t in list_inputs])
        target = torch.zeros((data.shape[0], ), dtype=torch.long)
        #TODO: implement target packing
        return data, target


class CubDataLoader(BaseDataLoader):
    def __init__(self, data_dir, batch_size, validation_split=0.0, validation_fold=0, shuffle=False, num_workers=4):
        
        self.batch_size = batch_size
            
        trsfm = transforms.Compose([
            transforms.CenterCrop(256),
            # transforms.Resize(64),
            transforms.ToTensor(),
        ])
        
        self.dataset = CubDataset(data_dir, transform=trsfm)
        super(CubDataLoader, self).__init__(self.dataset, self.batch_size, shuffle, validation_split, validation_fold, num_workers, collate_fn=self._collate)

    def _collate(self, list_inputs):
        data = torch.cat([d.unsqueeze(0) for d, t in list_inputs])
        target = torch.zeros((data.shape[0], ), dtype=torch.long)
        #TODO: implement target packing
        return data, target


def gen_wrapper(loader):
    for data, target in loader:
        yield data

def concat_loader(batch_size=8):
    skc_loader = SketchDataLoader('../data', batch_size) 
    # lfw_loader = LfwDataLoader('../data/lfw', batch_size)
    celeba_loader = CelebADataLoader('../data', batch_size)
    s_gen = gen_wrapper(skc_loader)
    l_gen = gen_wrapper(celeba_loader)

    skc_end = False
    lfw_end = False
    skc_label = 0
    lfw_label = 1
    # to iterate both data_loaders to the end, replace 'and' with 'or'
    while not skc_end or not lfw_end:
        flag = np.random.rand() > 0.5
        if flag:
            try:
                data = next(s_gen)
                yield data, skc_label
            except StopIteration:
                skc_end = True
        else:
            try:
                data = next(l_gen)
                yield data, lfw_label
            except StopIteration:
                lfw_end = True



if __name__ == '__main__':
    # cub_loader = CubDataLoader('../data/birds', 4)
    # coco_loader = CocoDataLoader('../cocoapi', 4)

    # for i, (data_coco, data_cub) in enumerate(zip(coco_loader, cub_loader)):
    #     print(data_coco[0].shape)
    #     print(data_cub[0].shape)
    #     if i == 5: break

    # skc_loader = SketchDataLoader('../data/sketch_images/human', 4)
    # lfw_loader = LfwDataLoader('../data/lfw', 4)
    # for i, (data, target) in enumerate(loader):
    #     print(data.shape)
    #     print(target)
    #     # print(data_cub[0].shape)
    #     if i == 5: break
    # skc_end = False
    # lfw_end = False

    loader = CelebADataLoader('../data', 32)
    # print(type(skc_loader))
    # print(len(skc_loader))
    # data, target = skc_loader
    # print(data.shape)
    # print(target)
    for i, (data, target) in enumerate(loader):
        
        print(data.shape)
        # print(i, '\t', target)
        if i == 100: 
            break

