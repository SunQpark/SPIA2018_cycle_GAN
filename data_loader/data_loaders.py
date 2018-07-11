import sys, os
import torch
import numpy as np
sys.path.append('../')
from datasets import CubDataset, CocoWrapper
from base import BaseDataLoader
from torchvision import transforms


class CocoDataLoader(BaseDataLoader):
    def __init__(self, data_dir, batch_size, validation_split=0.0, validation_fold=0, shuffle=False, num_workers=4):
        self.batch_size = batch_size
            
        self.img_dir = os.path.join(data_dir, "images/train2014")
        self.ann_dir = os.path.join(data_dir, "annotations/captions_train2014.json")

        trsfm = transforms.Compose([
            transforms.CenterCrop(256),
            # transforms.Resize(64),
            transforms.ToTensor(),
            # Normalization that every pytorch pretrained models expect 
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                  std=[0.229, 0.224, 0.225]),
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
            # Normalization that every pytorch pretrained models expect
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                  std=[0.229, 0.224, 0.225])
        ])
        
        self.dataset = CubDataset(data_dir, transform=trsfm)
        super(CubDataLoader, self).__init__(self.dataset, self.batch_size, shuffle, validation_split, validation_fold, num_workers, collate_fn=self._collate)

    def _collate(self, list_inputs):
        data = torch.cat([d.unsqueeze(0) for d, t in list_inputs])
        target = torch.zeros((data.shape[0], ), dtype=torch.long)
        #TODO: implement target packing
        return data, target


if __name__ == '__main__':
    cub_loader = CubDataLoader('../data/birds', 4)
    coco_loader = CocoDataLoader('../cocoapi', 4)

    for i, (data_coco, data_cub) in enumerate(zip(coco_loader, cub_loader)):
        print(data_coco[0].shape)
        print(data_cub[0].shape)
        if i == 5: break
