import os
import re

import cv2
import torch
import numpy as np
import pandas as pd
import torchvision as tv
import matplotlib.pyplot as plt

from PIL import Image
from tqdm import tqdm
from torchvision import transforms
from torch.utils.data import Dataset
from torch.utils.data import DataLoader, RandomSampler, BatchSampler

from model import UNet


# class Transform(object):
#     def __init__(self):
#         self.tsfm = al.Compose([
#             al.HorizontalFlip(p=0.5),
#             al.Flip(p=0.5),
#             # al.RandomBrightnessContrast(0.2, 0.2),
#         ], bbox_params=al.BboxParams(format='pascal_voc', min_visibility=0.75, label_fields=['labels']))
#
#     def __call__(self, sample):
#         image, annots, source = sample['img'], sample['annot'], sample['source']
#         augmented = self.tsfm(image=image, bboxes=annots[:, :4], labels=annots[:, 4])
#         img, boxes, labels = augmented['image'], augmented['bboxes'], augmented['labels']
#
#         return {'img': img, 'annot': np.array([[*b,  l] for b, l in zip(boxes, labels)]), 'source': source}


class Normalizer:
    def __call__(self, sample):
        out = {}
        image = sample['img']
        image = 2 * image.astype(np.float32) - 1
        out['img'] = image

        if sample.get('mask') is not None:
            out['mask'] = sample['mask']

        if sample.get('annot') is not None:
            out['annot'] = sample['annot']

        return out


class ToTensor:
    def __call__(self, sample):
        out = {}
        image = sample['img']
        image = torch.from_numpy(image.copy())
        out['img'] = image

        if sample.get('mask') is not None:
            out['mask'] = torch.from_numpy(sample['mask'].copy())

        if sample.get('annot') is not None:
            out['annot'] = torch.from_numpy(sample['annot'].copy())

        return out


class CommonDataset(Dataset):
    def __init__(self, filename, mode, transform):
        self.transform = transform
        self.mode = mode
        if mode == 'seg':
            self.data = self.pars_data_seg(filename)
        elif mode == 'od':
            self.data = self.pars_data_od(filename)

    @staticmethod
    def pars_data_seg(filename):
        with open(filename, 'r') as file:
            data = file.read()

        data = data.split('\n')
        result = []
        for line in data[:-1]:
            img, mask = line.split(',')
            result.append({'img': img, 'mask': mask})
        return result

    @staticmethod
    def pars_data_od(filename):
        dir_img = r'/media/storage/datasets/SpaceNet2/AOI_2_Vegas_Train/RGB-PanSharpen'
        data = []
        data_file = pd.read_csv(filename)
        for image_id, image_df in tqdm(data_file.groupby(by='ImageId')):
            image_bbox = []
            for idx, item in image_df.iterrows():
                polygon = []
                for split_item in item['PolygonWKT_Pix'].split(','):
                    point_xyz = re.findall('[0-9]*[.]?[0-9]+', split_item)
                    if len(point_xyz) == 3:
                        point_xy = list(map(lambda x: round(float(x)), point_xyz[:2]))
                        polygon.append(point_xy)
                if len(polygon):
                    polygon = np.array(polygon)

                    min_x = polygon[:, 0].min()
                    min_y = polygon[:, 1].min()
                    max_x = polygon[:, 0].max()
                    max_y = polygon[:, 1].max()

                    image_bbox.append([min_x, min_y, max_x, max_y])
            data.append({'img': os.path.join(dir_img, f'RGB-PanSharpen_{image_id}.tif'), 'annot': image_bbox})
        return data

    def load_annot(self):
        return 0

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        sample = {}
        img = cv2.imread(item['img']) / 255.0
        sample['img'] = img
        #sf

        if self.mode == 'seg':
            mask = np.array(Image.open(item['mask']))[..., np.newaxis]
            sample['mask'] = mask
        elif self.mode == 'od':
            sample['annot'] = item['annot']

        if self.transform:
            sample = self.transform(sample)

        return sample


# TODO доделать формат annot
def collater(data):
    imgs = []
    masks = []
    for s in data:
        imgs.append(s['img'])
        masks.append(s['mask'])

    image_batch = torch.stack(imgs)
    mask_batch = torch.stack(masks)

    # (B, H, W, C)
    # (B, C, H, W) -> Pytoch

    image_batch = image_batch.permute(0, 3, 1, 2)
    mask_batch = mask_batch.permute(0, 3, 1, 2)

    return {'img': image_batch, 'mask': mask_batch}


def creat_file_annot():
    filenames_input = os.listdir(
        '/home/teplykhna/storage/datasets/Massachusetts_Roads/road_segmentation_ideal/training/input')
    filenames_output = os.listdir(
        '/home/teplykhna/storage/datasets/Massachusetts_Roads/road_segmentation_ideal/training/output')
    filenames_input_fil = list(filter(lambda x: x in filenames_output, filenames_input))
    root_dir = r'/home/teplykhna/storage/datasets/Massachusetts_Roads/road_segmentation_ideal/training'
    with open('data.txt', 'w') as file:
        for filename_i, filename_o in zip(sorted(filenames_input_fil), sorted(filenames_output)):
            file.write(
                f"{os.path.join(root_dir, 'input', filename_i)},{os.path.join(root_dir, 'output', filename_o)}\n")


seg = False
if seg:
    creat_file_annot()
    dataset_train = CommonDataset(filename='data.txt',
                                  mode='seg',
                                  transform=transforms.Compose([Normalizer(), ToTensor()]))
else:
    dataset_train = CommonDataset(filename='/media/storage/datasets/SpaceNet2/AOI_2_Vegas_Train/summaryData/'
                                           'AOI_2_Vegas_Train_Building_Solutions.csv',
                                  mode='od',
                                  transform=[])
batch_size = 2
sampler = BatchSampler(RandomSampler(dataset_train), batch_size=batch_size, drop_last=False)
dataloader_loader = DataLoader(dataset_train, num_workers=0, collate_fn=collater, batch_sampler=sampler)

model = UNet(3, )

for data in dataloader_loader:
    image = data['img']
    annot = data['annot']

