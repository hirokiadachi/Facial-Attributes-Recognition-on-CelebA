import os
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset

class CelebA_loader(Dataset):
    def __init__(self, 
                 datapath='/root/mnt/datasets/facial_datasets/celebA/original', 
                 attribute_index = None,
                 size=224,
                 transform=None,
                 train=True):
        self.size = size
        self.transform = transform
        self.attribute_index = np.arange(attribute_index)
        
        ## load landmark information
        landmark_path = os.path.join(datapath, 'Anno/list_landmarks_align_celeba.txt')
        with open(landmark_path, 'r') as f:
            self.landmark_lines = f.readlines()[2:]
        
        ## load attribute index    
        attribute_path = os.path.join(datapath, 'Anno/list_attr_celeba.txt')
        with open(attribute_path, 'r') as f:
            self.attribute_lines = f.readlines()[2:]
        self.attributes_all_samples = self.choose_attribute()
        
        self.image_path = os.path.join(datapath, 'img_celeba')
        self.image_items = sorted(os.listdir(self.image_path))
        self.datalength = len(os.listdir(self.image_path))
        threshold = int(self.datalength * 0.9)
        if train:
            self.length = threshold
            self.images_list = self.image_items[:self.length]
            self.attributes = self.attributes_all_samples[:self.length]
            print('Number of training samples: %d' % self.length)
        else:
            self.length = self.datalength - threshold
            self.images_list = self.image_items[threshold:]
            self.attributes = self.attributes_all_samples[threshold:]
            print('Number of test samples: %d' % self.length)
        
    def __len__(self):
        return self.length
    
    def choose_attribute(self):
        attrs = []
        for item in self.attribute_lines:
            item_list = item.split()[1:]
            attrs.append([int(item_list[i]) if int(item_list[i])==1 else 0 for i in self.attribute_index])
        return attrs
    
    def __getitem__(self, i):
        img = Image.open(os.path.join(self.image_path, self.images_list[i])).resize((self.size, self.size))
        attr = torch.tensor(self.attributes[i]).long()
        onehot = torch.eye(len(attr), 2)[attr].permute(1,0).view(2,1,len(attr))
        if self.transform:
            img = self.transform(img)
        
        return img, onehot