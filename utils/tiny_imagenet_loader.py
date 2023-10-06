import torch
import numpy as np
#from outlier_synthesis import *

class customTinyImageNet(torch.utils.data.Dataset):
    def __init__(self, tiny_imagenet):
        self.tiny_imagenet=tiny_imagenet
        self.offset = 0  # offset index

    def __len__(self):
        return len(self.tiny_imagenet)

    def __getitem__(self, index):
        index = (index + self.offset) % len(self.tiny_imagenet)

        img,_ = self.tiny_imagenet[index]

        return img, 0  # 0 is the class

class SimpleDataset(torch.utils.data.Dataset):
    def __init__(self, images, labels, transform=None):
        self.labels = labels
        self.images = images
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        # Load data and get label
        X = self.images[index]
        if self.transform:
            X = self.transform(X)
        y = self.labels[index]

        return X, y