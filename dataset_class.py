import os
import pandas as pd
import torch
import cv2
import numpy as np
from torch.utils.data import Dataset as BaseDataset

class Dataset(BaseDataset):

    def __init__(
            self,
            root,
            images_dir,
            csv,
            image_net = False,
            aug_fn=None,
            preprocessing=None,
            column_list = ["Image_ID", "Class 1", "Class 2", "Class 3", "Class 4", "Class 5", "Class 6"]
    ):
        images_dir = os.path.join(root,images_dir)
        df = pd.read_csv(os.path.join(root,csv))

        self.ids = [
            (r[column_list[0]], r[column_list[1]], r[column_list[2]], r[column_list[3]], r[column_list[4]], r[column_list[5]], r[column_list[6]]) for i, r in df.iterrows()
        ]

        self.images = [os.path.join(images_dir, item[0]) for item in self.ids]
        self.labels = [item[1:] for item in self.ids]

        self.aug_fn = aug_fn
        self.preprocessing = preprocessing

        self.imagenet = image_net

    def __getitem__(self, i):
        if not self.imagenet:
            image = cv2.imread(self.images[i],0)
            image = np.expand_dims(image, axis=-1)
        else:
            image = cv2.imread(self.images[i])
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        label = self.labels[i]

        if self.aug_fn:
            sample = self.aug_fn(image.shape)(image=image)
            image = sample['image']

        if self.preprocessing:
            sample = self.preprocessing(image=image)
            image = sample['image']
            label = torch.tensor(label, dtype=torch.float32)

        return image, label
    
    def __len__(self):
        return len(self.images)