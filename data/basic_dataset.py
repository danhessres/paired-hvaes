import os
import cv2
import torch
import random
import torchvision.transforms.functional as F

from torch.utils.data import Dataset


def ls(path):
    contents = os.listdir(path)
    return [os.path.join(path, a) for a in contents]

def cv2_read(path):
    raw = cv2.imread(path)
    raw = cv2.cvtColor(raw, cv2.COLOR_BGR2RGB)
    return torch.tensor(raw)

class BasicDataset(Dataset):
    def __init__(self, path, res, cha, alt_c = None, preload = False, transforms = None):
        super().__init__()
        self.dir_files = os.listdir(path)
        self.files = ls(path)
        self.is_loaded = False
        self.loaded_data = None
        self.transforms = transforms if transforms is not None else {}
        self.res = res
        self.cha = cha
        self.alt = cha if alt_c is None else alt_c
        if preload:
            self.load_all()

    def preprocess(self, data, cha):
        data = data.float() / 255
        data = data.moveaxis(-1, 0)
        data = data[:cha]
        return data

    def transform(self, a,b):
        if self.transforms.get('vflip', False):
            if random.random() > 0.5:
                a = F.vflip(a)
                b = F.vflip(b)
        if self.transforms.get('hflip', False):
            if random.random() > 0.5:
                a = F.hflip(a)
                b = F.hflip(b)
        if self.transforms.get('sf', False) and self.transforms.get('offs', False):
            if random.random() > 0.5:
                #Random scale
                sf_sze = self.transforms.get('sf', 0.0)
                new_hsf = 1 + sf_sze * (2 * random.random() - 1)
                new_vsf = 1 + sf_sze * (2 * random.random() - 1)
                new_wid = int(self.res * new_hsf)
                new_hei = int(self.res * new_vsf)
                #Random offset
                off_sze = self.transforms.get('offs', 0)
                hoff = random.randrange(-(off_sze // 2), off_sze // 2 + 1)
                voff = random.randrange(-(off_sze // 2), off_sze // 2 + 1)
                a = F.resized_crop(a, hoff, voff, new_hei, new_wid, (self.res, self.res))
                b = F.resized_crop(b, hoff, voff, new_hei, new_wid, (self.res, self.res))
        return (a, b)

    def load_ind(self, ind):
        raw = cv2_read(self.files[ind])
        a, b = raw.chunk(2, axis = 1) #Split horizontally
        a = self.preprocess(a, self.cha)
        b = self.preprocess(b, self.alt)
        return a,b

    def load_all(self):
        #Get reference to prep array
        a,b = self.load_ind(0)
        count = len(self.files)
        adata = torch.zeros([count] + list(a.shape))
        bdata = torch.zeros([count] + list(b.shape))
        for i in range(count):
            a,b = self.load_ind(i)
            adata[i] = a
            bdata[i] = b
        self.loaded_data = [adata, bdata]
        self.is_loaded = True
        return

    def __getitem__(self, ref):
        if self.is_loaded and (self.loaded_data is not None):
            a = self.loaded_data[0][ref]
            b = self.loaded_data[1][ref]
        else:
            a,b = self.load_ind(ref)
        a, b = self.transform(a,b)
        return [a, b]

    def __len__(self):
        return len(self.files)
