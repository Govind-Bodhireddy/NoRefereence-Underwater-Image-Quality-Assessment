import torch
from torch.utils.data import DataLoader,Dataset
from torchvision.transforms import transforms
from PIL import Image
import os
import numpy as np
from torch.utils.data import DataLoader
class dataload(Dataset):
    def __init__(self,data_path):
        self.data_path=data_path
    def __len__(self):
            return 1000
    def __getitem__(self, index):
        image=Image.open(self.data_path[index])
        image=transforms.Resize(230)(image)
        image=torch.tensor(np.array(image)).permute(2,1,0).float()
        return image
