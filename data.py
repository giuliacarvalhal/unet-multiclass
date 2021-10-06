import os
from PIL import Image 
import torch
from torch.utils.data import Dataset
from torchvision import transforms, datasets
import numpy as np
import os
import albumentations as A
from PIL import ImageOps

class BreastPhantom(Dataset):
    def __init__(self, mask_path, img_path, transform=None):
        self.transform = transform
        self.img_path=img_path
        self.mask_path=mask_path

        self.mapping = {0: 0, # 0 = no risk / background
                        1: 1, # 1 = low risk
                        2: 2, # 2 = medium risk
                        3: 3} # 3 = high risk 
        
        
        # Preparing a list of all labelTrainIds rgb and 
        # ground truth images. 
        self.yLabel_list=os.listdir(mask_path)
        self.XImg_list=os.listdir(img_path)

    def mask_to_class(self, y):
        for k in self.mapping:
            y[y==k] = self.mapping[k]
        return y    
                
    def __len__(self):
        length = len(self.XImg_list)
        return length
        

    def __getitem__(self, index):
        image = Image.open(self.img_path+'/'+self.XImg_list[index])
        y = Image.open(self.mask_path+'/'+self.yLabel_list[index])
        y = y.crop((0,1,98,256))
        image = image.astype('uint8')
        image = image / np.max(image) if max_val is None else image / max_val
        if int8:
          image = (image * 255).astype(np.uint8)        
        image = Image.open(image).convert("RGB")
        y = Image.open(y).convert("RGB")
       
        
        if self.transform is not None:
            image = self.transform(image)
            y = self.transform(y)

        image = transforms.ToTensor()(image)
        y = np.array(y)
        y = torch.from_numpy(y)
        y = self.mask_to_class(y)
        
        y = y.type(torch.LongTensor)
        
        
        
        return image, y
            
