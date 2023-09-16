from torch.utils.data import Dataset
import os
from PIL import Image
from torchvision import transforms
import torch

transformation = transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Resize((256,256))

    ])
aug_transformation1 = transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Resize((256,256)),
                    transforms.RandomRotation(40)

    ])

class Dataloader(Dataset):
    def __init__(self, path, mode,trans=False):
        self.path = path
        self.mode = mode
        self.image_path = os.path.join(self.path, self.mode)
        self.image_list = os.listdir(self.image_path)
        self.transform = trans

    
    def __len__(self):
        return len(self.image_list)
    
    def __getitem__(self, index):
        image = Image.open(os.path.join(self.image_path, self.image_list[index]))
        label = torch.tensor([0 if self.image_list[index].split("_")[0]=='cat' else 1])
        if self.transform:
            image = aug_transformation1(image)
        else:
            image = transformation(image)

        return image, label