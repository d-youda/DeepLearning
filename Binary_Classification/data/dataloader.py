from torch.utils.data import Dataset
import os


class Dataloader(Dataset):
    def __init__(self, path, mode):
        self.path = path
        self.mode = mode

        self.dog_image_path = os.path.join(path, f'{mode}', 'dogs')
        self.cat_image_path = os.path.join(path, f'{mode}', 'cats')
        self.image_list = os.listdir(self.dog_image_path) +  os.listdir(self.cat_image_path)

    
    def __len__(self):
        return len(self.image_list)
    
    def __getitem__(self, index):
        image = self.image_list[index]
        label = self.image_list[index].split("_")[0]

        return image, label