import core.config.configuration as cnfg
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import os
from torchvision import datasets, transforms
from torchvision.transforms import v2
from PIL import Image
from torch.utils.data import Dataset, ConcatDataset
from torchvision.transforms import functional as TF

from sklearn.model_selection import train_test_split
import numpy as np


train_transform = transforms.Compose([
    transforms.Resize(size=(cnfg.image_width, cnfg.image_width)),
    v2.RandomHorizontalFlip(p=0.5),
    transforms.ToTensor(),
    transforms.Normalize(mean=cnfg.mean, std=cnfg.std),
])

test_transform = transforms.Compose([transforms.Resize(size=(cnfg.image_width, cnfg.image_width)),
    transforms.ToTensor(),
    transforms.Normalize(mean=cnfg.mean, std=cnfg.std),
])



class BlackImagesDataset(Dataset):
    def __init__(self, num_images, img_size=cnfg.image_size, transform=None):
        self.num_images = num_images
        self.img_size = img_size
        self.transform = transform
        self.black_image = Image.fromarray(np.zeros((img_size[1], img_size[2], img_size[0]), dtype=np.uint8))

    def __len__(self):
        return self.num_images

    def __getitem__(self, idx):
        image = self.black_image
        if self.transform:
            image = self.transform(image)
        
        #label = -5 # Dummy label for black images
        return image#, torch.tensor(label, dtype=torch.int64)

class WhiteImagesDataset(Dataset):
    def __init__(self, num_images, img_size=cnfg.image_size, transform=None):
        self.num_images = num_images
        self.img_size = img_size
        self.transform = transform
        self.white_image = Image.fromarray(np.full((img_size[1], img_size[2], img_size[0]), 255, dtype=np.uint8))

    def __len__(self):
        return self.num_images

    def __getitem__(self, idx):
        image = self.white_image
        if self.transform:
            image = self.transform(image)
        #label = -4 
        return image#, torch.tensor(label, dtype=torch.int64)


class ImageOnlyDatasetWrapper(Dataset):
    """
    A dataset wrapper that returns only images from a given dataset,
    ignoring the labels or any other information.
    """
    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        image, _ = self.dataset[idx]  # Fetch the dataset item (ignoring the label)
        return image

class CustomImageDataset(Dataset):
    def __init__(self, data_dir, file_names, transform=None, num_black_images=cnfg.num_black_images,
                  num_white_images=cnfg.num_white_images, image_size=cnfg.image_size):
        self.data_dir = data_dir
        self.transform = transform
        self.file_names = file_names
        self.image_size = image_size

        # Add placeholders for black & white images
        self.black_image_placeholder = "<black_image>"
        self.file_names.extend([self.black_image_placeholder] * num_black_images)
        self.white_image_placeholder = "<white_image>"
        self.file_names.extend([self.white_image_placeholder] * num_white_images)

    def __len__(self):
        return len(self.file_names)

    def __getitem__(self, idx):
        if self.file_names[idx] == self.black_image_placeholder:
            # Create a black image
            black_image = torch.zeros(self.image_size)
            black_image = TF.to_pil_image(black_image)
            return transforms.ToTensor()(black_image)
        
        if self.file_names[idx] == self.white_image_placeholder:
            # Create a white image
            white_image = torch.ones(self.image_size)
            white_image = TF.to_pil_image(white_image)
            return transforms.ToTensor()(white_image) 

        
        img_name = os.path.join(self.data_dir, self.file_names[idx])
        image = Image.open(img_name).convert('RGB')

        if self.transform:
            image = self.transform(image)

        return image

def prepare_oxford_flowers_vae():
    black_dataset = BlackImagesDataset(cnfg.num_black_images, transform=train_transform)
    white_dataset = WhiteImagesDataset(cnfg.num_white_images, transform=train_transform)

    train_1 = datasets.Flowers102(root=cnfg.dataset_root, transform=train_transform, split='train', download = True)
    test = datasets.Flowers102(root=cnfg.dataset_root,transform=test_transform, split='val', download = True)
    train_2 = datasets.Flowers102(root=cnfg.dataset_root,transform=train_transform, split='test', download = True)

    train_1 = ImageOnlyDatasetWrapper(train_1)
    test = ImageOnlyDatasetWrapper(test)
    train_2 = ImageOnlyDatasetWrapper(train_2)

    train_dataset = ConcatDataset([train_1, train_2, black_dataset, white_dataset])
    test_dataset = ConcatDataset([test, black_dataset, white_dataset])

    train_loader = DataLoader(train_dataset, batch_size=cnfg.batch_size, shuffle=True, drop_last=True)
    test_loader = DataLoader(test_dataset, batch_size=cnfg.batch_size, shuffle=True, drop_last=True)

    test_dataiter = iter(test_loader)
    test_images = next(test_dataiter)

    return train_loader, test_images

def prepare_oxford_pets_vae():
    all_images = [img for img in os.listdir(cnfg.dataset_root) if img.endswith('.jpg')]  # Adjust for your file type
    train_images_, test_images_ = train_test_split(all_images, test_size=cnfg.test_size_vae, random_state=7)

    train_dataset = CustomImageDataset(data_dir=cnfg.dataset_root, file_names=train_images_, transform=train_transform, num_black_images=cnfg.num_black_images)
    test_dataset = CustomImageDataset(data_dir=cnfg.dataset_root, file_names=test_images_, transform=test_transform, num_black_images=cnfg.num_white_images)


    train_loader = DataLoader(train_dataset, batch_size=cnfg.batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=cnfg.batch_size, shuffle=True)

    test_dataiter = iter(test_loader)
    test_images = next(test_dataiter)

    return train_loader, test_images


def prepare_data_vae():
    if cnfg.dataset_name == "oxford_pets":
        train_loader, test_images = prepare_oxford_pets_vae()
        return train_loader, test_images
    else:
        train_loader, test_images = prepare_oxford_flowers_vae()
        return train_loader, test_images

