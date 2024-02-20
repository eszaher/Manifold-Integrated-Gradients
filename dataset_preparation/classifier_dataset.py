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

import os
from PIL import Image
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split


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


class CustomImageDatasetWithLabels(Dataset):
    def __init__(self, data_dir, file_names, labels, label_to_index, transform=None):
        self.data_dir = data_dir
        self.file_names = file_names
        self.labels = labels
        self.transform = transform
        self.label_to_index = label_to_index

    def __len__(self):
        return len(self.file_names)

    def __getitem__(self, idx):
        img_path = os.path.join(self.data_dir, self.file_names[idx])
        image = Image.open(img_path).convert('RGB')
        label_name = self.labels[idx]
        label_index = self.label_to_index[label_name]

        if self.transform:
            image = self.transform(image)

        return image, label_index
    

def prepare_oxford_flowers_classifier():
    

    train_dataset = datasets.Flowers102(root=cnfg.dataset_root, transform=train_transform, split='train', download = True)
    test_dataset = datasets.Flowers102(root=cnfg.dataset_root,transform=train_transform, split='test', download = True)


    train_loader = DataLoader(train_dataset, batch_size=cnfg.batch_size, shuffle=True, drop_last=True)
    test_loader = DataLoader(test_dataset, batch_size=cnfg.batch_size, shuffle=True, drop_last=True)

    return train_loader, test_loader

def load_dataset(data_dir):
    file_names = [f for f in os.listdir(data_dir) if f.endswith('.jpg')]  # Adjust as needed
    labels = ['_'.join(f.split('_')[:-1]) for f in file_names]
    return file_names, labels

def prepare_oxford_pets_classifier():
    # Load dataset
    data_dir = cnfg.dataset_root
    file_names, labels = load_dataset(data_dir)

    unique_labels = set()  # A set to store all unique labels

    # Assuming 'all_labels' is a list of all labels in the dataset
    for label in labels:
        unique_labels.add(label)

    # Create a mapping from label to index
    label_to_index = {label: idx for idx, label in enumerate(unique_labels)}
    # Split dataset
    train_files, test_files, train_labels, test_labels = train_test_split(file_names, labels, test_size=cnfg.test_size_clf, random_state=7)

    train_dataset = CustomImageDatasetWithLabels(data_dir, file_names=train_files, labels=train_labels, label_to_index=label_to_index, transform=train_transform)
    test_dataset = CustomImageDatasetWithLabels(data_dir, file_names=test_files, labels=test_labels,label_to_index=label_to_index, transform=test_transform)

    train_loader = DataLoader(train_dataset, batch_size=cnfg.batch_size, shuffle=True, num_workers=cnfg.num_workers)
    test_loader = DataLoader(test_dataset, batch_size=cnfg.batch_size, shuffle=False, num_workers=cnfg.num_workers)
    return train_loader, test_loader


def prepare_data_classifier():
    if cnfg.dataset_name == "oxford_pets":
        train_loader, test_loader = prepare_oxford_pets_classifier()
        return train_loader, test_loader
    else:
        train_loader, test_loader = prepare_oxford_flowers_classifier()
        return train_loader, test_loader

