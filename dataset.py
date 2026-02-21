import torch
from torchvision import datasets, transforms
import torchvision
import numpy as np


class IICDataset(torch.utils.data.Dataset):
    """
    STL-10 Dataset wrapper for IIC that returns original and transformed versions of the image.
    It automatically downloads the dataset using torchvision.
    """

    def __init__(self, root, split, download=True):
        super().__init__()
        self.dataset = datasets.STL10(root=root, split=split, download=download)

        # Base Transformation (common for original and transformed evaluation logic)
        self.base_transform = transforms.Compose(
            [
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )

        # Random Augmentation (used to create the transformed view for IIC loss)
        self.random_transform = transforms.Compose(
            [
                transforms.RandomAffine(10, (0.2, 0.4), (0.6, 0.75)),
                transforms.ColorJitter(
                    brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2
                ),
                transforms.RandomHorizontalFlip(p=0.5),
            ]
        )

        # We need to convert PIL image to tensor first before applying basic normalization and random_transform
        self.to_tensor = transforms.ToTensor()

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        img_pil, label = self.dataset[index]

        # Convert PIL to Tensor [0, 1]
        img_tensor = self.to_tensor(img_pil)

        # Apply base normalization
        x = self.base_transform(img_tensor)

        # Apply random transforms to generate x_prime
        x_prime = self.random_transform(x.clone())

        return x, x_prime, label


def create_dataloaders(root_dir, batch_size=256):
    """
    Creates dataloaders for the STL-10 dataset using torchvision defaults.
    """
    # Create dataset objects for train(unlabeled) and test
    train_dataset = IICDataset(root=root_dir, split="unlabeled", download=True)
    test_dataset = IICDataset(root=root_dir, split="test", download=True)

    train_loader = torch.utils.data.DataLoader(
        dataset=train_dataset, batch_size=batch_size, shuffle=True, drop_last=True
    )

    test_loader = torch.utils.data.DataLoader(
        dataset=test_dataset, batch_size=batch_size, shuffle=False, drop_last=False
    )

    return train_loader, test_loader
