import os
import torch
from torchvision.datasets import CocoDetection
import torchvision.transforms as transforms
from utils import collate_fn

def get_dataloader(batch_size=4, num_workers=4):
    # Define the COCO transform
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])

    # Load the training and test sets
    train_dataset = CocoDetection(root='data/PascalCocoDataset/train', annFile='data/PascalCocoDataset/train/train_coco.json', transform=transform)
    test_dataset = CocoDetection(root='data/PascalCocoDataset/val', annFile='data/PascalCocoDataset/val/val_coco.json', transform=transform)

    # Define the data loaders
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, collate_fn=collate_fn
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, collate_fn=collate_fn
    )

    return train_loader, test_loader
