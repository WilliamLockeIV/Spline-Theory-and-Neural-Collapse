import torch
from torch.utils.data import Dataset
import numpy as np

class CharacterDataset(Dataset):
    def __init__(self, images, labels, label_dict):
        # Add an extra dimension for "channel" at the start of each image,
        # so each batch will be of shape B,C,H,W.
        self.images = torch.tensor(images[:,np.newaxis], dtype=torch.float)
        self.labels = labels
        self.class_labels = [label_dict[label] for label in labels]

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return (self.images[idx], self.class_labels[idx])