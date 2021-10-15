from PIL import Image
import numpy as np
import torch
from torchvision import transforms


def get_train_transforms(img_size=256):
    return transforms.Compose(
        [
            transforms.RandomApply(
                torch.nn.ModuleList(
                    [
                        transforms.RandomResizedCrop(img_size, scale=(0.7, 0.9)),
                    ]
                ),
                p=0.8,
            ),
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )


def get_valid_transforms(img_size=256):
    return transforms.Compose(
        [
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )


class PetFinderDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        samples,
        image_dir,
        img_size=256,
        mode="train",
    ):
        self.ids = list(samples["Id"])
        self.labels = list(samples["Pawpularity"])
        self.image_dir = image_dir
        self.img_size = img_size
        self.mode = mode
        if self.mode == "train":
            self.tfms = get_train_transforms(self.img_size)
        else:
            self.tfms = get_valid_transforms(self.img_size)

    def __getitem__(self, idx):
        img_id = self.ids[idx]
        label = torch.tensor(self.labels[idx] / 100, dtype=torch.float)
        img_path = self.image_dir + img_id + ".jpg"
        img = self.tfms(Image.open(img_path))
        img = torch.tensor(img.numpy(), dtype=torch.float)
        return {"images": img, "labels": label}

    def __len__(self):
        return len(self.labels)
