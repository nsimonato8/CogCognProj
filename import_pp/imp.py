"""
Importing & Preprocessing module
"""
import random

import torch.utils.data
import torchvision as tv
from matplotlib import pyplot as plt
from torchvision.transforms import functional
from torchvision.transforms import transforms

seed_ = 123
split = "letters"


def get_mean_std(dataset):
    mean, std = 0., 0.
    loader = torch.utils.data.DataLoader(dataset=dataset, batch_size=50, shuffle=True)
    image_count_tot = 0

    for images, _ in loader:
        image_count = images.size(0)
        images = images.view(image_count, images.size(1), -1)
        mean += images.mean(2).sum(0)
        std += images.std(2).sum(0)
        image_count_tot += image_count

    return mean / image_count_tot, std / image_count_tot


def show_processed_imgs(dataset) -> None:
    idx = random.randint(1, 4)
    img, label = dataset.data[idx], dataset.targets[idx]
    print(f"Label: {label}")
    plt.imshow(img, cmap="gray")
    pass


train_transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=3),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ToTensor()
])

test_transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=3),
    transforms.ToTensor()
])

train_ds = tv.datasets.EMNIST('data/',
                              split=split,
                              train=True,
                              download=True,
                              transform=train_transform)

validation_ds = tv.datasets.EMNIST("data/",
                                   split=split,
                                   train=False,
                                   download=True,
                                   transform=test_transform)

img_width, img_height = functional.get_image_size(train_ds.__getitem__(0)[0])

# train_ds.data = (train_ds.data.type(torch.FloatTensor) / 255)
# validation_ds.data = (validation_ds.data.type(torch.FloatTensor) / 255)
