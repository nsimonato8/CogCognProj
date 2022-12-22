"""
Importing & Preprocessing module
"""

import numpy as np
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
    loader = torch.utils.data.DataLoader(dataset, batch_size=6,
                                         shuffle=True)
    images, labels = next(iter(loader))
    img = tv.utils.make_grid(images)
    img = img / 2 + 0.5  # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    pass


train_transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ToTensor()
])

test_transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
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
