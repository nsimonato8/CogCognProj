"""
Importing & Preprocessing module
"""
import numpy as np
import torch.utils.data
import torchvision as tv
from matplotlib import pyplot as plt
from torchvision.transforms import transforms

images_count = 32  # Dummy value
seed_ = 123
batch_size_tr, batch_size_vd = images_count, images_count
img_height, img_width = 150, 150


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


train_transform = transforms.Compose([
    transforms.Resize((img_height, img_width)),
    transforms.Grayscale(),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ToTensor()
])

test_transform = transforms.Compose([
    transforms.Resize((img_height, img_width)),
    transforms.Grayscale(),
    transforms.ToTensor()
])

train_ds = tv.datasets.EMNIST('data/',
                              split="byclass",
                              train=True,
                              download=True,
                              transform=train_transform)

validation_ds = tv.datasets.EMNIST("data/",
                                   split="byclass",
                                   train=False,
                                   download=True,
                                   transform=test_transform)


def show_processed_imgs(dataset) -> None:
    loader = torch.utils.data.DataLoader(dataset, batch_size=6, shuffle=True)
    batch = next(iter(loader))
    images, labels = batch

    grid = tv.utils.make_grid(images, n_row=3)
    plt.figure(figsize=(25, 25))
    plt.imshow(np.transpose(grid, (1, 2, 0)))
    plt.savefig(f'training_data_peek.png')
    print(f"Labels: {labels}\n")
    pass
