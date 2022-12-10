import matplotlib
import numpy as np

matplotlib.use('Agg')
import os

import torch.utils.data
import torchvision

# Importing & Preprocessing
from matplotlib import pyplot as plt
from torchvision.transforms import transforms

TRAINING_DIR = "import_pp/tmp/rps/"
VALIDATION_DIR = "import_pp/tmp/rps-test-set/"
images_count = 444  # Dummy value
seed_ = 123
batch_size_tr, batch_size_vd = images_count * 0.8, images_count * 0.2
img_height, img_width = 150, 150

# train_datagen = ImageDataGenerator(rescale=1. / 255)  # Only rescaling is done, in order to not introduce noise in the data. That will be done succesively.
# test_datagen = ImageDataGenerator(rescale=1. / 255)
#
# train_generator = train_datagen.flow_from_directory(
#     TRAINING_DIR,
#     target_size=(img_height, img_width),
#     batch_size=batch_size_tr,
#     class_mode='categorical')
#
# validation_generator = test_datagen.flow_from_directory(
#     VALIDATION_DIR,
#     target_size=(img_height, img_width),
#     batch_size=batch_size_vd,
#     class_mode='categorical')


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

    return mean/image_count_tot, std/image_count_tot


train_transform = transforms.Compose([
    transforms.Resize((img_height, img_width)),
    transforms.Grayscale(),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ToTensor(),
    # transforms.Normalize(torch.Tensor(train_mean), torch.Tensor(train_std))
])

test_transform = transforms.Compose([
    transforms.Resize((img_height, img_width)),
    transforms.Grayscale(),
    transforms.ToTensor(),
    # transforms.Normalize(torch.Tensor(test_mean), torch.Tensor(test_std))
])

train_ds = torchvision.datasets.ImageFolder(root=TRAINING_DIR, transform=train_transform)
validation_ds = torchvision.datasets.ImageFolder(root=VALIDATION_DIR, transform=test_transform)


def show_processed_imgs(dataset) -> None:
    loader = torch.utils.data.DataLoader(dataset, batch_size=6, shuffle=True)
    batch = next(iter(loader))
    images, labels = batch

    grid = torchvision.utils.make_grid(images, n_row=3)
    plt.figure(figsize=(25, 25))
    plt.imshow(np.transpose(grid, (1, 2, 0)))
    plt.savefig(f'training_data_peek.png')
    print(f"Labels: {labels}\n")
    pass


def get_dbn_library():
    files = ["DBN.py", "RBM.py"]
    repository_url = "https://raw.githubusercontent.com/flavio2018/Deep-Belief-Network-pytorch/master/"
    for file in files:
        os.system("wget -O {file} {repository_url}{file}")
