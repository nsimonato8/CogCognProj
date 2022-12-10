from datetime import datetime
from random import randint

import torch
from matplotlib import pyplot as plt


def data_preprocessing():
    pass


if __name__ == "__main__":
    print(f"{'=' * 5}Initialiazing project{'=' * 5}\n")

    print(f"{'-' * 3}System set up...{'-' * 3}")
    timestamp = datetime.now()
    device = [torch.device(f"cpu:{i}") for i in range(8)]
    timestamp -= datetime.now()

    print(f"{'-' * 3}Importing dataset...{'-' * 3}")
    print(f"{'-' * 3}[Preprocessing is performed contextually...]{'-' * 3}")
    timestamp = datetime.now()
    from import_pp.imp import train_generator
    from import_pp.imp import validation_generator
    train_dataset, test_dataset = train_generator, validation_generator

    print("Visualizing some random images:")
    for i in range(5):
        idx = int(randint(0, train_dataset.__len__() - 1))
        img = train_dataset.__getitem__(idx)
        print(f"The picture shown is {train_dataset.targets[idx]}")
        plt.imshow(img, cmap='gray')
        plt.show()
        plt.savefig(f'{[i]}data.png')

    timestamp -= datetime.now()
    print(f"Done! Time elapsed: {timestamp}")

    # print(f"{'-' * 3}Preprocessing RPS images...{'-' * 3}")
    # timestamp = datetime.now()
    # data_preprocessing()
    # timestamp -= datetime.now()
    # print(f"Done! Time elapsed: {timestamp}")
