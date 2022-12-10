from datetime import datetime

import torch


def data_preprocessing():
    pass


if __name__ == "__main__":
    print(f"{'=' * 5}Initialiazing project{'=' * 5}\n")

    print(f"{'-' * 3}System set up...{'-' * 3}")
    timestamp = datetime.now()
    device = [torch.device(f"cpu:{i}") for i in range(8)]
    print(f"\tNumber of devices: {len(device)}")
    timestamp -= datetime.now()
    print(f"\tDone! Time elapsed: {timestamp}\n")

    print(f"{'-' * 3}Importing dataset...{'-' * 3}")
    print(f"[Preprocessing is performed contextually...]")
    timestamp = datetime.now()
    from import_pp.imp import train_ds, validation_ds, show_processed_imgs
    train_dataset, test_dataset = train_ds, validation_ds

    print("Visualizing some random images:")
    show_processed_imgs(train_dataset)
    timestamp -= datetime.now()
    print(f"Done! Time elapsed: {timestamp}\n")

    # print(f"{'-' * 3}Preprocessing RPS images...{'-' * 3}")
    # timestamp = datetime.now()
    # data_preprocessing()
    # timestamp -= datetime.now()
    # print(f"Done! Time elapsed: {timestamp}")
