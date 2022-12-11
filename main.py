import sys
from datetime import datetime

import torch

from models.CNN import CNN, train_CNN
from models.DBN import DBN

if __name__ == "__main__":
    print(f"{'=' * 5}Initialiazing project{'=' * 5}\n")

    print(f"{'-' * 3}System set up...{'-' * 3}")
    timestamp = datetime.now()
    device = [torch.device(f"cpu:{i}") for i in range(8)]
    print(f"\tNumber of devices: {len(device)}")
    timestamp = datetime.now() - timestamp
    print(f"\tDone! Time elapsed: {timestamp}\n")

    print(f"{'-' * 3}Importing dataset...{'-' * 3}")
    print(f"[Preprocessing is performed contextually...]")
    timestamp = datetime.now()
    from import_pp.imp import train_ds, validation_ds, show_processed_imgs, img_height, img_width

    print(
        f"\tTrain Dataset size: {len(train_ds)}\n\tTest Dataset size: {len(validation_ds)}\n\tElements shape: {train_ds[0][0].shape}")

    print("Visualizing some random images:")
    show_processed_imgs(train_ds)
    timestamp = datetime.now() - timestamp
    print(f"Done! Time elapsed: {timestamp}\n")

    print(f"{'-' * 3}Model Training & Evaluation...{'-' * 3}")
    print("Training & Evaluation model 1 [Deep Belief Network]")

    with open('model1_log.txt', 'wt') as log_file:  # Print the output to file
        old_stdout = sys.stdout
        sys.stdout = log_file

        DBN_param = {
            'learning_rate': 0.05,
            'initial_momentum': 0.5,
            'final_momentum': 0.95,
            'weight_decay': 0.00001,
            'num_epochs': 50,
            'batch_size': 32
        }
        print(f"DBN training parameters:{DBN_param}")

        model_1 = DBN(visible_units=img_height * img_width,
                      hidden_units=[200, 500, 600, 800],
                      k=1,
                      learning_rate=DBN_param['learning_rate'],
                      learning_rate_decay=False,
                      initial_momentum=DBN_param['initial_momentum'],
                      final_momentum=DBN_param['final_momentum'],
                      weight_decay=DBN_param['weight_decay'],
                      xavier_init=False,
                      increase_to_cd_k=False,
                      use_gpu=torch.cuda.is_available())

        timestamp_local = datetime.now()

        model_1.train_static(
            train_ds.data,
            train_ds.targets,
            DBN_param['num_epochs'],
            DBN_param['batch_size']
        )

        print(f"[Training of the DBN is conlcuded, time elapsed: {datetime.now() - timestamp_local}]")

        sys.stdout = old_stdout

    print("Training & Evaluation model 2 [Convolutional Neural Network]")

    with open('model2_log.txt', 'wt') as log_file:  # Print the output to file
        old_stdout = sys.stdout
        sys.stdout = log_file

        model_2 = CNN(input_shape=(img_height, img_width))

        CNN_param = {
            'learning_rate': 0.05,
            'num_epochs': 50,
            'batch_size': 32
        }
        print(f"DBN training parameters:{CNN_param}")

        timestamp_local = datetime.now()

        train_CNN(model=model_2, num_epoch=CNN_param['num_epochs'], learning_rate=CNN_param['learning_rate'],
                  batch_size=CNN_param['batch_size'], train_ds=train_ds, test_ds=validation_ds)

        print(f"[Training of the CNN is conlcuded, time elapsed: {datetime.now() - timestamp_local}]")

        sys.stdout = old_stdout

    print(f"Done! Time elapsed: {timestamp}\n")
