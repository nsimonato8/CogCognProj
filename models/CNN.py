from datetime import datetime

import torch
from torch import nn


# https://androidkt.com/convolutional-neural-network-using-sequential-model-in-pytorch/

class CNN:
    def __init__(self, input_shape, optimizer=None, loss_fn=None, learning_rate=0.1,
                 momentum=0.9, device=None):
        super().__init__()
        self.input_shape = input_shape
        self.model = nn.Sequential(
            nn.Conv2d(input_shape[0] * input_shape[1], 64, kernel_size=(3, 3), padding=1),
            nn.Conv2d(32, 64, kernel_size=(3, 3), padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(64, 128, kernel_size=(3, 3), padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            nn.Flatten(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, 26))

        if optimizer is None or loss_fn is None:
            self.optimizer = torch.optim.SGD(self.model.parameters(), lr=learning_rate, momentum=momentum)
            self.loss_fn = nn.CrossEntropyLoss()

        if device is not None:
            self.model.to(device)
        pass

    def train_CNN(self, num_epoch, train_ds, test_ds, batch_size=32):
        timestamp = datetime.now()

        train_ds_loader = torch.utils.data.DataLoader(train_ds, shuffle=True, batch_size=batch_size)
        test_ds_loader = torch.utils.data.DataLoader(test_ds, shuffle=True, batch_size=batch_size)

        train_losses = []
        valid_losses = []

        for epoch in range(1, num_epoch + 1):
            train_loss = 0.0
            valid_loss = 0.0

            self.model.train()
            for img, lbl in train_ds_loader:
                img = img.cuda()
                lbl = lbl.cuda()

                self.optimizer.zero_grad()
                predict = self.model(img)
                loss = self.loss_fn(predict, lbl)
                loss.backward()
                self.optimizer.step()
                train_loss += loss.item() * img.size(0)

            self.model.eval()
            for img, lbl in test_ds_loader:
                img = img.cuda()
                lbl = lbl.cuda()

                predict = self.model(img)
                loss = self.loss_fn(predict, lbl)

                valid_loss += loss.item() * img.size(0)

            train_loss = train_loss / len(train_ds_loader.sampler)
            valid_loss = valid_loss / len(test_ds_loader.sampler)

            train_losses.append(train_loss)
            valid_losses.append(valid_loss)

            print('Epoch:{} Train Loss:{:.4f} Test Losss:{:.4f}'.format(epoch, train_loss, valid_loss))
            print(f"[Training of the CNN is conlcuded, time elapsed: {datetime.now() - timestamp}]")
