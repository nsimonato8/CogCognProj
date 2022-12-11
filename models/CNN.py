from datetime import datetime

import torch
from torch import nn


# https://androidkt.com/convolutional-neural-network-using-sequential-model-in-pytorch/

class CNN(nn.Module):
    def __init__(self, input_shape):
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv2d(input_shape[0] * input_shape[1], 64, kernel_size=(3, 3), padding=0),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(64, 128, kernel_size=(3, 3), padding=0),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            nn.Flatten(),
            nn.Linear(128 * 4 * 4, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, 3))

    # Prediction (x is input)
    # The forward function is automatically called when we create an instance of the class and call it.
    def forward(self, x):
        x = self.cnn1(x)
        x = self.conv1_bn(x)
        x = self.leaky_relu(x)
        x = self.maxpool1(x)

        x = self.cnn2(x)
        x = self.conv2_bn(x)
        x = self.leaky_relu(x)
        x = self.maxpool2(x)

        # Flattening cnn2's output and passing it into a fully connected layer
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        return x


def train_CNN(model, num_epoch, train_ds, test_ds, optimizer=None, loss_fn=None, learning_rate=0.001, momentum=0.9,
              batch_size=32):
    timestamp = datetime.now()
    if optimizer is None or loss_fn is None:
        optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum)
        loss_fn = nn.CrossEntropyLoss()

    train_ds_loader = torch.utils.data.DataLoader(train_ds, shuffle=True, batch_size=batch_size, num_workers=8)
    test_ds_loader = torch.utils.data.DataLoader(test_ds, shuffle=True, batch_size=batch_size, num_workers=8)

    train_losses = []
    valid_losses = []

    for epoch in range(1, num_epoch + 1):
        train_loss = 0.0
        valid_loss = 0.0

        model.train()
        for img, lbl in train_ds_loader:
            img = img.cuda()
            lbl = lbl.cuda()

            optimizer.zero_grad()
            predict = model(img)
            loss = loss_fn(predict, lbl)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * img.size(0)

        model.eval()
        for img, lbl in test_ds_loader:
            img = img.cuda()
            lbl = lbl.cuda()

            predict = model(img)
            loss = loss_fn(predict, lbl)

            valid_loss += loss.item() * img.size(0)

        train_loss = train_loss / len(train_ds_loader.sampler)
        valid_loss = valid_loss / len(test_ds_loader.sampler)

        train_losses.append(train_loss)
        valid_losses.append(valid_loss)

        print('Epoch:{} Train Loss:{:.4f} Test Losss:{:.4f}'.format(epoch, train_loss, valid_loss))
        print(f"[Training of the CNN is conlcuded, time elapsed: {datetime.now() - timestamp}]")
