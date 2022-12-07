import torch

print(torch.cuda.is_available())
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


class LinearModel(torch.nn.Module):  # This class is needed for linear readout
    def __init__(self, last_layer_size):
        super().__init__()
        self.linear = torch.nn.Linear(last_layer_size, 10)

    def forward(self, x):
        return self.linear(x)

    def train_(self, network, input, data, epochs=1500):
        """
        Training function for the linear model
        :param network:
        :param input:
        :param data:
        :param epochs:
        :return:
        """
        optimizer = torch.optim.SGD(network.parameters(), lr=0.05)
        loss_fn = torch.nn.CrossEntropyLoss()

        for epoch in range(epochs):
            optimizer.zero_grad()
            predictions = network(input).squeeze()
            targets = data.targets.reshape(predictions.shape[0])  # here are the labels
            loss = loss_fn(predictions, targets)
            loss.backward()
            optimizer.step()

            if epoch % 100 == 0:
                print("epoch : {:3d}/{}, loss = {:.4f}".format(epoch + 1, epochs, loss))



