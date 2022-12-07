import torch


class Feedforward(torch.nn.Module):
    def __init__(self, first_hidden_layer_size, second_hidden_layer_size, third_hidden_layer_size):
        super().__init__()
        self.first_hidden = torch.nn.Linear(784, first_hidden_layer_size)
        self.second_hidden = torch.nn.Linear(first_hidden_layer_size, second_hidden_layer_size)
        self.third_hidden = torch.nn.Linear(second_hidden_layer_size, third_hidden_layer_size)
        self.output = torch.nn.Linear(third_hidden_layer_size, 10)

    def forward(self, input):
        relu = torch.nn.ReLU()
        first_hidden_repr = relu(self.first_hidden(input))
        second_hidden_repr = relu(self.second_hidden(first_hidden_repr))
        third_hidden_repr = relu(self.third_hidden(second_hidden_repr))
        output = self.output(third_hidden_repr)
        return output
