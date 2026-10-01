import torch
from torch import nn

class BaselineModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super().__init__()
        self.lstm = nn.LSTM(input_size = input_size, hidden_size = hidden_size, num_layers = num_layers, batch_first = True)
        self.classifier = nn.Linear(hidden_size, num_classes)

    def forward(self, inputs):
        outputs, (h_n, c_n) = self.lstm(inputs)
        last_hidden = h_n[-1]
        outputs = self.classifier(last_hidden)
        return outputs