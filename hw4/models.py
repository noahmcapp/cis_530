# models.py
import torch.nn as nn
from torch.autograd import Variable
from torch import optim
import torch
import torch.nn.functional as F

'''
Please add default values for all the parameters of __init__.
'''


class CharRNNClassify(nn.Module):
    def __init__(self, input_size=57, hidden_size=110, output_size=9):
        super().__init__()
        self.hidden_size = hidden_size
        self.gru = nn.GRU(input_size, hidden_size)
        self.h2o = nn.Linear(hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim=1)
        self.relu = nn.LeakyReLU()

    def forward(self, input, hidden=None):
        out, hidden = self.gru(input.view(1, 1, -1), hidden)
        output = self.h2o(self.relu(out[:, -1]))
        output = self.softmax(output)
        return output, hidden

    def initHidden(self):
        return torch.zeros(1, 1, self.hidden_size)


### Unused structures

## RNN
# class CharRNNClassify(nn.Module):
#     def __init__(self, input_size=57, hidden_size=50, output_size=9):
#       super().__init__()
#       self.hidden_size = hidden_size
#       self.i2h = nn.Linear(input_size + hidden_size, hidden_size)
#       self.i2o = nn.Linear(input_size + hidden_size, output_size)
#       self.softmax = nn.LogSoftmax(dim=1)
#       self.transform = nn.Tanh()
#
#     def forward(self, input, hidden=None):
#       combined = torch.cat((input, hidden), 1)
#       hidden = self.transform(self.i2h(combined))
#       output = self.transform(self.i2o(combined))
#       output = self.softmax(output)
#       return output, hidden
#
#     def initHidden(self):
#       return torch.zeros(1, self.hidden_size)

## RNN with LSTM
# class CharRNNClassify(nn.Module):
#     def __init__(self, input_size=57, hidden_size=50, output_size=9):
#       super().__init__()
#       self.hidden_size = hidden_size
#       self.lstm = nn.LSTM(input_size, hidden_size)
#       self.h2o = nn.Linear(hidden_size, output_size)
#       self.softmax = nn.LogSoftmax(dim=2)
#       self.tanh = nn.Tanh()
#
#     def forward(self, input, hidden=None):
#       out, hidden = self.lstm(input.view(1, 1, -1), hidden)
#       output = self.h2o(self.tanh(hidden[0]))
#       output = self.softmax(output)
#       return output.view(1, -1), hidden
#
#     def initHidden(self):
#       return (torch.zeros(1, 1, self.hidden_size), torch.zeros(1, 1, self.hidden_size))
