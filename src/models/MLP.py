import numpy as np

import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import torch.optim as optim

from utils import *


class MLP(nn.Module):

    def __init__(self, layers, dropout):
        """
        @param 
            layers: list of number of neurons in each layer, 
            both input and output layers are included
        """
        super(MLP, self).__init__()
        self.name = "MLP"
        self.dropout = dropout

        self.fc1 = nn.Linear(layers[0], layers[1])
        self.fc2 = nn.Linear(layers[1], layers[2])
        self.fc3 = nn.Linear(layers[2], layers[3])
        self.fc = [self.fc1, self.fc2, self.fc3]

        """
        for i in range(len(layers) - 1):
            self.fc.append(nn.Linear(layers[i], layers[i+1]))
        """

    def forward(self, x):
        for i in range(len(self.fc)):
            x = lrelu(self.fc[i](x))
            x = F.dropout(x, training=self.training)
        x = F.log_softmax(x)
        return x

    def train_(self, data, label, lr, conf):
        self.train()
        opt = optim.Adam(self.parameters(), lr=lr, weight_decay=conf.L2)
        loss_fn = nn.CrossEntropyLoss()

        total_loss = 0.
        total_acc = 0.
        data_size = len(data)

        for batch, (x, y) in enumerate(batchify(data, label, conf.batch_size, True)):
            x = Variable(torch.Tensor(x), volatile=False)
            y = Variable(torch.LongTensor(y))
            if torch.cuda.is_available:
                x = x.cuda()
                y = y.cuda()

            self.zero_grad()
            y_hat = self.forward(x)
            loss = loss_fn(y_hat, y)
            loss.backward()
            opt.step()

            total_loss += loss.data.cpu().numpy()[0]
            total_acc += acc(y_hat, y)

            if (batch + 1) % conf.log_interval == 0:
                size = conf.batch_size * batch + len(x)
                print("[{:5d}/{:5d}] batches\tLoss: {:5.6f}\tAccuracy: {:2.6f}"
                        .format(size, data_size, total_loss / size, total_acc / size))

        return total_loss/data_size, total_acc/data_size


    def evaluate(self, data, label):
        self.eval()
        loss_fn = nn.CrossEntropyLoss()

        total_loss = 0.
        total_acc = 0.
        data_size = len(data)

        for batch, (x, y) in enumerate(batchify(data, label)):
            x = Variable(torch.Tensor(x), volatile=True)
            y = Variable(torch.LongTensor(y))
            if torch.cuda.is_available:
                x = x.cuda()
                y = y.cuda()

            y_hat = self.forward(x)
            loss = loss_fn(y_hat, y)

            total_loss += loss.data.cpu().numpy()[0]
            total_acc += acc(y_hat, y)

        return total_loss/data_size, total_acc/data_size
