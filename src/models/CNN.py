import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

from utils import *

class CNN2d(nn.Module):

    def __init__(self, n_feature, max_T, n_label, dropout):
        """
        Trust me, CNN cannot handle variable-length sequential data well.
        """
        super(CNN2d, self).__init__()
        self.name = "CNN2d"
        self.max_T = max_T

        self.conv1 = nn.Conv2d(1, 10, (3,1), padding=(1,0)) 
        self.conv2 = nn.Conv2d(10, 20, (3,5))
        self.conv3 = nn.Conv2d(20, 20, (3,5))
        self.conv4 = nn.Conv2d(20, 40, (3,5))
        self.conv2_drop = nn.Dropout2d()

        x = self.__get_features(
                Variable(torch.zeros(1, 1, max_T, n_feature)))
        self.nfeature = int(np.prod(x.size()))

        self.fc1 = nn.Linear(self.nfeature, 500)
        self.fc2 = nn.Linear(500, 50)
        self.fc3 = nn.Linear(50, n_label)


    def __get_features(self, x):
        x = lrelu(F.max_pool2d(self.conv2_drop(self.conv1(x)), 2))
        x = lrelu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = lrelu(F.max_pool2d(self.conv2_drop(self.conv3(x)), 2))
        x = lrelu(F.max_pool2d(self.conv2_drop(self.conv4(x)), 2))
        return x


    def forward(self, x):
        x = lrelu(F.max_pool2d(self.conv2_drop(self.conv1(x)), 2))
        x = lrelu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = lrelu(F.max_pool2d(self.conv2_drop(self.conv3(x)), 2))
        x = lrelu(F.max_pool2d(self.conv2_drop(self.conv4(x)), 2))
        x = x.view(-1, self.nfeature)
        x = F.dropout(x, training=self.training)
        x = lrelu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = lrelu(self.fc2(x))
        x = F.dropout(x, training=self.training)
        x = lrelu(self.fc3(x))
        return F.log_softmax(x)


    def train_(self, data, label, lr, conf):
        self.train()
        opt = optim.Adam(self.parameters(), lr=lr, weight_decay=conf.L2)
        loss_fn = nn.CrossEntropyLoss()

        total_loss = 0.
        total_acc = 0.
        data_size = len(data)

        for batch, (x, y, seq_len) in enumerate(
                batchify(data, label, conf.batch_size, True, True, self.max_T)):
            x = x[:,:self.max_T,:]
            x = Variable(torch.unsqueeze(torch.Tensor(x), 1), volatile=False)
            y = Variable(torch.LongTensor(y))
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

        for batch, (x, y, seq_len) in enumerate(
                batchify(data, label, var_len=True, max_len=self.max_T)):
            x = x[:,:self.max_T,:]
            x = Variable(torch.unsqueeze(torch.Tensor(x), 1), volatile=True)
            y = Variable(torch.LongTensor(y))
            y_hat = self.forward(x)
            loss = loss_fn(y_hat, y)

            total_loss += loss.data.cpu().numpy()[0]
            total_acc += acc(y_hat, y)

        return total_loss/data_size, total_acc/data_size

