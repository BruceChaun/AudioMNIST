import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import torch.optim as optim
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence

from utils import *


class RNN(nn.Module):

    def __init__(self, cell_name, ninp, nhid, nout, nlayers, dropout):
        """
        Apply multi-layer stacked bi-RNN
        """
        super(RNN, self).__init__()
        self.name = cell_name
        self.ninp = ninp
        self.nhid = nhid
        self.nlayers = nlayers
        self.rnn = getattr(nn, cell_name)(ninp, nhid, nlayers, batch_first=True, 
                dropout=dropout, bidirectional=True)

        self.fc1 = nn.Linear(nhid, nout)


    def init_hidden(self, batch_size):
        weight = next(self.parameters()).data
        if self.name == "LSTM":
            return (Variable(weight.new(self.nlayers*2, batch_size, self.nhid).zero_()),
                    Variable(weight.new(self.nlayers*2, batch_size, self.nhid).zero_()))
        else:
            return Variable(weight.new(self.nlayers*2, batch_size, self.nhid).zero_())


    def __retrieve_final_output(self, output, seq_len):
        batch_size = output.size(0)
        max_len = output.size(1)
        idx = torch.range(0, batch_size-1).long() * max_len + torch.LongTensor(seq_len-1)
        final_output = output.view(batch_size*max_len, output.size(2))
        final_output = final_output.index_select(0, Variable(idx))
        return final_output


    def forward(self, x, h, seq_len):
        packed_seq = pack_padded_sequence(x, seq_len, batch_first=True)
        output, h = self.rnn(packed_seq, h)
        output, out_len = pad_packed_sequence(output, True)

        feature = output[:,:,:self.nhid] + output[:,:,self.nhid:]
        feature = self.__retrieve_final_output(feature, seq_len)

        o = self.fc1(feature)
        return F.log_softmax(o)


    def train_(self, data, label, lr, conf):
        self.train()
        opt = optim.Adam(self.parameters(), lr=lr, weight_decay=conf.L2)
        loss_fn = nn.CrossEntropyLoss()

        total_loss = 0.
        total_acc = 0.
        data_size = len(data)

        for batch, (x, y, seq_len) in enumerate(batchify(data, label, conf.batch_size, True, True)):
            h = self.init_hidden(len(x))
            x = Variable(torch.Tensor(x), volatile=False)
            y = Variable(torch.LongTensor(y))
            self.zero_grad()
            y_hat = self.forward(x, h, seq_len)
            loss = loss_fn(y_hat, y)
            loss.backward()
            opt.step()

            total_loss += loss.data[0]
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

        for batch, (x, y, seq_len) in enumerate(batchify(data, label, var_len=True)):
            h = self.init_hidden(len(x))
            x = Variable(torch.Tensor(x), volatile=True)
            y = Variable(torch.LongTensor(y))
            y_hat = self.forward(x, h, seq_len)
            loss = loss_fn(y_hat, y)

            total_loss += loss.data[0]
            total_acc += acc(y_hat, y)

        return total_loss/data_size, total_acc/data_size

