"""
Construct a Gaussian-binary RBM
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable

import utils


class RBM(nn.Module):
    """
    @params
        n_visible: int
            number of visible units
        n_hidden: int
            number of hidden units
        k: int
            Contrastive Divergence k
    """

    def __init__(self, n_visible, n_hidden, k): 
        super(RBM, self).__init__()
        self.n_visible = n_visible
        self.n_hidden = n_hidden
        self.k = k

        self.w = nn.Parameter(torch.randn(n_hidden, n_visible) * 0.01)
        self.v_bias = nn.Parameter(torch.zeros(n_visible))
        self.h_bias = nn.Parameter(torch.zeros(n_hidden))


    def pretrain(self, data, label, conf):
        self.mean = data.mean(0)
        self.std = np.sqrt(data.var(0))
        data = (data - self.mean) / self.std

        opt = optim.Adam(self.parameters(), lr=conf.lr, weight_decay=conf.L2)
        data_size = len(data)

        for epoch in range(conf.epochs):
            total_loss = 0
            for _, (x, y) in enumerate(utils.batchify(data, label, conf.batch_size)):
                x = Variable(torch.Tensor(x), volatile=False)
                self.zero_grad()
                loss = self.CD(x)
                loss.backward()
                opt.step()
                total_loss += loss.data[0]
            print("Epoch {}\tloss: {:5.6f}".format(epoch, total_loss/data_size))


    def CD(self, x):
        """
        Contrastive Divergence k learning algorithm
        """
        h = self._sample_h_given_v(x)
        for _ in range(self.k):
            v = self._sample_v_given_h(h)
            h = self._sample_h_given_v(v)

        loss = self._free_energy(x) - self._free_energy(v)
        return loss


    def _free_energy(self, x):
        #pos = (x - self.v_bias.expand(x.size())) ** 2 / 2
        #pos = pos.sum(1)
        pos = -x.mv(self.v_bias)
        neg = F.linear(x, self.w, self.h_bias).exp().add(1).log().sum(1)
        return (pos - neg).sum()


    def _sample_h_given_v(self, v):
        """
        p(h=1|v)
        """
        p_h = F.sigmoid(F.linear(v, self.w, self.h_bias))
        h_samples = p_h >= Variable(torch.rand(p_h.size()))
        return h_samples.float()


    def _sample_v_given_h(self, h, std=1):
        """
        p(v|h) where v ~ Gaussian
        """
        v = F.linear(h, self.w.t(), self.v_bias)
        v += std * Variable(torch.randn(v.size()))
        return v

