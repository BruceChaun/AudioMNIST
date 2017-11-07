import numpy as np
import os

import torch

from logger import Logger
from config import Config
import utils

from models.MLP import MLP
from models.RNN import RNN
from models.CNN import *
from models.RBM import RBM


def main():
    conf = Config()

    load_data_fn = {
            "normal" : utils.load_data,
            "stft" : utils.extract_stft
            }
    mode = "normal"

    # Load data
    train_data, train_label = load_data_fn[mode](conf.train_path, conf)
    n_features, n_labels = (len(train_data[0]), 11)
    valid_data, valid_label = load_data_fn[mode](conf.valid_path, conf)
    test_data, test_label = load_data_fn[mode](conf.test_path, conf)

    # define model
    if conf.model_name == "MLP":
        model = MLP([n_features, 100, 50, n_labels], conf.dropout)
    elif conf.model_name == "GRU" or conf.model_name == "LSTM":
        n_features = train_data[0].shape[1]
        model = RNN(conf.model_name, n_features, 100, n_labels, 2, conf.dropout)
    elif conf.model_name == "CNN2d":
        max_T = 60
        n_features = train_data[0].shape[1]
        model = CNN2d(n_features, max_T, n_labels, conf.dropout)
    elif conf.model_name == "RBM":
        rbm = RBM(n_features, 128, 1)
        rbm.pretrain(train_data, train_label, conf)
        return 

    # training preparation
    min_loss = float("inf")
    lr = conf.lr

    save_file = os.path.join(conf.save_path, model.name)
    if os.path.exists(save_file):
        try:
            model = torch.load(save_file)
            min_loss, _ = model.evaluate(valid_data, valid_label)
            print("Initial validation loss: {:5.6f}".format(min_loss))
        except RuntimeError, e:
            print("[loading existing model error] {}".format(str(e)))
            model = MLP([n_features, 100, 50, n_labels], conf.dropout)

    logger = Logger(conf.log_path)

    # Training
    for epoch in range(conf.epochs):
        print("Epoch {}".format(epoch))
        train_loss, train_acc = model.train_(train_data, train_label, lr, conf)
        valid_loss, valid_acc = model.evaluate(valid_data, valid_label)

        if valid_loss < min_loss:
            min_loss = valid_loss
            with open(save_file, "wb") as f:
                torch.save(model, f)
        #else:
        #    lr = max(lr*0.9, 1e-5)

        print("Training set\tLoss: {:5.6f}\tAccuracy: {:2.6f}"
                .format(train_loss, train_acc))
        print("Validation set\tLoss: {:5.6f}\tAccuracy: {:2.6f}"
                .format(valid_loss, valid_acc))

        logger.scalar_summary("dev loss", valid_loss, epoch)
        logger.scalar_summary("dev acc", valid_acc, epoch)

    # Test
    model = torch.load(save_file)
    test_loss, test_acc = model.evaluate(test_data, test_label)
    print("Best validation loss: {:5.6f}\nTest set\tLoss: {:5.6f}\tAccuracy: {:2.6f}"
            .format(min_loss, test_loss, test_acc))


if __name__ == "__main__":
    main()
