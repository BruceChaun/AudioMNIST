import numpy as np
import os

import torch

from logger import Logger
from config import Config
import utils
import pickle

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
    mode = "normal" if conf.model_name == 'MLP' else 'stft'

    # Load data
    if os.path.exists(mode+"_music_train_data.p"):
        with open(mode+"_music_train_data.p", "rb") as f:
            train_data = pickle.load(f, encoding='latin1')
        with open(mode+"_music_train_label.p", "rb") as f:
            train_label = pickle.load(f, encoding='latin1')
        with open(mode+"_music_valid_data.p", "rb") as f:
            valid_data = pickle.load(f, encoding='latin1')
        with open(mode+"_music_valid_label.p", "rb") as f:
            valid_label = pickle.load(f, encoding='latin1')
        with open(mode+"_music_test_data.p", "rb") as f:
            test_data = pickle.load(f, encoding='latin1')
        with open(mode+"_music_test_label.p", "rb") as f:
            test_label = pickle.load(f, encoding='latin1')
    else:
        train_data, train_label = load_data_fn[mode](conf.train_path, conf)
        valid_data, valid_label = load_data_fn[mode](conf.valid_path, conf)
        test_data, test_label = load_data_fn[mode](conf.test_path, conf)

    print("train set = {}".format(len(train_data)))
    print("valid set = {}".format(len(valid_data)))
    print("test set = {}".format(len(test_data)))

    n_features, n_labels = (len(train_data[0]), 9)

    # define model
    if conf.model_name == "MLP":
        model = MLP([n_features, 100, 50, n_labels], conf.dropout)
    elif conf.model_name == "GRU" or conf.model_name == "LSTM":
        n_features = train_data[0].shape[1]
        model = RNN(conf.model_name, n_features, 100, n_labels, 2, conf.dropout)
    elif conf.model_name == "CNN2d":
        max_T = 400 if "music" in conf.data_path else 60
        n_features = train_data[0].shape[1]
        model = CNN2d(n_features, max_T, n_labels, conf.dropout)
    elif conf.model_name == "RBM":
        rbm = RBM(n_features, 128, 1)
        rbm.pretrain(train_data, train_label, conf)
        return 

    # training preparation
    min_loss = float("inf")
    best_acc = 0
    lr = conf.lr

    save_file = os.path.join(conf.save_path, model.name + "_music")
    if os.path.exists(save_file):
        try:
            model = torch.load(save_file)
            min_loss, best_acc = model.evaluate(valid_data, valid_label)
            print("Initial validation loss: {:5.6f}".format(min_loss))
            print("Initial validation acc: {:5.6f}".format(best_acc))
        except Exception:
            #print("[loading existing model error] {}".format(str(e)))
            model = MLP([n_features, 100, 50, n_labels], conf.dropout)

    logger = Logger(conf.log_path)

    if torch.cuda.is_available():
        model.cuda()

    # Training
    for epoch in range(conf.epochs):
        print("Epoch {}".format(epoch))
        train_loss, train_acc = model.train_(train_data, train_label, lr, conf)
        valid_loss, valid_acc = model.evaluate(valid_data, valid_label)

        if valid_acc > best_acc:
            best_acc = valid_acc
            min_loss = valid_loss
            with open(save_file, "wb") as f:
                torch.save(model, f)
        else:
            lr = max(lr*0.9, 1e-4)

        print("Training set\tLoss: {:5.6f}\tAccuracy: {:2.6f}"
                .format(train_loss, train_acc))
        print("Validation set\tLoss: {:5.6f}\tAccuracy: {:2.6f}"
                .format(valid_loss, valid_acc))

        logger.scalar_summary("dev loss", valid_loss, epoch)
        logger.scalar_summary("dev acc", valid_acc, epoch)

    # Test
    model = torch.load(save_file)
    test_loss, test_acc = model.evaluate(test_data, test_label)
    print("Best validation acc: {:5.6f}\nTest set\tLoss: {:5.6f}\tAccuracy: {:2.6f}"
            .format(best_acc, test_loss, test_acc))


if __name__ == "__main__":
    main()
