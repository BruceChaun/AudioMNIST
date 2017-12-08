import os


class Config(object):

    data_path = "../data/music"
    train_path = data_path + "/train"
    valid_path = data_path + "/valid"
    test_path = data_path + "/test"
    log_path = "./logs"
    save_path = "saver"
    if not os.path.exists(save_path):
        os.mkdir(save_path)

    model_name = "MLP"

    sr = 44100
    n_mfcc = 40
    
    epochs = 100
    batch_size = 32
    dropout = 0.5
    lr = 0.01
    L2 = 0
    log_interval = 10
