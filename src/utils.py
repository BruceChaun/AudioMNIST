import librosa
import os
import random
import numpy as np
import torch


def lrelu(x, leaky=0.01):
    return torch.max(x, leaky * x)


def acc(output, target):
    pred = output.data.max(1)[1] # get the index of the max log-probability
    correct = pred.eq(target.data).cpu().sum()
    return correct


def load_dataset(path, sr):
    label_mapping = {
            "z" : 0, 
            "0" : 0, 
            "1" : 1, 
            "2" : 2, 
            "3" : 3, 
            "4" : 4, 
            "5" : 5, 
            "6" : 6, 
            "7" : 7, 
            "8" : 8, 
            "9" : 9, 
            "o" : 10
            }
    dataset = []
    for f in os.listdir(path):
        audio, _ = librosa.load(os.path.join(path, f), sr, duration=5)
        dataset.append((audio, label_mapping[f[0]]))
    return dataset


def load_data(path, conf):
    data = load_dataset(path, conf.sr)
    features = []
    labels = []
    for x, y in data:
        feature = np.concatenate(extract_features(x, conf.sr, conf.n_mfcc))
        features.append(feature)
        labels.append(y)

    return np.array(features), np.array(labels)


def extract_features(audio, sr, n_mfcc):
    """
    Extract features from @audio

    The features are 
        * mfcc
        * chroma
        * mel
        * spectral_contrast
        * tonnetz

    Refer to http://aqibsaeed.github.io/2016-09-03-urban-sound-classification-part-1/

    @param
        audio: audio file loaded from librosa
        sr: sample rate
        n_mfcc: number of mfcc components

    @return
        tuple of five extracted features
    """
    stft = np.abs(librosa.stft(audio))
    mfcc = np.mean(librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=n_mfcc).T,axis=0)
    chroma = np.mean(librosa.feature.chroma_stft(S=stft, sr=sr).T,axis=0)
    mel = np.mean(librosa.feature.melspectrogram(audio, sr=sr).T,axis=0)
    sc = np.mean(librosa.feature.spectral_contrast(S=stft, sr=sr).T,axis=0)
    tonnetz = np.mean(librosa.feature.tonnetz(
        y=librosa.effects.harmonic(audio), sr=sr).T,axis=0)

    return (mfcc, chroma, mel, sc, tonnetz)


def extract_stft(path, conf):
    """
    Only extract stft features, using default settings
    """
    dataset = load_dataset(path, conf.sr)
    stft = []
    label = []
    for x, y in dataset:
        stft.append(np.abs(librosa.stft(x).T))
        label.append(y)
    return np.array(stft), np.array(label)


def padding(batch_data, max_len=None):
    """
    For variable-length data in a batch, zero-padding the short ones

    @param 
        batch_data: numpy array, has dimension (batch_size, seq_len, n_feature)
        max_len: int, max sequence length

    @return 
        padded_data: numpy array, same format as batch_data, but padded
        seq_lens: list of lengths of sequences in a batch
    """
    if max_len:
        batch_data[0] = batch_data[0][:max_len,:]
        n_padding = max_len - batch_data[0].shape[0]
        batch_data[0] = np.concatenate(
                (batch_data[0], np.zeros((n_padding, batch_data[0].shape[1]))))

    batch_size = len(batch_data)
    lens = [data.shape[0] for data in batch_data]
    max_len = max(lens)
    nfeature = batch_data[0].shape[1]
    padded_data = np.zeros([batch_size, max_len, nfeature])
    for i in range(batch_size):
        length = batch_data[i].shape[0]
        pad = np.pad(batch_data[i], (0, max_len-length), "constant")
        padded_data[i] = pad[:,:nfeature]

    return padded_data, np.array(lens)


def batchify(data, label, batch_size=None, shuffle=False, var_len=False, max_len=None):
    if not batch_size:
        batch_size = len(data)

    data_size = len(data)
    order = list(range(data_size))
    if shuffle:
        random.shuffle(order)

    batches = int(np.ceil(1.*data_size/batch_size))
    for i in range(batches):
        start = i * batch_size
        indices = order[start:start+batch_size]
        if var_len:
            x, seq_len = padding(data[indices], max_len)
            idx = np.flip(np.argsort(seq_len), 0) # sort descendantly
            yield (x[idx], label[indices][idx], seq_len[idx])
        else:
            yield (data[indices], label[indices])


if __name__ == '__main__':
    import config
    import pickle
    conf = config.Config()

    load_data_fn = {
            "normal" : load_data,
            "stft" : extract_stft
            }
    mode = "normal" if conf.model_name == 'MLP' else 'stft'
    print(mode)

    # Load data
    train_data, train_label = load_data_fn[mode](conf.train_path, conf)
    n_features, n_labels = (len(train_data[0]), 11)
    valid_data, valid_label = load_data_fn[mode](conf.valid_path, conf)
    test_data, test_label = load_data_fn[mode](conf.test_path, conf)

    print("train {}".format(len(train_data)))
    print("valid {}".format(len(valid_data)))
    print("test {}".format(len(test_data)))

    pickle.dump(train_data, open(mode+"_music_train_data.p", "wb"))
    pickle.dump(train_label, open(mode+"_music_train_label.p", "wb"))
    pickle.dump(valid_data, open(mode+"_music_valid_data.p", "wb"))
    pickle.dump(valid_label, open(mode+"_music_valid_label.p", "wb"))
    pickle.dump(test_data, open(mode+"_music_test_data.p", "wb"))
    pickle.dump(test_label, open(mode+"_music_test_label.p", "wb"))
