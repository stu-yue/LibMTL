import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.signal import stft
import pdb
import os


def wgn_plot(data, noise, snr):
    """ data shape is (l/r, seq, dim) """
    x = pd.Series(data + noise)
    print(x.var(), x.skew(), x.kurt())
    plt.figure(figsize=(12, 6), dpi=100)
    k = 1000
    ax = plt.subplot(2, 2, 1)
    ax.plot(range(k), data[:k])
    ax = plt.subplot(2, 2, 2)
    ax.plot(range(k), noise[:k])
    ax = plt.subplot(2, 2, 3)
    ax.plot(range(k), (data + noise)[:k])
    fft_d = np.fft.fft((data + noise))
    fft_d = np.fft.fftshift(fft_d)
    n, fs = len(data), 100
    fft_x = np.array(range(-int(n/2), int(n/2))) * fs / n
    ax = plt.subplot(2, 2, 4)
    ax.plot(fft_x, fft_d, linewidth=0.5)
    
    plt.tight_layout()
    plt.show()


def stft_trfm(data, fs=100, window="hann", nperseg=20):
    """ data shape is (num_samples, l/r, seq, dim) 
    return: new_data shape is (num_samples, l/r, dim, f, t)
    """
    print(f"stft trfm: {data.shape}")
    new_data = []
    stft_func = lambda x: stft(x, fs=fs, window=window, nperseg=nperseg)[2]
    
    for x in data:
        new_data.append([
                [stft_func(x[0, :, j]) for j in range(data.shape[-1])],
                [stft_func(x[1, :, j]) for j in range(data.shape[-1])],
            ])
    new_data = np.array(new_data).real
    
    return new_data

    

def wgn_trfm(data, snr=-1):
    """ data shape is (num_samples, l/r, seq, dim) 
    return: new_data shape is (num_samples, l/r, seq, dim)
    """
    print(f"wgn trfm: {data.shape}, snr: {snr}")
    seq_len = data.shape[-2]
    # print(data)
    Ps = np.sum(np.power(data, 2),  axis=-2) / seq_len  # (bz, 2, dim)
    Ps = Ps[:, :, None, :]                              # (bz, 2, 1, dim)
    Pn = Ps / (np.power(10, snr / 10))
    noise = np.random.randn(*data.shape) * np.sqrt(Pn)
    wgn_plot(data[0, 0, :, -1], noise[0, 0, :, -1], snr=snr)
    return data + noise
    

if __name__ == "__main__":
    # data = np.load("../gait-in-parkinsons-disease-1.0.0/test_data.npy")
    # label =  np.load("../gait-in-parkinsons-disease-1.0.0/test_label.npy")
    # data = np.load("./stft_data.npy")
    # stft_trfm(data)
    # data = wgn_trfm(data, snr=-1)
    # data = np.load("../cnn_output.npy")
    # pdb.set_trace()
    # k = 100
    pdb.set_trace()
    # tsne_plot(data[:k, :-1], data[:k, -1])
    
    