import numpy as np
import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import re
import torch
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split
from trfms import stft_trfm, wgn_trfm
import pdb


def segment_overlap(data, T=300, stride=50, slices=[]):
    """ data shape is (timestamp, cols_1_19) 
    :return: (num_samples, l/r, seq, dim)
    """
    seg_nums = (data.shape[0] - T + 1) // stride
    seg_data = []
    start, end = 0, T
    for i in range(seg_nums):
        seg_data.append([
                data[start: end, slices[0]],
                data[start: end, slices[1]],
            ])
        start += stride
        end = start + T
    seg_data = np.array(seg_data)
    return seg_data


def smooth(data, w=15):
    ''' smooth raw data [low-pass filer]
    :param w: smooth window size
    :return: smoothed data
    '''
    weight = np.ones(w) / w
    result = np.convolve(weight, data[:, 0], 'valid').reshape((-1, 1))
    for i in range(1, data.shape[1]):
        result = np.concatenate((result, np.convolve(weight, data[:, i], 'valid').reshape((-1, 1))), axis=1)
    return result


def check_GT(dir_path="../gaittracker-1.0.0/"):
    """ check for gaittracker dataset
    Columns	1-3	    : acc   on the right thigh, [x, y, z]
    Columns	4-6	    : gyro  on the right thigh, [x, y, z]
    Columns	7-9	    : acc   on the right shank, [x, y, z]
    Columns	10-12	: gyro  on the right shank, [x, y, z]
    Columns	13-15	: acc   on the left thigh,  [x, y, z]
    Columns	16-18	: gyro  on the left thigh,  [x, y, z]
    Columns	19-21	: acc   on the left shank,  [x, y, z]
    Columns	22-24	: gyro  on the left shank,  [x, y, z]
    :return         : data-(numsamples, 2, seq, 12), label-(num_samples, 3)
    """
    names = os.listdir(dir_path)
    co_names = [name for name in names if name.startswith("Co")]
    print(co_names)
    all_data = []
    all_label = []
    for name in co_names:
        data = np.loadtxt(dir_path + name)
        data = smooth(data)
        seg_data = segment_overlap(
            data, T=200, stride=100,
            slices=[list(range(0, 12)), list(range(12, 24))],
        )
        labels = np.array(re.findall(r"(\d+)", name), dtype=np.int32)
        all_data.append(seg_data)
        all_label.append(np.array([labels] * len(seg_data)))

    all_data = np.concatenate(all_data, axis=0)
    # NOTE: whether to enable STFT transforms
    # all_data = stft_trfm(all_data, fs=100, window="hann", nperseg=20)
    all_label = np.concatenate(all_label, axis=0)
    pdb.set_trace()
    
    tra_val_data, tes_data, tra_val_label, tes_label = train_test_split(
        all_data, all_label, test_size=0.2, random_state=0,
    )
    np.save(os.path.join(dir_path, "train_data.npy"), tra_val_data)
    np.save(os.path.join(dir_path, "train_label.npy"), tra_val_label)
    np.save(os.path.join(dir_path, "test_data.npy"), tes_data)
    np.save(os.path.join(dir_path, "test_label.npy"), tes_label)
    
    
class GaitTackerDataset(Dataset):
    LOAD_LABELS = [0, 5]
    SLOPE_LABELS = [0, 5, 10]
    SPEED_LABELS = [2, 3, 4]
    """ load gaittracker dataset """
    def  __init__(
        self,
        data_root="../gaittracker-1.0.0/",
        mode="train",
        multi=False,
    ):
        """ data shape is (num_samples, 2, seq, 12) """
        super().__init__()
        self.data_root = data_root
        self.mode = mode
        self.multi = multi
        
        # data shape: (num_samples, l/r, dim, f, t)
        self.data, self.label = self._load_data()
        self.label = np.array([
            np.array([
                co - 1,
                self.LOAD_LABELS.index(w),
                self.SLOPE_LABELS.index(sl),
                self.SPEED_LABELS.index(sp),
            ]) for co, w, sl, sp in self.label
        ])
        print(f"label shape is {self.label.shape}")
        self.c = self.data.shape[3]

    def _load_data(self,):
        if self.mode == 'train':
            data_path = os.path.join(self.data_root, "train_data.npy")
            label_path = os.path.join(self.data_root, "train_label.npy")
        elif self.mode == 'test':
            data_path = os.path.join(self.data_root, "test_data.npy")
            label_path = os.path.join(self.data_root, "test_label.npy")
        data, label = np.load(data_path), np.load(label_path)
        return data, label

    def __getitem__(self, index):
        data = torch.tensor(self.data[index], dtype=torch.float32)
        label = torch.tensor(self.label[index], dtype=torch.int64)
        if self.multi:
            label = {"load": label[1], "slope": label[2], "speed": label[3]}
        else:
            label = label[-1]
        return data, label
    
    def __len__(self):
        return len(self.label)


if __name__ == "__main__":
    check_GT()
    # dataset = GaitTackerDataset()