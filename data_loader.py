import numpy as np
import scipy.io as sio
from torch.utils.data import Dataset, DataLoader
from utils import aligned_data_split, normalize
import torch
import random
from sklearn.preprocessing import MinMaxScaler
from numpy.random import randint
from sklearn.preprocessing import OneHotEncoder


def load_data(dataset, aligned_prop):
    all_data = []
    train_pairs = []
    un_align = []
    label = []
    un_label_X = []
    un_label_Y = []

    mat = sio.loadmat('./data/' + dataset + '.mat')

    if dataset == 'Scene15':
        data = mat['X'][0][0:2]
        label = np.squeeze(mat['Y'])

    elif dataset == 'HandWritten':
        scaler = MinMaxScaler()
        data = []
        label = np.squeeze(mat['Y'])
        data.append(scaler.fit_transform(mat['X'][0][0]))
        data.append(scaler.fit_transform(mat['X'][0][2]))

    elif dataset == 'Reuters':
        data = [normalize(np.vstack((mat['x_train'][0], mat['x_test'][0]))),
                normalize(np.vstack((mat['x_train'][1], mat['x_test'][1])))]
        label = np.squeeze(np.hstack((mat['y_train'], mat['y_test'])))

    elif dataset == 'BDGP':
        data = [mat['X1'], mat['X2']]
        label = np.squeeze(mat['Y'])

    elif dataset == 'Fashion':
        data = [mat['X1'].reshape(10000, -1).astype('float32'), mat['X2'].reshape(10000, -1).astype('float32')]
        label = np.squeeze(mat['Y'])

    elif dataset == 'Caltech101-7':
        data = []
        label = np.squeeze(mat['gt'])
        data.append(mat['X'][0][3].T)
        data.append(mat['X'][0][4].T)

    elif dataset == 'Caltech101-20':
        data = [mat['X1'], mat['X2']]
        label = np.squeeze(mat['Y'])

    divide_seed = random.randint(0, 1000)
    print('divide_seed:', divide_seed)
    # divide_seed = 100
    align_idx, un_align_idx, mask = aligned_data_split(len(label), aligned_prop, divide_seed)
    align_label, un_align_label = label[align_idx], label[un_align_idx]
    align_X, align_Y, un_align_X, un_align_Y = (
        data[0][align_idx], data[1][align_idx], data[0][un_align_idx], data[1][un_align_idx])

    if aligned_prop == 1:
        all_data.append(align_X.T)
        all_data.append(align_Y.T)
        all_label, all_label_X, all_label_Y = align_label, align_label, align_label
    else:
        shuffle_idx = random.sample(range(len(un_align_Y)), len(un_align_Y))
        un_align_Y = un_align_Y[shuffle_idx]
        un_align_label_X, un_align_label_Y = un_align_label, un_align_label[shuffle_idx]

        un_align.append(un_align_X)
        un_align.append(un_align_Y)
        un_label_X = un_align_label_X
        un_label_Y = un_align_label_Y

        all_data.append(np.concatenate((align_X, un_align_X)).T)
        all_data.append(np.concatenate((align_Y, un_align_Y)).T)
        all_label = np.concatenate((align_label, un_align_label))
        all_label_X = np.concatenate((align_label, un_align_label_X))
        all_label_Y = np.concatenate((align_label, un_align_label_Y))

    view0, view1 = get_pairs(align_X, align_Y)

    train_pairs.append(view0.T)
    train_pairs.append(view1.T)

    return (train_pairs, align_label, all_data, all_label, all_label_X, all_label_Y,
            un_align, un_label_X, un_label_Y, divide_seed, mask)


def get_pairs(align_X, align_Y):
    view0, view1 = [], []
    for i in range(len(align_X)):
        view0.append(align_X[i])
        view1.append(align_Y[i])

    view0, view1 = np.array(view0, dtype=np.float32), np.array(view1, dtype=np.float32)
    return view0, view1


class getAllDataset(Dataset):
    def __init__(self, data, labels, class_labels0, class_labels1, mask):
        self.data = data
        self.labels = labels
        self.class_labels0 = class_labels0
        self.class_labels1 = class_labels1
        self.mask = mask

    def __getitem__(self, index):
        fea0, fea1 = (torch.from_numpy(self.data[0][:, index])).type(torch.FloatTensor), (
            torch.from_numpy(self.data[1][:, index])).type(torch.FloatTensor)
        fea0, fea1 = fea0.unsqueeze(0), fea1.unsqueeze(0)
        label = np.int64(self.labels[index])
        class_labels0 = np.int64(self.class_labels0[index])
        class_labels1 = np.int64(self.class_labels1[index])
        mask = np.int64(self.mask[index])
        return fea0, fea1, label, class_labels0, class_labels1, mask

    def __len__(self):
        return len(self.labels)


def loader_cl(batch_size, aligned_prop, dataset):

    (train_pairs, align_label, all_data, all_label, all_label_X, all_label_Y,
     un_align, un_label_X, un_label_Y, divide_seed, mask) = (
        load_data(dataset, aligned_prop))

    train_pair_dataset = getAllDataset(train_pairs, align_label, align_label, align_label, mask)
    all_dataset = getAllDataset(all_data, all_label, all_label_X, all_label_Y, mask)
    un_align_dataset = getAllDataset(un_align, un_label_X, un_label_X, un_label_Y, mask)

    train_pair_loader = DataLoader(
        train_pair_dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True
    )
    all_loader = DataLoader(
        all_dataset,
        batch_size=batch_size,  # 1024
        shuffle=True
    )

    if len(un_align_dataset) == 0:
        un_align_loader = 0
    else:
        un_align_loader = DataLoader(
            un_align_dataset,
            batch_size=1024,
            shuffle=True
        )

    if len(all_label) > 10000:
        pvp_loader = DataLoader(
            all_dataset,
            batch_size=1024,
            shuffle=True
        )
    else:
        pvp_loader = DataLoader(
            all_dataset,
            batch_size=len(all_label),
            shuffle=True
        )

    return train_pair_loader, all_loader, un_align_loader, pvp_loader, divide_seed

