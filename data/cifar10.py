import os
import tarfile
import pickle as pkl
import numpy as np
from utils import get_data_path
from urllib.request import urlretrieve

def quantisize(images, levels):
    return (np.digitize(images, np.arange(levels) / levels) - 1).astype('i')

def load_cifar(levels=256, with_y=False):
    dataset = 'cifar-10-python.tar.gz'
    data_dir, data_file = os.path.split(dataset)
    if data_dir == "" and not os.path.isfile(dataset):
        # Check if dataset is in the data directory.
        new_path = os.path.join(
            get_data_path("cifar10"),
            dataset
        )
        if os.path.isfile(new_path) or data_file == 'cifar-10-python.tar.gz':
            dataset = new_path

    if (not os.path.isfile(dataset)) and data_file == 'cifar-10-python.tar.gz':
        origin = (
            'http://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz'
        )
        print("Downloading data from {}...".format(origin))
        urlretrieve(origin, dataset)

    f = tarfile.open(dataset, 'r:gz')
    b1 = pkl.load(f.extractfile("cifar-10-batches-py/data_batch_1"), encoding="bytes")
    b2 = pkl.load(f.extractfile("cifar-10-batches-py/data_batch_2"), encoding="bytes")
    b3 = pkl.load(f.extractfile("cifar-10-batches-py/data_batch_3"), encoding="bytes")
    b4 = pkl.load(f.extractfile("cifar-10-batches-py/data_batch_4"), encoding="bytes")
    b5 = pkl.load(f.extractfile("cifar-10-batches-py/data_batch_5"), encoding="bytes")
    test = pkl.load(f.extractfile("cifar-10-batches-py/test_batch"), encoding="bytes")
    train_x = np.concatenate([b1[b'data'], b2[b'data'], b3[b'data'], b4[b'data'], b5[b'data']], axis=0)/255.
    train_x = np.asarray(train_x, dtype='float32')
    train_t = np.concatenate([np.array(b1[b'labels']),
                              np.array(b2[b'labels']),
                              np.array(b3[b'labels']),
                              np.array(b4[b'labels']),
                              np.array(b5[b'labels'])], axis=0)

    test_x = test[b'data']/255.
    test_x = np.asarray(test_x, dtype='float32')
    test_t = np.array(test[b'labels'])
    f.close()


    train_x = train_x.reshape((train_x.shape[0], 3, 32, 32)).transpose((0, 2, 3, 1)).reshape((train_x.shape[0], -1))
    test_x = test_x.reshape((test_x.shape[0], 3, 32, 32)).transpose((0, 2, 3, 1)).reshape((test_x.shape[0], -1))

    train_x = quantisize(train_x, levels)/(levels-1.)
    test_x = quantisize(test_x, levels)/(levels-1.)

    if with_y:
        return (train_x, train_t), (test_x, test_t)
    return train_x, test_x
