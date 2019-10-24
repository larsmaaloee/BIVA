import os
import pickle as pkl
import numpy as np
from utils import get_data_path
from urllib.request import urlretrieve
import tarfile
import matplotlib.image as mpimg

def load_imagenet_32():
    datapath = get_data_path("imagenet_32")
    extension = "32x32"
    return load_imagenet(datapath, extension)

def load_imagenet_64():
    datapath = get_data_path("imagenet_64")
    extension = "64x64"
    return load_imagenet(datapath, extension)

def load_imagenet(datapath, extension):

    datasets = [os.path.join(datapath, "train_%s.tar"%extension), os.path.join(datapath, "valid_%s.tar"%extension)]
    dataset = os.path.join(datapath, "imagenet_%s.pkl"%extension)
    urls = ['http://image-net.org/small/train_%s.tar'%extension, 'http://image-net.org/small/valid_%s.tar'%extension]

    if not os.path.exists(datasets[0]):
        for ds, url in zip(datasets, urls):
            if not os.path.isfile(ds):
                print('Downloading data from %s' % url)
                urlretrieve(url, ds)

    train_x = []
    test_x = []
    if not os.path.isfile(dataset):
        for ds, url in zip(datasets, urls):
            print("loading dataset %s..." % ds)
            tar = tarfile.open(ds)
            for member in tar.getmembers():
                f = tar.extractfile(member)
                if f is not None:
                    if "train" in f.name:
                        train_x += [mpimg.imread(f)]
                    else:
                        test_x += [mpimg.imread(f)]
            tar.close()
        pkl.dump([train_x, test_x], open(dataset, "wb"), protocol=4)

    train_x, test_x = pkl.load(open(dataset, "rb"))

    return np.array(train_x, dtype="float32")/255., np.array(test_x, dtype="float32")/255.

