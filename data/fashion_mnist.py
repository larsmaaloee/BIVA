import os
from utils import get_data_path
from urllib.request import urlretrieve


def load_mnist(path, kind='train'):
    import os
    import gzip
    import numpy as np

    """Load MNIST data from `path`"""
    labels_path = os.path.join(path, '%s-labels-idx1-ubyte.gz' % kind)
    images_path = os.path.join(path, '%s-images-idx3-ubyte.gz' % kind)

    with gzip.open(labels_path, 'rb') as lbpath:
        labels = np.frombuffer(lbpath.read(), dtype=np.uint8,
                               offset=8)

    with gzip.open(images_path, 'rb') as imgpath:
        images = np.frombuffer(imgpath.read(), dtype=np.uint8,
                               offset=16).reshape(len(labels), 784)

    return np.array(images, dtype="float32") / 255., np.array(labels, dtype="float32")


def load_fashion_mnist(with_y=False):
    datapath = get_data_path("fashion_mnist")
    paths = ["train-images-idx3-ubyte.gz", "train-labels-idx1-ubyte.gz", "t10k-images-idx3-ubyte.gz",
             "t10k-labels-idx1-ubyte.gz"]
    datasets = [os.path.join(datapath, fp) for fp in paths]

    if not os.path.isfile(datasets[0]):
        urls = [os.path.join("http://fashion-mnist.s3-website.eu-central-1.amazonaws.com", fp) for fp in paths]

        for url, fn in zip(urls, paths):
            print("Downloading %s data..." % (fn))
            urlretrieve(url, os.path.join(datapath, fn))

    train_x, train_t = load_mnist(datapath, "train")
    test_x, test_t = load_mnist(datapath, "t10k")

    if with_y:
        return (train_x, train_t), (test_x, test_t)

    return train_x, test_x
