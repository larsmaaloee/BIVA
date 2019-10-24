import os
from utils import get_data_path
from urllib.request import urlretrieve
from scipy.io import loadmat


def load_omniglot():

    datapath = get_data_path("omniglot")
    dataset = os.path.join(datapath, "chardata.mat")

    if not os.path.isfile(dataset):
        origin = (
            'https://github.com/yburda/iwae/raw/'
            'master/datasets/OMNIGLOT/chardata.mat'
        )
        print('Downloading data from %s' % origin)
        urlretrieve(origin, dataset)

    data = loadmat(dataset)

    train_x = data['data'].astype('float32').T
    test_x = data['testdata'].astype('float32').T

    return train_x, test_x
