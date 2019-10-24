import numpy as np
from models import BIVA
import sys
from custom import DeepVAEEvaluator
from data import load_mnist_binarized, load_cifar
from utils import AdamaxOptimizer

import tensorflow as tf

config = tf.ConfigProto()
config.gpu_options.allow_growth = False


def get_deep_vae_mnist(name):
    filters = 64
    no_layers = 2
    enc = []
    z = []

    enc_z1 = [[filters, (5, 5), (1, 1)]] * no_layers
    enc_z1 += [[filters, (5, 5), (2, 2)]]
    z_1 = 48
    enc += [enc_z1]
    z += [z_1]

    enc_z2 = [[filters, (3, 3), (1, 1)]] * no_layers
    enc_z2 += [[filters, (3, 3), (1, 1)]]
    z_2 = 40
    enc += [enc_z2]
    z += [z_2]

    enc_z3 = [[filters, (3, 3), (1, 1)]] * no_layers
    enc_z3 += [[filters, (3, 3), (1, 1)]]
    z_3 = 32
    enc += [enc_z3]
    z += [z_3]

    enc_z4 = [[filters, (3, 3), (1, 1)]] * no_layers
    enc_z4 += [[filters, (3, 3), (1, 1)]]
    z_4 = 24
    enc += [enc_z4]
    z += [z_4]

    enc_z5 = [[filters, (3, 3), (1, 1)]] * no_layers
    enc_z5 += [[filters, (3, 3), (1, 1)]]
    z_5 = 16
    enc += [enc_z5]
    z += [z_5]

    enc_z6 = [[filters, (3, 3), (1, 1)]] * no_layers
    enc_z6 += [[filters, (3, 3), (2, 2)]]
    z_6 = 8
    enc += [enc_z6]
    z += [z_6]

    return BIVA(tf.Session(), [28, 28, 1], enc, z, tf.nn.elu, "bernoulli", model_name=name, dropout_inference=.5,
                dropout_generative=.5, eps=1e-8, is_log_var=True, minimum_kl=-2.)


def get_deep_vae_cifar(name):
    filters = 96
    no_layers = 2
    enc = []
    z = []

    enc_z1 = [[filters, (5, 5), (1, 1)]] * no_layers
    enc_z1 += [[filters, (5, 5), (2, 2)]]
    z_1 = [38, (16, 16), (1, 1)]
    enc += [enc_z1]
    z += [z_1]

    enc_z2 = [[filters, (3, 3), (1, 1)]] * no_layers
    enc_z2 += [[filters, (3, 3), (1, 1)]]
    z_2 = [36, (16, 16), (1, 1)]
    enc += [enc_z2]
    z += [z_2]

    enc_z3 = [[filters, (3, 3), (1, 1)]] * no_layers
    enc_z3 += [[filters, (3, 3), (1, 1)]]
    z_3 = [34, (16, 16), (1, 1)]
    enc += [enc_z3]
    z += [z_3]

    enc_z4 = [[filters, (3, 3), (1, 1)]] * no_layers
    enc_z4 += [[filters, (3, 3), (1, 1)]]
    z_4 = [32, (16, 16), (1, 1)]
    enc += [enc_z4]
    z += [z_4]

    enc_z5 = [[filters, (3, 3), (1, 1)]] * no_layers
    enc_z5 += [[filters, (3, 3), (1, 1)]]
    z_5 = [30, (16, 16), (1, 1)]
    enc += [enc_z5]
    z += [z_5]

    enc_z6 = [[filters, (3, 3), (1, 1)]] * no_layers
    enc_z6 += [[filters, (3, 3), (1, 1)]]
    z_6 = [28, (16, 16), (1, 1)]
    enc += [enc_z6]
    z += [z_6]

    enc_z7 = [[filters, (3, 3), (1, 1)]] * no_layers
    enc_z7 += [[filters, (3, 3), (1, 1)]]
    z_7 = [26, (16, 16), (1, 1)]
    enc += [enc_z7]
    z += [z_7]

    enc_z8 = [[filters, (3, 3), (1, 1)]] * no_layers
    enc_z8 += [[filters, (3, 3), (1, 1)]]
    z_8 = [24, (16, 16), (1, 1)]
    enc += [enc_z8]
    z += [z_8]

    enc_z9 = [[filters, (3, 3), (1, 1)]] * no_layers
    enc_z9 += [[filters, (3, 3), (1, 1)]]
    z_9 = [22, (16, 16), (1, 1)]
    enc += [enc_z9]
    z += [z_9]

    enc_z10 = [[filters, (3, 3), (1, 1)]] * no_layers
    enc_z10 += [[filters, (3, 3), (1, 1)]]
    z_10 = [20, (16, 16), (1, 1)]
    enc += [enc_z10]
    z += [z_10]

    enc_z11 = [[filters, (3, 3), (1, 1)]] * no_layers
    enc_z11 += [[filters, (3, 3), (2, 2)]]
    z_11 = [18, (8, 8), (1, 1)]
    enc += [enc_z11]
    z += [z_11]

    enc_z12 = [[filters, (3, 3), (1, 1)]] * no_layers
    enc_z12 += [[filters, (3, 3), (1, 1)]]
    z_12 = [16, (8, 8), (1, 1)]
    enc += [enc_z12]
    z += [z_12]

    enc_z13 = [[filters, (3, 3), (1, 1)]] * no_layers
    enc_z13 += [[filters, (3, 3), (1, 1)]]
    z_13 = [14, (8, 8), (1, 1)]
    enc += [enc_z13]
    z += [z_13]

    enc_z14 = [[filters, (3, 3), (1, 1)]] * no_layers
    enc_z14 += [[filters, (3, 3), (1, 1)]]
    z_14 = [12, (8, 8), (1, 1)]
    enc += [enc_z14]
    z += [z_14]

    enc_z15 = [[filters, (3, 3), (1, 1)]] * no_layers
    enc_z15 += [[filters, (3, 3), (2, 2)]]
    z_15 = [10, (4, 4), (1, 1)]
    enc += [enc_z15]
    z += [z_15]

    return BIVA(tf.Session(), [32, 32, 3], enc, z, tf.nn.elu, "discretized", model_name=name, dropout_inference=.2,
                eps=1e-8, is_log_var=True, minimum_kl=-2.)


if __name__ == "__main__":

    dataset_arg = sys.argv[1].lower()

    deep_vae = None

    if "mnist" in dataset_arg:

        print("Training a MNIST binarized model...")
        deep_vae = get_deep_vae_mnist("mnist_binarized")
        train_x, valid_x, test_x = load_mnist_binarized()
        train_batch_size = 32
        valid_batch_size = 50
        w, h = 28, 28
        train_x = np.append(train_x, valid_x, axis=0)
        preprocess_batch = lambda x: x
        updates_in_epoch = None
        temp = [1.]
        train_x = np.reshape(train_x, (-1, h, w, 1))
        test_x = np.reshape(test_x, (-1, h, w, 1))
        eval = DeepVAEEvaluator(test_x, n_images=10, eval_every=1, preprocess_batch=preprocess_batch)
        fn_eval = [eval.deep_vae_generate_evaluator]

    elif "cifar" in dataset_arg:
        print("Training a CIFAR10 model...")
        deep_vae = get_deep_vae_cifar("cifar")
        train_x, test_x = load_cifar(levels=256)
        train_x = train_x.reshape((-1, 32, 32, 3))
        test_x = test_x.reshape((-1, 32, 32, 3))
        train_batch_size = 48
        valid_batch_size = 50
        w, h = 32, 32
        preprocess_batch = lambda x: x
        updates_in_epoch = None
        temp = [1.]
        eval = DeepVAEEvaluator(test_x, iw_samples=1, n_images=5, eval_every=1)
        fn_eval = [eval.deep_vae_generate_evaluator]

    assert deep_vae is not None, "Please enter the name of the experiment as an argument."

    deep_vae.train_multi_gpu(train_x, test_x, n_epochs=10000, temperatures=temp, eq=1, iw=1,
                             train_batch_size=train_batch_size,
                             valid_batch_size=valid_batch_size, preprocess_batch=preprocess_batch,
                             optimizer=AdamaxOptimizer,
                             optimizer_args=(0.9, 0.999,), init_learning_rate=0.002, final_learning_rate=0.0001,
                             fn_decay=lambda x: x * 0.999999,
                             gradient_clipping=lambda x: x, fn_epoch=fn_eval,
                             updates_in_epoch=updates_in_epoch, debug_every=100)
