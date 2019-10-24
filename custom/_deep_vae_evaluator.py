import matplotlib

matplotlib.use('Agg')
import os
from utils import check_dir
import numpy as np
import scipy
import matplotlib.pyplot as plt
from time import time
from models import BIVA


class DeepVAEEvaluator(object):
    def __init__(self, images, n_images=5, iw_samples=1000, eval_every=1, preprocess_batch=lambda x: x, seed=1234):

        self.number, self.height, self.width, self.channels = images.shape

        self.rng = np.random.RandomState(seed)
        self.images = images
        self.n_images = n_images ** 2
        self.iw_samples = iw_samples
        self.eval_every = eval_every
        self.preprocess_batch = preprocess_batch

    def deep_vae_iw5000(self, model, epoch):
        assert isinstance(model, BIVA), "The model is not an instance of Deep VAE."

        if not epoch % self.eval_every == 0: return

        now = time()
        batch_size = 1
        validation_losses = []
        for update in range(int(self.number / batch_size)):

            batch_validation = self.preprocess_batch(self.images[update * batch_size: (update + 1) * batch_size])

            batch_losses = model.session.run(model.op_loss,
                                             feed_dict={model.ph_input: batch_validation, model.ph_temp: 1.,
                                                        model.ph_eq: 1, model.ph_iw: self.iw_samples,
                                                        model.ph_is_training: False})

            if np.isnan(np.mean(batch_losses)):
                model.logger.info("bad sample")
                continue
            validation_losses += [np.mean(batch_losses)]

            if update % 100 == 0:
                print(
                    "updates: {}/{}, elapsed time: {:.2f}, elbo: {:.3f}".format(update * batch_size, self.number,
                                                                                time() - now,
                                                                                np.nanmean(validation_losses)))
        model.logger.info("\nepoch: {}, iw: {}, elapsed time: {:.2f}, elbo: {:.3f}\n".format(epoch, self.iw_samples,
                                                                                             time() - now,
                                                                                             np.nanmean(
                                                                                                 validation_losses)))

    def deep_vae_generate_evaluator(self, model, epoch):
        assert isinstance(model, BIVA), "The model is not an instance of Deep VAE."

        if not epoch % self.eval_every == 0: return

        # Compute the shape of the highest latent variable.
        z_top_shp = model.session.run(model.q_layers_spatial[-1][0], feed_dict={model.ph_input: self.images[:1],
                                                                                model.ph_eq: 1, model.ph_iw: 1,
                                                                                model.ph_is_training: False}).shape

        # Sampled reconstruction from z_L
        z = self.rng.normal(0, 1, [self.n_images] + list(z_top_shp)[1:])

        px_z = model.session.run(model.op_generate,
                                 feed_dict={model.q_layers_spatial[-1][0]: z, model.ph_is_training: False})

        out_dir = check_dir(os.path.join(model.model_path, "generations"))
        path = os.path.join(out_dir, 'epoch_{}_out.png'.format(epoch))

        if self.channels > 1:
            save_images(px_z * 255, int(np.sqrt(self.n_images)), self.channels, self.height, self.width, path)
        else:
            save_gray_scale(px_z, int(np.sqrt(self.n_images)), self.height, self.width, path)


def save_gray_scale(images, count, height, width, path):
    plt.figure()
    i = 0
    img_out = np.zeros((height * count, width * count))
    for x in range(count):
        for y in range(count):
            xa, xb = x * width, (x + 1) * width
            ya, yb = y * height, (y + 1) * height
            im = np.reshape(images[i], (height, width))
            img_out[ya:yb, xa:xb] = im
            i += 1
    plt.matshow(img_out, cmap="gray")
    plt.xticks(np.array([]))
    plt.yticks(np.array([]))
    plt.savefig(path)


def save_images(images, count, channels, height, width, path):
    images = images.reshape((count, count, height, width, channels))
    images = images.transpose(1, 2, 0, 3, 4)
    images = images.reshape((height * count, width * count, channels))
    scipy.misc.toimage(images, cmin=0.0, cmax=255.0).save(path)
