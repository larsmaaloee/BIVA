from abc import ABCMeta, abstractmethod
from time import time
import numpy as np
import tensorflow as tf
from utils import get_model_path, init_logging
from os.path import join


class _BaseVAE(metaclass=ABCMeta):
    def __init__(self, session, input_shape, deterministic_layers, stochastic_layers, activation, output_distribution,
                 model_name="", dropout_inference=0., dropout_generative=0., eps=.0, is_log_var=True, minimum_kl=0.,
                 ema_decay=0.9995):
        self.session = session

        self.input_shape = input_shape
        self.deterministic_layers = deterministic_layers
        self.stochastic_layers = stochastic_layers
        self.activation = activation
        self.dropout_inference = dropout_inference
        self.dropout_generative = dropout_generative
        self.eps = eps
        self.is_log_var = is_log_var

        if ema_decay:
            self.ema = tf.train.ExponentialMovingAverage(decay=ema_decay)
        else:
            self.ema = None
        self.ema_decay = ema_decay

        self.minimum_kl = minimum_kl

        self.distributions = ["bernoulli", "categorical", "discretized", "gaussian"]
        assert output_distribution in self.distributions

        self.output_distribution = output_distribution

        assert len(self.deterministic_layers) == len(self.stochastic_layers)

        self.ph_iw = tf.placeholder(tf.int32, name="iw_samples")
        self.ph_eq = tf.placeholder(tf.int32, name="eq_samples")
        self.ph_temp = tf.placeholder(tf.float32, name="temperature")
        self.ph_input = tf.placeholder(tf.float32, shape=[None] + input_shape, name="inputs")
        self.ph_is_training = tf.placeholder(tf.bool, name="is_training")
        self.ph_lr = tf.placeholder(tf.float32)
        self.init = False

        self.trainable_variables = []
        self.global_variables = []

        self.op_output = None
        self.op_kls = None
        self.op_nll = None
        self.op_loss = None
        self.op_optimizer = None
        self.maintain_averages_op = None

        self.q_layers = []
        self.p_layers = []
        self.q_layers_spatial = []
        self.p_layers_spatial = []

        self.model_path = get_model_path("%s_%s" % (self.__class__.__name__, model_name))
        self.logger = init_logging(join(self.model_path, "out.log"))
        self.saver = None

    def load(self, path):
        self.logger.info("Loading model: %s..." % path)
        if self.saver is None:
            self._define_model()
            self.saver = tf.train.Saver(self.global_variables)
        self.saver.restore(self.session, join(path, "model"))

    def save(self, session):
        if self.saver is None:
            self.saver = tf.train.Saver(self.global_variables)
        self.saver.save(session, join(self.model_path, "model"))

    @abstractmethod
    def _define_model(self):
        raise NotImplementedError

    def train_multi_gpu(self, input_train, input_validation, n_epochs, temperatures, eq, iw, train_batch_size,
                        valid_batch_size, preprocess_batch, optimizer, optimizer_args, init_learning_rate,
                        final_learning_rate, fn_decay=lambda x: x, gradient_clipping=lambda x: x,
                        fn_epoch=None, updates_in_epoch=None, debug_every=100):

        from tensorflow.python.client import device_lib

        def get_available_gpus():
            local_device_protos = device_lib.list_local_devices()
            return [x.name for x in local_device_protos if x.device_type == 'GPU']

        gpus = get_available_gpus()
        no_gpus = len(gpus)
        if no_gpus == 0:
            gpus += ["/cpu:0"]
            no_gpus = 1

        updates_in_epoch = int(
            len(input_train) / train_batch_size) if updates_in_epoch is None else updates_in_epoch


        tower_losses = []
        tower_kls = []
        tower_nlls = []
        tower_grads = []
        tower_vars = []
        tower_inits = []
        tower_generate = []
        tower_inputs = []
        for i, gpu in enumerate(gpus):
            self.logger.info("Deploying model {}/{} to device {}...".format(i + 1, no_gpus, gpu))
            with tf.device(gpu):
                self.q_layers_spatial = []
                self.p_layers_spatial = []

                self.ph_input = tf.placeholder(tf.float32, shape=[None] + self.input_shape, name="inputs_{}".format(i))

                self._define_model()

                grad = tf.gradients(tf.reduce_mean(self.op_loss), self.trainable_variables, colocate_gradients_with_ops=True)
                tower_inputs += [self.ph_input]
                tower_grads += [grad]
                tower_vars += [self.trainable_variables]
                tower_losses += [[self.op_loss]]
                tower_kls += [self.op_kls]
                tower_nlls += [self.op_nll]
                tower_inits += [self.op_data_dependent_initialization]
                tower_generate += [self.op_generate]

        with tf.variable_scope("optimizer"):
            self.op_optimizer = optimizer(self.ph_lr, *optimizer_args)
            global_step = tf.get_variable('global_step', [], initializer=tf.constant_initializer(0), trainable=False)

            for i, grads in enumerate(tower_grads[:-1]):
                for j, grad in enumerate(grads):
                    tower_grads[-1][j] += grad

            gradients = tower_grads[-1]
            gradients_variables = gradient_clipping([(g / no_gpus, v) for g, v in zip(gradients, tower_vars[-1])])

            self.op_optimizer = self.op_optimizer.apply_gradients(gradients_variables, global_step=global_step)

            if self.maintain_averages_op is not None:
                self.op_optimizer = tf.group(self.op_optimizer, self.maintain_averages_op)

            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)

            mean_losses = tower_losses
            mean_nlls = tower_nlls
            mean_kls = tf.reduce_mean(tower_kls, axis=0)

        if self.saver is None:
            self.session.run(tf.variables_initializer(self.global_variables))
        self.session.run(tf.variables_initializer(tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope="optimizer")))

        total_parameters = 0
        for var in self.trainable_variables:
            shape = var.get_shape()
            variable_parameters = 1
            for dim in shape:
                variable_parameters *= dim.value
            total_parameters += variable_parameters

        self.logger.info(
            "Train set size: {}, Validation set size: {}.".format(len(input_train), len(input_validation)))
        self.logger.info(
            "Train batch size: {}, Validation batch size size: {}.".format(train_batch_size, valid_batch_size))
        self.logger.info(
            "Total number of variables in scope {}: {}.".format(self.__class__.__name__, total_parameters))
        self.logger.info("Optimizer: {}, Args: {}.".format(optimizer().__class__.__name__, str(optimizer_args)))
        self.logger.info("Temperature: {}.".format(str(np.array(temperatures))))
        self.logger.info("IW: {}, EQ: {}.".format(iw, eq))
        self.logger.info("Dropout inference: {}.".format(self.dropout_inference))
        self.logger.info("Dropout generative: {}.".format(self.dropout_generative))
        self.logger.info("log(var): {}.".format(self.is_log_var))
        self.logger.info("epsilon: {}.".format(self.eps))
        self.logger.info("minimum KL: {}.".format(self.minimum_kl))
        if self.ema:
            self.logger.info("exponential moving averages with decay: {}.".format(self.ema_decay))

        self.logger.info("Printing variables...")
        self.logger.info("####################")
        tensor_vars = tf.trainable_variables()
        for var in tensor_vars:
            self.logger.info("{}, {}".format(var.name, var.shape))

        current_best = np.inf
        epoch_at_current_best = -1
        self.logger.info("####################")

        # Data dependent initialization
        init_indices = np.random.choice(len(input_train), train_batch_size * no_gpus, replace=False)
        batch_init = preprocess_batch(input_train[init_indices])
        feed_dict = {}
        for i in range(no_gpus):
            feed_dict[self.ph_eq] = 1
            feed_dict[self.ph_iw] = 1
            feed_dict[self.ph_is_training] = True
            feed_dict[self.ph_temp] = 1.
            feed_dict[tower_inputs[i]] = batch_init

        self.session.run(tower_generate, feed_dict=feed_dict)

        if self.op_data_dependent_initialization is not None and self.saver is None:
            self.session.run(tower_inits, feed_dict=feed_dict)


        # Training loop
        learning_rate = init_learning_rate
        for epoch in range(n_epochs):
            train_losses = []
            train_nlls = []
            train_kls = []
            validation_losses = []
            validation_nlls = []
            validation_kls = []

            batch_time = []

            temperature = temperatures[epoch] if epoch < len(temperatures) else temperature

            if temperature != 1.:
                train_eq, train_iw = 1, 1
            else:
                train_eq, train_iw = eq, iw

            if fn_epoch is not None:
                if isinstance(fn_epoch, list):
                    for f in fn_epoch:
                        f(**{"model": self, "epoch": epoch})
                else:
                    fn_epoch(**{"model": self, "epoch": epoch})


            feed_dict = {}
            feed_dict[self.ph_eq] = train_eq
            feed_dict[self.ph_iw] = train_iw
            feed_dict[self.ph_is_training] = True
            feed_dict[self.ph_temp] = temperature

            for update in range(0, updates_in_epoch, no_gpus):
                feed_dict[self.ph_lr] = learning_rate

                now = time()

                train_indices = np.random.choice(len(input_train), train_batch_size*no_gpus, replace=False)
                batch_train = preprocess_batch(input_train[train_indices])

                for i in range(no_gpus):

                    mini_batch_train = batch_train[i*train_batch_size:(i+1)*train_batch_size]
                    feed_dict[tower_inputs[i]] = mini_batch_train

                batch_losses, batch_nll, kls, gs, _ = self.session.run([mean_losses, mean_nlls, mean_kls, global_step, (self.op_optimizer, update_ops)], feed_dict=feed_dict)


                train_losses += [batch_losses]
                train_nlls += [batch_nll]
                train_kls += [kls]

                batch_time += [time() - now]

                if update % debug_every == 0:
                    output_str = "epoch={}, update={}/{}, batch_time={:.2f}, ".format(epoch, update, int(updates_in_epoch), np.mean(batch_time))
                    output_str += " ".join(["loss-{}={:.2f}".format(i + 1, np.mean(l)) for i, l in enumerate(np.mean(train_losses, axis=0))])
                    output_str += " "
                    output_str += " ".join(["nll-{}={:.2f}".format(i + 1, np.mean(l)) for i, l in enumerate(np.mean(train_nlls, axis=0))])
                    output_str += " "
                    output_str += " ".join(["kl-{}={:.2f}".format(i + 1, np.mean(kl)) for i, kl in enumerate(np.mean(train_kls, axis=0))])
                    print(output_str)


                learning_rate = fn_decay(learning_rate)
                if learning_rate < final_learning_rate:
                    learning_rate = final_learning_rate

            feed_dict = {}
            feed_dict[self.ph_eq] = train_eq
            feed_dict[self.ph_iw] = train_iw
            feed_dict[self.ph_is_training] = False
            feed_dict[self.ph_temp] = 1.

            for update in range(0, int(len(input_validation) / valid_batch_size), no_gpus):
                for i in range(no_gpus):

                    if ((update+i)+1) * valid_batch_size < len(input_validation):
                        batch_validation = input_validation[(update+i) * valid_batch_size: ((update+i) + 1) * valid_batch_size]
                        batch_validation = preprocess_batch(batch_validation)

                    feed_dict[tower_inputs[i]] = batch_validation

                batch_losses, batch_nll, kls = self.session.run([mean_losses, mean_nlls, mean_kls], feed_dict=feed_dict)

                validation_losses += [batch_losses]
                validation_nlls += [batch_nll]
                validation_kls += [kls]

            mean_train_loss = np.mean(train_losses)
            mean_test_loss = np.mean(validation_losses)


            output_str = ""
            if not updates_in_epoch == 0:
                output_str += "epoch={}, updates={}, lr={:.6f}, temperature={:.2f}, epoch_time={:.2f}, batch_time={:.2f}, best={:.2f} (epoch={})\n".format(
                    epoch, int(gs), learning_rate, temperature, np.sum(batch_time), np.mean(batch_time), current_best, epoch_at_current_best)

                output_str += "\t train: eq={}, iw={}, loss={:.2f}, ".format(train_eq, train_iw, mean_train_loss)

                output_str += " ".join(
                    ["loss-{}={:.2f}".format(i + 1, np.mean(l)) for i, l in enumerate(np.mean(train_losses, axis=0))])

                output_str += " "

                output_str += " ".join(
                    ["nll-{}={:.2f}".format(i + 1, np.mean(l)) for i, l in enumerate(np.mean(train_nlls, axis=0))])

                output_str += " "

                output_str += " ".join(
                    ["kl-{}={:.2f}".format(i + 1, np.mean(kl)) for i, kl in enumerate(np.mean(train_kls, axis=0))])


            output_str += "\n\t valid: eq={}, iw={}, loss={:.2f}, ".format(eq, iw, mean_test_loss)

            output_str += " ".join(
                ["loss-{}={:.2f}".format(i + 1, np.mean(l)) for i, l in enumerate(np.mean(validation_losses, axis=0))])

            output_str += " "

            output_str += " ".join(
                ["nll-{}={:.2f}".format(i + 1, np.mean(l)) for i, l in enumerate(np.mean(validation_nlls, axis=0))])

            output_str += " "

            output_str += " ".join(
                ["kl-{}={:.2f}".format(i + 1, np.mean(kl)) for i, kl in enumerate(np.mean(validation_kls, axis=0))])

            self.logger.info(output_str + "\n")

            if mean_test_loss < current_best:
                self.logger.info("Saving model...")
                current_best = mean_test_loss
                epoch_at_current_best = epoch
                self.save(self.session)


def average_gradients(tower_grads):
  """Calculate the average gradient for each shared variable across all towers.
  Note that this function provides a synchronization point across all towers.
  Args:
    tower_grads: List of lists of (gradient, variable) tuples. The outer list
      is over individual gradients. The inner list is over the gradient
      calculation for each tower.
  Returns:
     List of pairs of (gradient, variable) where the gradient has been averaged
     across all towers.
  """
  average_grads = []
  for grad_and_vars in zip(*tower_grads):
    # Note that each grad_and_vars looks like the following:
    #   ((grad0_gpu0, var0_gpu0), ... , (grad0_gpuN, var0_gpuN))
    grads = []
    for g, _ in grad_and_vars:
      # Add 0 dimension to the gradients to represent the tower.
      expanded_g = tf.expand_dims(g, 0)

      # Append on a 'tower' dimension which we will average over below.
      grads.append(expanded_g)

    # Average over the 'tower' dimension.
    grad = tf.concat(axis=0, values=grads)
    grad = tf.reduce_mean(grad, 0)

    # Keep in mind that the Variables are redundant because they are shared
    # across towers. So .. we will just return the first tower's pointer to
    # the Variable.
    v = grad_and_vars[0][1]
    grad_and_var = (grad, v)
    average_grads.append(grad_and_var)
  return average_grads