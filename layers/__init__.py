from layers._vae import (stochastic_gaussian, gaussian_merge, extend_sample_dimension)
from layers._loss import (bernoulli_log_likelihood,
                          categorical_log_likelihood,
                          gaussian_log_likelihood,
                          kl_divergences,
                          variational_inference,
                          discretized_mix_logistic_loss)

from layers._neural import (conv2d, deconv2d, dense, nin,
                            gated_resnet, transposed_gated_resnet,
                            sample_from_discretized_mix_logistic)

