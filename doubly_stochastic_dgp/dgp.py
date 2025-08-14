# Copyright 2025 Boyuan Deng
#
# Copyright 2017 Hugh Salimbeni
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import tensorflow as tf
import numpy as np

from gpflow.base import Module
from gpflow.mean_functions import Identity, Linear, Zero
from gpflow.quadrature import mvhermgauss
from gpflow.likelihoods import Gaussian
from gpflow import default_float

from doubly_stochastic_dgp.utils import reparameterize

from doubly_stochastic_dgp.utils import BroadcastingLikelihood
from doubly_stochastic_dgp.layer_initializations import init_layers_linear
from doubly_stochastic_dgp.layers import GPR_Layer, SGPMC_Layer, GPMC_Layer, SVGP_Layer


class DGP_Base(Module):
    """
    The base class for Deep Gaussian process models.

    Implements a Monte-Carlo variational bound and convenience functions.

    """
    def __init__(self, X, Y, likelihood, layers,
                 minibatch_size=None,
                 num_samples=1, num_data=None,
                 **kwargs):
        Module.__init__(self, **kwargs)
        self.num_samples = num_samples
        self.num_data = num_data or X.shape[0]

        # --- CHANGE 1: minibatching setup ---
        X_full = tf.convert_to_tensor(X, dtype=default_float())
        Y_full = tf.convert_to_tensor(Y, dtype=default_float())
        self.X = X_full     # keep full dataset tensors for prediction etc.
        self.Y = Y_full

        if minibatch_size is not None:
            ds = tf.data.Dataset.from_tensor_slices((X_full, Y_full))
            # buffer_size = dataset size; seeded shuffle like original Minibatch(...)
            ds = ds.shuffle(buffer_size=tf.shape(X_full)[0], seed=0, reshuffle_each_iteration=True)
            ds = ds.repeat().batch(minibatch_size, drop_remainder=False)
            self._mb_iter = iter(ds)
        else:
            self._mb_iter = None
        # --- END CHANGE 1 ---

        self.likelihood = BroadcastingLikelihood(likelihood)
        self.layers = layers

    def propagate(self, X, full_cov=False, S=1, zs=None):
        sX = tf.tile(tf.expand_dims(X, 0), [S, 1, 1])

        Fs, Fmeans, Fvars = [], [], []

        F = sX
        zs = zs or [None, ] * len(self.layers)
        for layer, z in zip(self.layers, zs):
            F, Fmean, Fvar = layer.sample_from_conditional(F, z=z, full_cov=full_cov)

            Fs.append(F)
            Fmeans.append(Fmean)
            Fvars.append(Fvar)

        return Fs, Fmeans, Fvars

    def _build_predict(self, X, full_cov=False, S=1):
        Fs, Fmeans, Fvars = self.propagate(X, full_cov=full_cov, S=S)
        return Fmeans[-1], Fvars[-1]

    def E_log_p_Y(self, X, Y):
        """
        Calculate the expectation of the data log likelihood under the variational distribution
         with MC samples
        """
        Fmean, Fvar = self._build_predict(X, full_cov=False, S=self.num_samples)
        var_exp = self.likelihood.variational_expectations(Fmean, Fvar, Y)  # S, N, D
        return tf.reduce_mean(var_exp, 0)  # N, D

    # helper to fetch current batch
    def _get_batch(self):
        if self._mb_iter is not None:
            return next(self._mb_iter)
        return self.X, self.Y
    
    def maximum_log_likelihood_objective(self):
        Xb, Yb = self._get_batch()
        L = tf.reduce_sum(self.E_log_p_Y(Xb, Yb))
        KL = tf.reduce_sum([layer.KL() for layer in self.layers])
        batch_size = tf.cast(tf.shape(Xb)[0], default_float())
        scale = tf.cast(self.num_data, default_float()) / batch_size
        return L * scale - KL

    def predict_f(self, Xnew, num_samples):
        return self._build_predict(Xnew, full_cov=False, S=num_samples)

    def predict_f_full_cov(self, Xnew, num_samples):
        return self._build_predict(Xnew, full_cov=True, S=num_samples)

    def predict_all_layers(self, Xnew, num_samples):
        return self.propagate(Xnew, full_cov=False, S=num_samples)

    def predict_all_layers_full_cov(self, Xnew, num_samples):
        return self.propagate(Xnew, full_cov=True, S=num_samples)

    def predict_y(self, Xnew, num_samples):
        Fmean, Fvar = self._build_predict(Xnew, full_cov=False, S=num_samples)
        return self.likelihood.predict_mean_and_var(Fmean, Fvar)

    def predict_density(self, Xnew, Ynew, num_samples):
        Fmean, Fvar = self._build_predict(Xnew, full_cov=False, S=num_samples)
        l = self.likelihood.predict_density(Fmean, Fvar, Ynew)
        log_num_samples = tf.math.log(tf.cast(num_samples, default_float()))
        return tf.math.reduce_logsumexp(l - log_num_samples, axis=0)


class DGP_Quad(DGP_Base):
    """
    A DGP with quadrature instead of MC sampling. This scales exponentially in the sum of the inner layer dims
    
    The key ref is:
    [in progress]
    
    """
    def __init__(self, *args, H=100, **kwargs):
        DGP_Base.__init__(self, *args, **kwargs)

        # set up the quadrature points
        self.H = H
        self.D_quad = sum([layer.q_mu.shape[1] for layer in self.layers[:-1]])
        gh_x, gh_w = mvhermgauss(H, self.D_quad)
        gh_x *= 2. ** 0.5  # H**quad_dims, quad_dims
        self.gh_w = gh_w * np.pi ** (-0.5 * self.D_quad)  # H**quad_dims

        # split z into each layer, to work with the loop over layers
        # the shape is S, 1, D as this will broadcast correctly with S,N,D (never used with full cov)
        s, e = 0, 0
        self.gh_x = []
        for layer in self.layers[:-1]:
            e += layer.q_mu.shape[1]
            self.gh_x.append(gh_x[:, None, s:e])
            s += layer.q_mu.shape[1]

        # finish with zeros (we don't need to do quadrature over the final layer and this will never get used
        self.gh_x.append(tf.zeros((1, 1, 1), dtype=default_float()))

    def E_log_p_Y(self, X, Y):
        """
        Calculate the expectation of the data log likelihood under the variational distribution
         with quadrature 
        """
        _, Fmeans, Fvars = self.propagate(X, zs=self.gh_x, full_cov=False, S=self.H**self.D_quad)
        var_exp = self.likelihood.variational_expectations(Fmeans[-1], Fvars[-1], Y)  # S, N, D
        return tf.reduce_sum(var_exp * self.gh_w[:, None, None], 0)  # N, D


class DGP(DGP_Base):
    """
    This is the Doubly-Stochastic Deep GP, with linear/identity mean functions at each layer.

    The key reference is

    ::
      @inproceedings{salimbeni2017doubly,
        title={Doubly Stochastic Variational Inference for Deep Gaussian Processes},
        author={Salimbeni, Hugh and Deisenroth, Marc},
        booktitle={NIPS},
        year={2017}
      }

    """
    def __init__(self, X, Y, Z, kernels, likelihood,
                 num_outputs=None,
                 mean_function=Zero(),  # the final layer mean function,
                 white=False, **kwargs):
        layers = init_layers_linear(X, Y, Z, kernels,
                                    num_outputs=num_outputs,
                                    mean_function=mean_function,
                                    white=white)
        DGP_Base.__init__(self, X, Y, likelihood, layers, **kwargs)