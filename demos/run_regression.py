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

import sys, os
import numpy as np
import tensorflow as tf
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from scipy.cluster.vq import kmeans2
from scipy.stats import norm
from scipy.special import logsumexp

from gpflow.likelihoods import Gaussian
from gpflow.kernels import SquaredExponential, White
import gpflow

HERE = os.path.dirname(__file__)
if HERE not in sys.path:
    sys.path.insert(0, HERE)

from doubly_stochastic_dgp.dgp import DGP
from datasets import Datasets


def main():
    dataset_name = str(sys.argv[1])
    L = int(sys.argv[2])
    split = int(sys.argv[3])

    # Config
    iterations = 10000
    log_every = 100
    minibatch_size = 10000

    # Use float64 for GPflow numerical stability
    gpflow.config.set_default_float(np.float64)

    datasets = Datasets()
    data = datasets.all_datasets[dataset_name].get_data(split=split)
    X, Y, Xs, Ys, Y_std = [data[_] for _ in ['X', 'Y', 'Xs', 'Ys', 'Y_std']]

    print('############################ {} L={} split={}'.format(dataset_name, L, split))
    print('N: {}, D: {}, Ns: {}'.format(X.shape[0], X.shape[1], Xs.shape[0]))

    # Inducing points via k-means
    Z = kmeans2(X, 100, minit='points')[0]

    # Kernels per layer
    kernels = []
    for _ in range(L):
        k = SquaredExponential()
        # Provide expected attribute for downstream init code
        k.input_dim = X.shape[1]
        kernels.append(k)
    for i in range(max(0, L - 1)):
        kernels[i] = kernels[i] + White(variance=2e-6)

    mb = minibatch_size if X.shape[0] > minibatch_size else None
    model = DGP(X, Y, Z, kernels, Gaussian(), num_samples=1, minibatch_size=mb)

    # Start the inner layers almost deterministically
    for layer in model.layers[:-1]:
        layer.q_sqrt.assign(layer.q_sqrt * 1e-5)
    # Set observation noise
    model.likelihood.likelihood.variance.assign(0.05)

    optimizer = tf.optimizers.Adam(0.01)

    @tf.function
    def train_step():
        with tf.GradientTape() as tape:
            # We maximize the ELBO, so minimize its negative
            loss = -model.maximum_log_likelihood_objective()
        grads = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))
        return -loss

    def evaluate(S=100, batch_size=1000):
        means, vars_ = [], []
        num_batches = -(-len(Xs) // batch_size)
        for mb_idx in range(num_batches):
            start = mb_idx * batch_size
            end = (mb_idx + 1) * batch_size
            m, v = model.predict_y(Xs[start:end, :], S)
            means.append(m.numpy())
            vars_.append(v.numpy())
        mean_SND = np.concatenate(means, axis=1)
        var_SND = np.concatenate(vars_, axis=1)
        mean_ND = np.average(mean_SND, axis=0)
        rmse = np.average(Y_std * np.mean((Ys - mean_ND) ** 2.0) ** 0.5)
        test_nll_ND = logsumexp(
            norm.logpdf(Ys * Y_std, mean_SND * Y_std, np.sqrt(var_SND) * Y_std),
            axis=0,
            b=1 / float(S),
        )
        nll = np.average(test_nll_ND)
        return rmse, nll

    # Training loop
    for step in range(1, iterations + 1):
        elbo = train_step().numpy()
        if step % log_every == 0:
            rmse, nll = evaluate()
            print(f"iter {step:6d} | ELBO: {elbo:.3f} | test_rmse: {rmse:.4f} | test_nlpp: {nll:.4f}")


if __name__ == "__main__":
    main()