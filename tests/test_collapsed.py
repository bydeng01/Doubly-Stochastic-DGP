import numpy as np
import pytest
from numpy.testing import assert_allclose

import gpflow
from gpflow.kernels import RBF
from gpflow.likelihoods import Gaussian
from gpflow.models import GPR
from gpflow.optimizers import NaturalGradient

from doubly_stochastic_dgp.dgp import DGP_Quad
from doubly_stochastic_dgp.layer_initializations import init_layers_linear
from doubly_stochastic_dgp.layers import SGPR_Layer
from doubly_stochastic_dgp.model_zoo import DGP_Collapsed


gpflow.config.set_default_float(np.float64)
gpflow.config.set_default_jitter(1e-12)


def test_single_collapsed_layer_matches_gpr():
    rng = np.random.default_rng(100)
    Ns, N, D_X, D_Y = 5, 4, 3, 1
    X = rng.uniform(size=(N, D_X))
    Y = rng.uniform(size=(N, D_Y))
    Xs = rng.uniform(size=(Ns, D_X))
    lik_var = 0.1

    layers = init_layers_linear(X, Y, X, [RBF(lengthscales=0.1)])
    last_layer = SGPR_Layer(
        layers[-1].kern,
        layers[-1].feature.Z.numpy(),
        D_Y,
        layers[-1].mean_function,
    )
    model = DGP_Collapsed(X, Y, Gaussian(), layers[:-1] + [last_layer])
    model.likelihood.likelihood.variance.assign(lik_var)

    exact = GPR(data=(X, Y), kernel=RBF(lengthscales=0.1))
    exact.likelihood.variance.assign(lik_var)

    assert_allclose(
        model.maximum_log_likelihood_objective().numpy(),
        exact.log_marginal_likelihood().numpy(),
        atol=1e-5,
        rtol=1e-5,
    )

    mean_model, var_model = model.predict_f_full_cov(Xs, 1)
    mean_exact, var_exact = exact.predict_f(Xs, full_cov=True)
    assert_allclose(mean_model[0].numpy(), mean_exact.numpy(), atol=1e-5, rtol=1e-5)
    assert_allclose(
        np.squeeze(var_model[0].numpy()),
        np.squeeze(var_exact.numpy()),
        atol=1e-5,
        rtol=1e-5,
    )


def test_collapsed_layer_matches_natgrad_after_one_step():
    rng = np.random.default_rng(101)
    N, M, D_X, D_Y = 1, 50, 1, 1
    X = rng.uniform(size=(N, D_X))
    Y = rng.uniform(size=(N, D_Y))
    Z = rng.uniform(size=(M, D_Y))
    Z[:N, :] = X[:M, :]
    lik_var = 0.1

    def kernels():
        return [RBF(lengthscales=0.1), RBF(lengthscales=0.5)]

    layers_col = init_layers_linear(X, Y, Z, kernels())
    layers_ng = init_layers_linear(X, Y, Z, kernels())

    last_layer = SGPR_Layer(
        layers_col[-1].kern,
        layers_col[-1].feature.Z.numpy(),
        D_Y,
        layers_col[-1].mean_function,
    )
    model_col = DGP_Collapsed(X, Y, Gaussian(), layers_col[:-1] + [last_layer])
    model_ng = DGP_Quad(X, Y, Gaussian(), layers_ng, H=50)
    model_col.likelihood.likelihood.variance.assign(lik_var)
    model_ng.likelihood.likelihood.variance.assign(lik_var)

    q_mu1 = rng.normal(size=(M, D_X))
    q_sqrt1 = np.tril(rng.normal(size=(M, M)))[None, :, :]
    for model in [model_col, model_ng]:
        model.layers[0].q_mu.assign(q_mu1)
        model.layers[0].q_sqrt.assign(q_sqrt1)

    NaturalGradient(gamma=1.0).minimize(
        model_ng.training_loss,
        var_list=[(model_ng.layers[-1].q_mu, model_ng.layers[-1].q_sqrt)],
    )

    assert_allclose(
        model_col.maximum_log_likelihood_objective().numpy(),
        model_ng.maximum_log_likelihood_objective().numpy(),
        atol=1e-5,
        rtol=1e-5,
    )


def test_collapsed_model_rejects_minibatches():
    rng = np.random.default_rng(102)
    X = rng.uniform(size=(4, 1))
    Y = rng.uniform(size=(4, 1))
    layers = init_layers_linear(X, Y, X, [RBF(lengthscales=0.1)])
    last_layer = SGPR_Layer(
        layers[-1].kern,
        layers[-1].feature.Z.numpy(),
        Y.shape[1],
        layers[-1].mean_function,
    )

    with pytest.raises(ValueError, match="does not support minibatches"):
        DGP_Collapsed(X, Y, Gaussian(), layers[:-1] + [last_layer], minibatch_size=2)
