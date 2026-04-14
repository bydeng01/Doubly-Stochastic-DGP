import numpy as np
import tensorflow as tf
from numpy.testing import assert_allclose

import gpflow
from gpflow.kernels import Matern52, RBF
from gpflow.likelihoods import Gaussian
from gpflow.models import SVGP
from gpflow.optimizers import Scipy
from gpflow.utilities import set_trainable

from doubly_stochastic_dgp.dgp import DGP, DGP_Base, DGP_Quad
from doubly_stochastic_dgp.layer_initializations import init_layers_input_prop, init_layers_linear


gpflow.config.set_default_float(np.float64)
gpflow.config.set_default_jitter(1e-12)


def _as_numpy(value):
    return value.numpy() if hasattr(value, "numpy") else value


class TestVsSingleLayer:
    def setup_method(self):
        rng = np.random.default_rng(0)
        self.Ns, self.N, self.D_X, self.D_Y = 20, 19, 2, 3
        self.X = rng.uniform(size=(self.N, self.D_X))
        self.Xs = rng.uniform(size=(self.Ns, self.D_X))
        self.Y = rng.normal(size=(self.N, self.D_Y))
        self.Ys = rng.normal(size=(self.Ns, self.D_Y))
        self.q_mu = rng.normal(size=(self.N, self.D_Y))
        self.q_sqrt = 0.001 * np.tile(np.eye(self.N)[None, :, :], [self.D_Y, 1, 1])

    def test_gaussian_matches_svgp_single_layer(self):
        for white in [True, False]:
            likelihood = Gaussian()
            likelihood.variance.assign(0.01)
            svgp = SVGP(
                kernel=Matern52(lengthscales=0.5),
                likelihood=likelihood,
                inducing_variable=self.X,
                whiten=white,
                num_latent_gps=self.D_Y,
            )
            svgp.q_mu.assign(self.q_mu)
            svgp.q_sqrt.assign(self.q_sqrt)

            dgp_likelihood = Gaussian()
            dgp_likelihood.variance.assign(0.01)
            dgp = DGP(
                self.X,
                self.Y,
                self.X,
                [Matern52(lengthscales=0.5)],
                dgp_likelihood,
                white=white,
                num_samples=2,
            )
            dgp.layers[-1].q_mu.assign(self.q_mu)
            dgp.layers[-1].q_sqrt.assign(self.q_sqrt)

            assert_allclose(
                dgp.maximum_log_likelihood_objective().numpy(),
                svgp.maximum_log_likelihood_objective((self.X, self.Y)).numpy(),
                rtol=1e-6,
                atol=1e-6,
            )

            mean_svgp, var_svgp = svgp.predict_y(self.Xs)
            mean_dgp, var_dgp = dgp.predict_y(self.Xs, 1)
            assert_allclose(mean_dgp[0].numpy(), mean_svgp.numpy(), rtol=1e-6, atol=1e-6)
            assert_allclose(var_dgp[0].numpy(), var_svgp.numpy(), rtol=1e-6, atol=1e-6)

            logp_svgp = svgp.predict_log_density((self.Xs, self.Ys))
            logp_dgp = dgp.predict_density(self.Xs, self.Ys, 1)
            assert_allclose(logp_dgp.numpy(), logp_svgp.numpy(), rtol=1e-6, atol=1e-6)

    def test_training_loss_is_negative_objective(self):
        likelihood = Gaussian()
        model = DGP(
            self.X,
            self.Y,
            self.X,
            [Matern52(lengthscales=0.5)],
            likelihood,
            num_samples=1,
        )

        assert_allclose(
            model.training_loss().numpy(),
            -model.maximum_log_likelihood_objective().numpy(),
        )

    def test_full_size_minibatch_matches_full_data_objective(self):
        def make_model(minibatch_size):
            likelihood = Gaussian()
            likelihood.variance.assign(0.01)
            model = DGP(
                self.X,
                self.Y,
                self.X,
                [Matern52(lengthscales=0.5)],
                likelihood,
                minibatch_size=minibatch_size,
                num_samples=1,
            )
            model.layers[-1].q_mu.assign(self.q_mu)
            model.layers[-1].q_sqrt.assign(self.q_sqrt)
            return model

        full = make_model(None)
        minibatched = make_model(self.N)

        assert_allclose(
            minibatched.maximum_log_likelihood_objective().numpy(),
            full.maximum_log_likelihood_objective().numpy(),
            rtol=1e-6,
            atol=1e-6,
        )


class TestQuad:
    def test_quadrature_is_deterministic(self):
        rng = np.random.default_rng(1)
        N = 2
        X = rng.uniform(size=(N, 1))
        Y = np.sin(20 * X) + rng.normal(size=X.shape) * 0.001

        def layers():
            return init_layers_linear(
                X,
                Y,
                X,
                [RBF(lengthscales=0.1), RBF(lengthscales=0.1)],
            )

        likelihood = Gaussian()
        likelihood.variance.assign(0.01)
        model = DGP_Quad(X, Y, likelihood, layers(), H=20)

        first = model.maximum_log_likelihood_objective()
        second = model.maximum_log_likelihood_objective()
        assert_allclose(first.numpy(), second.numpy())

    def test_scipy_optimizer_accepts_training_loss(self):
        rng = np.random.default_rng(2)
        X = rng.uniform(size=(3, 1))
        Y = np.sin(X)
        likelihood = Gaussian()
        model = DGP_Base(
            X,
            Y,
            likelihood,
            init_layers_linear(X, Y, X, [RBF(lengthscales=0.2)]),
        )
        set_trainable(model.likelihood, False)

        Scipy().minimize(model.training_loss, model.trainable_variables, options={"maxiter": 1})


def test_step_up_model_builds_objective():
    X = np.zeros((2, 1))
    Y = np.zeros((2, 1))
    kern1 = RBF(active_dims=[0])
    kern2 = RBF(active_dims=[0, 1])
    likelihood = Gaussian()
    model = DGP(X, Y, X, [kern1, kern2], likelihood)

    objective = model.maximum_log_likelihood_objective()
    assert tf.math.is_finite(objective)


def test_input_propagation_model_builds_objective():
    X = np.zeros((3, 1))
    Y = np.zeros((3, 1))
    kernels = [RBF(active_dims=[0]), RBF(active_dims=[0, 1])]
    likelihood = Gaussian()
    layers = init_layers_input_prop(X, Y, X, kernels)
    model = DGP_Base(X, Y, likelihood, layers)

    objective = model.maximum_log_likelihood_objective()
    samples, _, _ = model.predict_all_layers(X, 1)

    assert tf.math.is_finite(objective)
    assert samples[0].shape[-1] == 2
    assert samples[-1].shape[-1] == 1
