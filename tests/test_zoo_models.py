import numpy as np
from numpy.testing import assert_allclose

import gpflow
from gpflow.kernels import Matern52
from gpflow.likelihoods import Gaussian
from gpflow.mean_functions import Identity, Zero
from gpflow.models import GPR
from gpflow.optimizers import NaturalGradient

from doubly_stochastic_dgp.dgp import DGP
from doubly_stochastic_dgp.layers import GPMC_Layer, GPR_Layer
from doubly_stochastic_dgp.model_zoo import DGP_Heinonen


gpflow.config.set_default_float(np.float64)
gpflow.config.set_default_jitter(1e-12)


class TestHeinonen:
    def setup_method(self):
        rng = np.random.default_rng(0)
        self.N, self.D_X, self.D_Y = 6, 3, 1
        self.X = rng.uniform(size=(self.N, self.D_X))
        self.Xs = self.X.copy()
        self.Y = rng.normal(size=(self.N, self.D_Y))
        self.Ys = rng.normal(size=(self.N, self.D_Y))

    def test_matches_single_layer_gpr_when_inner_layer_is_identity(self):
        lik_var = 0.01
        kernel = Matern52(lengthscales=0.5)
        mean_function = Zero()

        exact = GPR(data=(self.X, self.Y), kernel=kernel, mean_function=mean_function)
        exact.likelihood.variance.assign(lik_var)

        likelihood = Gaussian()
        likelihood.variance.assign(lik_var)
        layer0 = GPMC_Layer(Matern52(lengthscales=0.5, variance=1e-1), self.X.copy(), self.D_X, Identity())
        layer1 = GPR_Layer(Matern52(lengthscales=0.5), mean_function, self.D_Y)
        model = DGP_Heinonen(self.X, self.Y, likelihood, [layer0, layer1])

        mean_model, var_model = model.predict_y(self.Xs, 1)
        mean_exact, var_exact = exact.predict_y(self.Xs)
        assert_allclose(mean_model[0].numpy(), mean_exact.numpy(), atol=1e-4, rtol=1e-4)
        assert_allclose(var_model[0].numpy(), var_exact.numpy(), atol=1e-4, rtol=1e-4)

        logp_model = model.predict_density(self.Xs, self.Ys, 1)
        logp_exact = exact.predict_log_density((self.Xs, self.Ys))
        assert_allclose(logp_model.numpy(), logp_exact.numpy(), atol=1e-4, rtol=1e-4)

    def test_matches_dgp_after_natgrad_update(self):
        rng = np.random.default_rng(1)
        lik_var = 0.1
        q_mu = rng.normal(size=(self.N, self.D_X))
        mean_function = Zero()

        likelihood_dgp = Gaussian()
        likelihood_dgp.variance.assign(lik_var)
        kernels = [Matern52(lengthscales=0.5), Matern52(lengthscales=0.5)]
        dgp = DGP(self.X, self.Y, self.X, kernels, likelihood_dgp, mean_function=mean_function, white=True)
        dgp.layers[0].q_mu.assign(q_mu)
        dgp.layers[0].q_sqrt.assign(dgp.layers[0].q_sqrt * 1e-24)

        _, means, _ = dgp.predict_all_layers(self.Xs, 1)
        Z = self.X.copy()
        Z[: len(self.Xs)] = means[0][0].numpy()
        dgp.layers[1].feature.Z.assign(Z)

        NaturalGradient(gamma=1.0).minimize(
            dgp.training_loss,
            var_list=[(dgp.layers[1].q_mu, dgp.layers[1].q_sqrt)],
        )

        likelihood_heinonen = Gaussian()
        likelihood_heinonen.variance.assign(lik_var)
        heinonen = DGP_Heinonen(
            self.X,
            self.Y,
            likelihood_heinonen,
            [
                GPMC_Layer(Matern52(lengthscales=0.5), self.X.copy(), self.D_X, Identity()),
                GPR_Layer(Matern52(lengthscales=0.5), mean_function, self.D_Y),
            ],
        )
        heinonen.layers[0].q_mu.assign(q_mu)

        mean_dgp, var_dgp = dgp.predict_y(self.Xs, 1)
        mean_heinonen, var_heinonen = heinonen.predict_y(self.Xs, 1)
        assert_allclose(mean_dgp.numpy(), mean_heinonen.numpy(), atol=1e-4, rtol=1e-4)
        assert_allclose(var_dgp.numpy(), var_heinonen.numpy(), atol=1e-4, rtol=1e-4)
