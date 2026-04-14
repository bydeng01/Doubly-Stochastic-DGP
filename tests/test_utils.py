import numpy as np
import tensorflow as tf
from numpy.testing import assert_allclose

import gpflow
from gpflow.likelihoods import Bernoulli, Gaussian

from doubly_stochastic_dgp.utils import BroadcastingLikelihood, reparameterize


gpflow.config.set_default_float(np.float64)


def _to_numpy(value):
    return value.numpy() if hasattr(value, "numpy") else value


def test_reparameterize_diag_matches_numpy():
    rng = np.random.default_rng(0)
    mean = rng.normal(size=(4, 3, 2))
    var = rng.normal(size=(4, 3, 2)) ** 2
    z = rng.normal(size=(4, 3, 2))

    actual = reparameterize(
        tf.constant(mean, dtype=gpflow.default_float()),
        tf.constant(var, dtype=gpflow.default_float()),
        tf.constant(z, dtype=gpflow.default_float()),
    )
    expected = mean + z * (var + gpflow.default_jitter()) ** 0.5

    assert_allclose(actual.numpy(), expected)


def test_reparameterize_full_cov_matches_numpy():
    rng = np.random.default_rng(1)
    S, N, D = 4, 3, 2
    mean = rng.normal(size=(S, N, D))
    U = rng.normal(size=(S, N, N, D))
    var = np.einsum("SnNd,SmNd->Snmd", U, U)
    z = rng.normal(size=(S, N, D))

    var_flat = np.reshape(np.transpose(var, [0, 3, 1, 2]), [S * D, N, N])
    eye = np.eye(N)[None, :, :] * gpflow.default_jitter()
    chol_flat = np.linalg.cholesky(var_flat + eye)
    chol = np.transpose(np.reshape(chol_flat, [S, D, N, N]), [0, 2, 3, 1])
    expected = mean + np.einsum("Snmd,Smd->Snd", chol, z)

    actual = reparameterize(
        tf.constant(mean, dtype=gpflow.default_float()),
        tf.constant(var, dtype=gpflow.default_float()),
        tf.constant(z, dtype=gpflow.default_float()),
        full_cov=True,
    )

    assert_allclose(actual.numpy(), expected)


def test_broadcasting_likelihood_matches_per_sample_gaussian():
    rng = np.random.default_rng(2)
    S, N, D = 5, 4, 3
    X = rng.normal(size=(N, 2))
    Fmu = rng.normal(size=(S, N, D))
    Fvar = rng.normal(size=(S, N, D)) ** 2
    Y = rng.normal(size=(N, D))
    likelihood = Gaussian()
    wrapped = BroadcastingLikelihood(likelihood)

    actual_ve = wrapped.variational_expectations(X, Fmu, Fvar, Y)
    expected_ve = tf.stack(
        [likelihood.variational_expectations(X, Fmu[s], Fvar[s], Y) for s in range(S)]
    )
    assert_allclose(actual_ve.numpy(), expected_ve.numpy())

    actual_mean, actual_var = wrapped.predict_mean_and_var(X, Fmu, Fvar)
    expected_mean, expected_var = zip(
        *[likelihood.predict_mean_and_var(X, Fmu[s], Fvar[s]) for s in range(S)]
    )
    assert_allclose(actual_mean.numpy(), tf.stack(expected_mean).numpy())
    assert_allclose(actual_var.numpy(), tf.stack(expected_var).numpy())

    actual_log_density = wrapped.predict_density(X, Fmu, Fvar, Y)
    expected_log_density = tf.stack(
        [likelihood.predict_log_density(X, Fmu[s], Fvar[s], Y) for s in range(S)]
    )
    assert_allclose(actual_log_density.numpy(), expected_log_density.numpy())


def test_broadcasting_likelihood_matches_per_sample_bernoulli():
    rng = np.random.default_rng(3)
    S, N, D = 5, 4, 2
    X = rng.normal(size=(N, 2))
    Fmu = rng.normal(size=(S, N, D))
    Fvar = rng.normal(size=(S, N, D)) ** 2
    Y = rng.choice([0.0, 1.0], size=(N, D))
    likelihood = Bernoulli()
    wrapped = BroadcastingLikelihood(likelihood)

    actual = wrapped.variational_expectations(X, Fmu, Fvar, Y)
    expected = tf.stack(
        [likelihood.variational_expectations(X, Fmu[s], Fvar[s], Y) for s in range(S)]
    )

    assert_allclose(_to_numpy(actual), _to_numpy(expected[..., None]))
