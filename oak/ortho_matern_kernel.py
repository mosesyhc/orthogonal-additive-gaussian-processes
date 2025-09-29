# orthogonal_matern32_kernel.py

import gpflow
import numpy as np
import tensorflow as tf
from typing import Optional
from oak.input_measures import (
    Measure,
    EmpiricalMeasure,
    GaussianMeasure,
    MOGMeasure,
    UniformMeasure,
)

# ==== Helpers for Matérn-3/2 closed forms (1D) ====

@tf.function
def _phi(z):
    return tf.exp(-0.5 * tf.square(z)) / np.sqrt(2.0 * np.pi)

@tf.function
def _Phi(z):
    # Standard normal CDF via erf for stability
    return 0.5 * (1.0 + tf.math.erf(z / np.sqrt(2.0)))

def _matern32_expectation_against_gaussian(x, mu, var, ell, sigma2):
    """
    g(x) = E_{Y~N(mu, var)}[ k_m32(x, Y) ]
    where k_m32(x,y) = sigma2 * (1 + alpha |x-y|) * exp(-alpha |x-y|),
    alpha = sqrt(3)/ell.
    Shapes:
      x: [..., N, 1], mu: scalar, var: scalar
    Returns: [..., N, 1]
    """
    alpha = np.sqrt(3.0) / ell
    alpha2 = 3.0 / tf.square(ell)
    s = np.sqrt(var + 0.0)
    # μ = x - m
    mu_diff = x - mu  # shape [..., N, 1]
    # u_±
    u_minus = (-mu_diff - alpha * var) / s
    u_plus  = (mu_diff - alpha * var) / s
    A = tf.exp(0.5 * alpha2 * var)

    term_minus = tf.exp(-alpha * mu_diff) * (
        (1.0 - alpha2 * var + alpha * mu_diff) * _Phi(u_minus)
        + alpha * s * _phi(u_minus)
    )
    term_plus = tf.exp(alpha * mu_diff) * (
        (1.0 - alpha2 * var - alpha * mu_diff) * _Phi(u_plus)
        + alpha * s * _phi(u_plus)
    )
    return sigma2 * A * (term_minus + term_plus)

def _matern32_double_gaussian(var, ell, sigma2):
    """
    E_{X~N(mu1,var1), Y~N(mu2,var2)}[ k_m32(X,Y) ] (1D, closed form)
    Reduce to the single-Gaussian expectation by Z = X - Y ~ N(mu1-mu2, var1+var2).
    """
    alpha = np.sqrt(3.0) / ell
    alpha2 = 3.0 / tf.square(ell)
    s2 = 2 * var
    s = np.sqrt(s2)

    A = tf.exp(0.5 * alpha2 * s2)

    term_minus = 2 * (1.0 - alpha2 * s2) * _Phi(-alpha * s)
    term_plus = 2 * alpha * s * _phi(alpha * s)
    return sigma2 * A * (term_minus + term_plus)

def _matern32_expectation_against_uniform(x, a, b, ell, sigma2):
    """
    g(x) = E_{Y~Unif[a,b]}[ k_m32(x, Y) ], exact piecewise 1D formula.
    """
    alpha = np.sqrt(3.0) / ell
    L = b - a

    def G(T):
        # ∫_0^T (1 + α t) e^{-α t} dt = 2/α - (T + 2/α) e^{-α T}
        return (2.0 / alpha) - (T + 2.0 / alpha) * tf.exp(-alpha * T)

    # piecewise based on where x falls relative to [a,b]
    x = tf.convert_to_tensor(x)
    # Broadcast scalars
    a = tf.cast(a, x.dtype)
    b = tf.cast(b, x.dtype)

    # three cases: x <= a, a < x < b, x >= b
    case_left  = x <= a
    case_mid   = tf.logical_and(x > a, x < b)
    case_right = x >= b

    # Left: ∫_a^b f(y-x) dy = F(b-x)-F(a-x), F(t) = -t e^{-αt} - 2/α e^{-αt}
    def F(t):
        return -t * tf.exp(-alpha * t) - 2.0 / alpha * tf.exp(-alpha * t)

    g_left  = sigma2 * (F(b - x) - F(a - x))
    g_right = sigma2 * (F(x - a) - F(x - b))
    g_mid   = sigma2 * (G(x - a) + G(b - x))

    return tf.where(
        case_left, g_left,
        tf.where(case_mid, g_mid, g_right)
    )

def _matern32_double_uniform(a, b, ell, sigma2):
    """
    E_{X,Y ~ Unif[a,b]}[ k_m32(X,Y) ], exact closed form depending only on L=b-a.
    """
    alpha = np.sqrt(3.0) / ell
    L = b - a
    # 2/((L^2) α^2) * (2 L α - 3 + (L α + 3) e^{-α L})
    return sigma2 * (2.0 / (tf.square(L) * tf.square(alpha))) * (
        2.0 * L * alpha - 3.0 + (L * alpha + 3.0) * tf.exp(-alpha * L)
    )

# ==== Orthogonal kernel using Matérn–3/2 base ====

class OrthogonalMatern32Kernel(gpflow.kernels.Kernel):
    """
    Orthogonalized Matérn–3/2 kernel against a given input measure (1D).

    base_kernel: gpflow.kernels.Matern32 (1D)
    measure: one of {UniformMeasure, GaussianMeasure, EmpiricalMeasure, MOGMeasure}
    """

    def __init__(
        self, base_kernel: gpflow.kernels.Matern32, measure: Measure, active_dims=None
    ):
        super().__init__(active_dims=active_dims)
        self.base_kernel, self.measure = base_kernel, measure
        self.active_dims = self.active_dims

        if not isinstance(base_kernel, gpflow.kernels.Matern32):
            raise NotImplementedError("Base kernel must be gpflow.kernels.Matern32 (1D).")

        if not isinstance(
            measure,
            (UniformMeasure, GaussianMeasure, EmpiricalMeasure, MOGMeasure),
        ):
            raise NotImplementedError("Unsupported measure type.")

        # --- UniformMeasure: exact expectation and variance (same interval for both args) ---
        if isinstance(self.measure, UniformMeasure):
            a = tf.cast(self.measure.a, tf.float64)
            b = tf.cast(self.measure.b, tf.float64)

            def cov_X_s(X):
                # X: [N,1] (or [..., N, 1])
                tf.debugging.assert_shapes([(X, (..., "N", 1))])
                ell = tf.cast(self.base_kernel.lengthscales, X.dtype)
                sigma2 = tf.cast(self.base_kernel.variance, X.dtype)
                return _matern32_expectation_against_uniform(X, a, b, ell, sigma2)

            def var_s():
                ell = tf.cast(self.base_kernel.lengthscales, tf.float64)
                sigma2 = tf.cast(self.base_kernel.variance, tf.float64)
                return _matern32_double_uniform(a, b, ell, sigma2)

        # --- GaussianMeasure: exact expectation and variance (centered double integral) ---
        if isinstance(self.measure, GaussianMeasure):
            def cov_X_s(X):
                tf.debugging.assert_shapes([(X, (..., "N", 1))])

                ell = self.base_kernel.lengthscales
                sigma2 = self.base_kernel.variance
                return _matern32_expectation_against_gaussian(X, self.measure.mu, self.measure.var, ell, sigma2)

            def var_s():
                # E[k(X,Y)] with X,Y ~ N(mu, var) independent -> μ=0, s^2=2*var
                ell = self.base_kernel.lengthscales
                sigma2 = self.base_kernel.variance
                return _matern32_double_gaussian(var=self.measure.var, ell=ell, sigma2=sigma2)

        # --- EmpiricalMeasure: kernel-agnostic sums (same as RBF version) ---
        if isinstance(self.measure, EmpiricalMeasure):
            def cov_X_s(X):
                location = tf.cast(self.measure.location, X.dtype)
                weights  = tf.cast(self.measure.weights,  X.dtype)
                tf.debugging.assert_shapes(
                    [(X, ("N", 1)), (location, ("M", 1)), (weights, ("M", 1))]
                )
                return tf.matmul(self.base_kernel(X, location), weights)

            def var_s():
                location = tf.cast(self.measure.location, tf.float64)
                weights  = tf.cast(self.measure.weights,  tf.float64)
                tf.debugging.assert_shapes([(location, ("M", 1)), (weights, ("M", 1))])
                return tf.squeeze(
                    tf.matmul(
                        tf.matmul(weights, self.base_kernel(location), transpose_a=True),
                        weights,
                    )
                )

        # --- MOGMeasure: exact via Gaussian closed forms + component summations ---
        if isinstance(self.measure, MOGMeasure):
            means    = tf.cast(self.measure.means, tf.float64)      # [M,1]
            variances= tf.cast(self.measure.variances, tf.float64)  # [M,1] or [M]
            weights  = tf.cast(self.measure.weights, tf.float64)    # [M,1]

            def cov_X_s(X):
                tf.debugging.assert_shapes([(X, ("N", 1))])
                ell = tf.cast(self.base_kernel.lengthscales, tf.float64)
                sigma2 = tf.cast(self.base_kernel.variance, tf.float64)

                # Sum_i w_i * E_{Y~N(mu_i, var_i)}[k(X,Y)]
                # Vectorize over components i
                # X shape [N,1] -> broadcast against [M,1]
                X64 = tf.cast(X, tf.float64)
                # Compute per-component expectations, then weight-sum
                comps = _matern32_expectation_against_gaussian(
                    x=X64, mu=tf.transpose(means), var=tf.transpose(variances),
                    ell=ell, sigma2=sigma2
                )  # shape [N, M] if broadcasting works across transpose; ensure dims
                # Ensure shapes: expand to [N, M]
                comps = tf.squeeze(comps, axis=-1)  # [N, M]
                w = tf.reshape(weights, (-1, 1))    # [M,1]
                return tf.matmul(comps, w)          # [N,1]

            def var_s():
                ell = tf.cast(self.base_kernel.lengthscales, tf.float64)
                sigma2 = tf.cast(self.base_kernel.variance, tf.float64)
                mu = tf.reshape(means, (-1,))         # [M]
                v  = tf.reshape(variances, (-1,))     # [M]
                w  = tf.reshape(weights, (-1,))       # [M]

                # Build pairwise E[k(N(mu_i,v_i), N(mu_j,v_j))] and weight with w_i w_j
                mu_i = tf.expand_dims(mu, 1)  # [M,1]
                mu_j = tf.expand_dims(mu, 0)  # [1,M]
                v_i  = tf.expand_dims(v,  1)  # [M,1]
                v_j  = tf.expand_dims(v,  0)  # [1,M]

                raise NotImplementedError
                Ek = _matern32_double_gaussian(
                    var=variances, ell=ell, sigma2=sigma2
                )  # [M,M]
                W = tf.tensordot(w, w, axes=0)  # [M,M]
                return tf.reduce_sum(W * Ek)

        self.cov_X_s = cov_X_s
        self.var_s = var_s

    def K(self, X: np.ndarray, X2: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Kernel matrix K(X, X2) with orthogonality to the measure.
        """
        cov_X_s = self.cov_X_s(X)          # [N,1] (or [...,N,1])
        cov_X2_s = cov_X_s if X2 is None else self.cov_X_s(X2)  # [M,1]
        denom = self.var_s()               # scalar
        return self.base_kernel(X, X2) - tf.tensordot(cov_X_s, tf.transpose(cov_X2_s), 1) / denom

    def K_diag(self, X):
        cov_X_s = self.cov_X_s(X)
        return self.base_kernel.K_diag(X) - tf.square(cov_X_s[:, 0]) / self.var_s()
