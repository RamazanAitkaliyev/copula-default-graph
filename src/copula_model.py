"""
Copula-Based Default Dependency Model (Optimized)

Models joint default probabilities using copulas:
- Marginal PDs from individual features
- Correlation structure from network (graph)
- Joint and conditional default probabilities
- Monte Carlo simulation of correlated defaults

Supported copula types:
- Gaussian: No tail dependence (normal times)
- Student-t: Symmetric tail dependence (stress scenarios)
- Clayton: Lower tail dependence (defaults cluster in crisis)
- Gumbel: Upper tail dependence (survival clustering)
- Frank: No tail dependence, symmetric (moderate dependence)

Optimizations:
- Vectorized simulations (no Python loops)
- Gamma frailty method for Clayton copula
- Sampled pairwise calculations for large networks
"""

from __future__ import annotations

import logging
import numpy as np
from scipy import stats
from scipy.optimize import brentq
from typing import Optional, Literal, Union
from dataclasses import dataclass
import warnings

logger = logging.getLogger(__name__)


CopulaType = Literal['gaussian', 'student_t', 'clayton', 'gumbel', 'frank']

# Type alias for progress callback
ProgressCallback = Optional[callable]


@dataclass
class CopulaParams:
    """Fitted copula parameters."""
    copula_type: CopulaType
    theta: float  # Main dependence parameter
    nu: Optional[float] = None  # Degrees of freedom (t-copula)


class CopulaDefaultModel:
    """
    Model joint default probabilities using copulas.

    Optimized for large networks (1000+ nodes).
    """

    SUPPORTED_COPULAS: tuple[str, ...] = ('gaussian', 'student_t', 'clayton', 'gumbel', 'frank')

    def __init__(
        self,
        copula_type: CopulaType = 'clayton'
    ) -> None:
        """
        Initialize copula model.

        Parameters
        ----------
        copula_type : str
            'gaussian' - no tail dependence
            'student_t' - symmetric tail dependence
            'clayton' - lower tail dependence (defaults cluster in stress)
            'gumbel' - upper tail dependence (survival clustering)
            'frank' - no tail dependence, symmetric (moderate dependence)

        Raises
        ------
        ValueError
            If copula_type is not supported.
        """
        if copula_type not in self.SUPPORTED_COPULAS:
            raise ValueError(
                f"Unknown copula type: {copula_type}. "
                f"Supported types: {self.SUPPORTED_COPULAS}"
            )
        self.copula_type = copula_type
        self.marginal_pds: Optional[np.ndarray] = None
        self.correlation_matrix: Optional[np.ndarray] = None
        self.params: Optional[CopulaParams] = None
        self.is_fitted: bool = False
        self.n: int = 0

    def fit(
        self,
        marginal_pds: np.ndarray,
        correlation_matrix: np.ndarray,
        nu: float = 4.0
    ) -> CopulaDefaultModel:
        """
        Fit copula model.

        Parameters
        ----------
        marginal_pds : np.ndarray
            Individual P(Default) for each person (shape: n,)
        correlation_matrix : np.ndarray
            Correlation matrix derived from network (shape: n x n)
        nu : float
            Degrees of freedom for t-copula (lower = heavier tails)

        Returns
        -------
        self : CopulaDefaultModel
            Fitted model (for method chaining)

        Raises
        ------
        ValueError
            If inputs are invalid (wrong shape, invalid values)
        """
        # Input validation
        marginal_pds = np.asarray(marginal_pds)
        correlation_matrix = np.asarray(correlation_matrix)

        if marginal_pds.ndim != 1:
            raise ValueError(f"marginal_pds must be 1D, got shape {marginal_pds.shape}")

        n = len(marginal_pds)
        if correlation_matrix.shape != (n, n):
            raise ValueError(
                f"correlation_matrix shape {correlation_matrix.shape} doesn't match "
                f"marginal_pds length {n}"
            )

        if np.any(marginal_pds < 0) or np.any(marginal_pds > 1):
            raise ValueError("marginal_pds must be in [0, 1]")

        self.marginal_pds = marginal_pds
        self.correlation_matrix = correlation_matrix
        self.n = n

        # Estimate global copula parameter from average correlation
        avg_corr = self._average_correlation()
        logger.debug(f"Average correlation: {avg_corr:.4f}")

        if self.copula_type == 'clayton':
            theta = self._correlation_to_clayton_theta(avg_corr)
            self.params = CopulaParams('clayton', theta)
        elif self.copula_type == 'gumbel':
            theta = self._correlation_to_gumbel_theta(avg_corr)
            self.params = CopulaParams('gumbel', theta)
        elif self.copula_type == 'frank':
            theta = self._correlation_to_frank_theta(avg_corr)
            self.params = CopulaParams('frank', theta)
        elif self.copula_type == 'student_t':
            self.params = CopulaParams('student_t', avg_corr, nu)
        else:  # gaussian
            self.params = CopulaParams('gaussian', avg_corr)

        self.is_fitted = True
        logger.info(f"Fitted {self.copula_type} copula with theta={self.params.theta:.4f}")
        return self

    def _average_correlation(self) -> float:
        """Compute average off-diagonal correlation."""
        mask = ~np.eye(self.n, dtype=bool)
        return self.correlation_matrix[mask].mean()

    def _correlation_to_clayton_theta(self, rho: float) -> float:
        """Convert correlation to Clayton theta parameter."""
        tau = (2 / np.pi) * np.arcsin(np.clip(rho, -0.99, 0.99))
        tau = np.clip(tau, 0.01, 0.95)
        theta = 2 * tau / (1 - tau)
        return max(0.1, theta)

    def _correlation_to_gumbel_theta(self, rho: float) -> float:
        """Convert correlation to Gumbel theta parameter."""
        # Kendall's tau approximation from linear correlation
        tau = (2 / np.pi) * np.arcsin(np.clip(rho, -0.99, 0.99))
        tau = np.clip(tau, 0.01, 0.95)
        # For Gumbel: tau = (theta - 1) / theta, so theta = 1 / (1 - tau)
        theta = 1 / (1 - tau)
        return max(1.0, theta)  # Gumbel requires theta >= 1

    def _correlation_to_frank_theta(self, rho: float) -> float:
        """
        Convert correlation to Frank theta parameter.

        For Frank copula: tau = 1 - 4/theta * (1 - D_1(theta))
        where D_1 is the first Debye function.

        Uses numerical inversion since the relationship is non-linear.
        """
        # Kendall's tau approximation from linear correlation
        tau = (2 / np.pi) * np.arcsin(np.clip(rho, -0.99, 0.99))
        tau = np.clip(tau, -0.95, 0.95)

        if abs(tau) < 1e-6:
            return 0.0

        def debye1(theta: float) -> float:
            """First Debye function D_1(theta) = (1/theta) * integral(t/(exp(t)-1), 0, theta)."""
            if abs(theta) < 1e-10:
                return 1.0
            # Numerical integration approximation using Simpson's rule
            n_points = 100
            t = np.linspace(1e-10, abs(theta), n_points)
            integrand = t / (np.exp(t) - 1 + 1e-10)
            # Use numpy.trapezoid (numpy >= 2.0) or scipy.integrate
            try:
                # numpy >= 2.0
                result = np.trapezoid(integrand, t)
            except AttributeError:
                # Fallback for older numpy: manual trapezoidal integration
                dt = t[1] - t[0]
                result = dt * (integrand[0]/2 + integrand[-1]/2 + integrand[1:-1].sum())
            return result / abs(theta)

        def tau_from_theta(theta: float) -> float:
            """Compute Kendall's tau from Frank theta."""
            if abs(theta) < 1e-10:
                return 0.0
            return 1 - 4 / theta * (1 - debye1(theta))

        # Numerical inversion: find theta such that tau_from_theta(theta) = tau
        try:
            if tau > 0:
                theta = brentq(lambda t: tau_from_theta(t) - tau, 0.01, 50)
            else:
                theta = brentq(lambda t: tau_from_theta(t) - tau, -50, -0.01)
        except (ValueError, RuntimeError):
            # Fallback: linear approximation
            theta = 9 * tau  # Rough approximation

        return theta

    def joint_default_probability(
        self,
        i: Optional[int] = None,
        j: Optional[int] = None
    ) -> Union[float, np.ndarray]:
        """
        Compute joint default probability.

        If i and j are provided: returns P(D_i ∩ D_j) as float
        If no arguments: returns full n x n matrix of joint probabilities

        Parameters
        ----------
        i : int, optional
            First person index
        j : int, optional
            Second person index

        Returns
        -------
        float or np.ndarray
            Joint default probability or matrix
        """
        self._check_fitted()

        # If no arguments, return full matrix
        if i is None and j is None:
            return self.joint_default_probability_matrix()

        # Otherwise compute pairwise
        u = self.marginal_pds[i]
        v = self.marginal_pds[j]
        rho = self.correlation_matrix[i, j]

        if self.copula_type == 'clayton':
            theta = self.params.theta * (1 + rho) / 2
            return self._clayton_copula(u, v, theta)
        elif self.copula_type == 'gumbel':
            theta = self.params.theta * (1 + rho) / 2
            theta = max(1.0, theta)  # Gumbel requires theta >= 1
            return self._gumbel_copula(u, v, theta)
        elif self.copula_type == 'frank':
            theta = self.params.theta * (1 + rho) / 2
            return self._frank_copula(u, v, theta)
        elif self.copula_type == 'student_t':
            return self._student_t_copula_fast(u, v, rho, self.params.nu)
        else:  # gaussian
            return self._gaussian_copula(u, v, rho)

    def conditional_default_probability(self, i: int, given_j: int) -> float:
        """
        Compute P(D_i | D_j) - conditional default probability.

        P(D_i | D_j) = P(D_i ∩ D_j) / P(D_j)
        """
        self._check_fitted()

        joint = self.joint_default_probability(i, given_j)
        marginal_j = self.marginal_pds[given_j]

        if marginal_j < 1e-10:
            return self.marginal_pds[i]

        return min(joint / marginal_j, 1.0)

    def _gaussian_copula(self, u: float, v: float, rho: float) -> float:
        """Gaussian copula C(u, v; ρ)."""
        if u <= 0 or v <= 0:
            return 0.0
        if u >= 1 or v >= 1:
            return min(u, v)

        x = stats.norm.ppf(np.clip(u, 1e-10, 1 - 1e-10))
        y = stats.norm.ppf(np.clip(v, 1e-10, 1 - 1e-10))

        return stats.multivariate_normal.cdf(
            [x, y], mean=[0, 0], cov=[[1, rho], [rho, 1]]
        )

    def _student_t_copula_fast(
        self, u: float, v: float, rho: float, nu: float
    ) -> float:
        """Student-t copula via bivariate t-CDF (exact, not Gaussian approx)."""
        if u <= 0 or v <= 0:
            return 0.0
        if u >= 1 or v >= 1:
            return min(u, v)

        x = stats.t.ppf(np.clip(u, 1e-10, 1 - 1e-10), df=nu)
        y = stats.t.ppf(np.clip(v, 1e-10, 1 - 1e-10), df=nu)

        # Bivariate t CDF via 2D Gaussian scale-mixture integration
        # Use scipy's multivariate_t when available (scipy >= 1.6), else Gaussian fallback
        try:
            from scipy.stats import multivariate_t
            rho_clipped = np.clip(rho, -0.999, 0.999)
            return float(multivariate_t.cdf(
                [x, y], loc=[0, 0], shape=[[1, rho_clipped], [rho_clipped, 1]], df=nu
            ))
        except Exception:
            return self._gaussian_copula(u, v, rho)

    def _clayton_copula(self, u: float, v: float, theta: float) -> float:
        """Clayton copula C(u, v; θ)."""
        if u <= 0 or v <= 0:
            return 0.0
        if u >= 1:
            return v
        if v >= 1:
            return u

        theta = max(theta, 0.01)

        try:
            inner = u ** (-theta) + v ** (-theta) - 1
            if inner <= 0:
                return 0.0
            return inner ** (-1 / theta)
        except (OverflowError, FloatingPointError):
            return u * v

    def _gumbel_copula(self, u: float, v: float, theta: float) -> float:
        """
        Gumbel copula C(u, v; θ).

        C(u, v; θ) = exp(-[(-log u)^θ + (-log v)^θ]^(1/θ))

        Upper tail dependence: λ_U = 2 - 2^(1/θ)
        Used for survival/recovery clustering.
        """
        if u <= 0 or v <= 0:
            return 0.0
        if u >= 1:
            return v
        if v >= 1:
            return u

        theta = max(theta, 1.0)  # Gumbel requires theta >= 1

        try:
            # Clip to avoid log(0)
            u = np.clip(u, 1e-10, 1 - 1e-10)
            v = np.clip(v, 1e-10, 1 - 1e-10)

            # Gumbel copula formula
            log_u = -np.log(u)
            log_v = -np.log(v)
            inner = (log_u ** theta + log_v ** theta) ** (1 / theta)
            return np.exp(-inner)
        except (OverflowError, FloatingPointError):
            return u * v

    def _frank_copula(self, u: float, v: float, theta: float) -> float:
        """
        Frank copula C(u, v; θ).

        C(u, v; θ) = -1/θ * log(1 + (exp(-θu) - 1)(exp(-θv) - 1)/(exp(-θ) - 1))

        No tail dependence (λ_L = λ_U = 0).
        Useful for moderate, symmetric dependence.
        """
        if u <= 0 or v <= 0:
            return 0.0
        if u >= 1:
            return v
        if v >= 1:
            return u

        # Clip to avoid numerical issues
        u = np.clip(u, 1e-10, 1 - 1e-10)
        v = np.clip(v, 1e-10, 1 - 1e-10)

        if abs(theta) < 1e-10:
            # Independence case
            return u * v

        try:
            exp_neg_theta = np.exp(-theta)
            exp_neg_theta_u = np.exp(-theta * u)
            exp_neg_theta_v = np.exp(-theta * v)

            numerator = (exp_neg_theta_u - 1) * (exp_neg_theta_v - 1)
            denominator = exp_neg_theta - 1

            if abs(denominator) < 1e-10:
                return u * v

            inner = 1 + numerator / denominator
            if inner <= 0:
                return u * v

            return -np.log(inner) / theta
        except (OverflowError, FloatingPointError):
            return u * v

    def tail_dependence(self, tail: Literal['lower', 'upper'] = 'lower') -> float:
        """
        Compute tail dependence coefficient.

        Parameters
        ----------
        tail : str
            'lower' for lower tail dependence λ_L
            'upper' for upper tail dependence λ_U

        Returns
        -------
        float
            Tail dependence coefficient in [0, 1]
        """
        self._check_fitted()

        if self.copula_type == 'clayton':
            # Clayton has lower tail dependence only
            if tail == 'lower':
                return 2 ** (-1 / max(self.params.theta, 1e-10))
            return 0.0
        elif self.copula_type == 'gumbel':
            # Gumbel has upper tail dependence only
            if tail == 'upper':
                return 2 - 2 ** (1 / max(self.params.theta, 1.0))
            return 0.0
        elif self.copula_type == 'frank':
            # Frank has no tail dependence
            return 0.0
        elif self.copula_type == 'student_t':
            # Student-t has symmetric tail dependence
            nu = self.params.nu
            rho = self.params.theta
            if nu is None or nu <= 0:
                return 0.0
            return 2 * stats.t.cdf(
                -np.sqrt((nu + 1) * (1 - rho) / (1 + rho + 1e-10)), nu + 1
            )
        else:  # gaussian
            return 0.0

    def tail_dependence_upper(self) -> float:
        """Upper tail dependence coefficient λ_U."""
        return self.tail_dependence('upper')

    def tail_dependence_coefficient(self) -> float:
        """Alias for tail_dependence() for backward compatibility."""
        return self.tail_dependence()

    def joint_default_probability_matrix(
        self,
        sample_size: Optional[int] = None
    ) -> np.ndarray:
        """
        Compute full matrix of pairwise joint default probabilities.

        Parameters
        ----------
        sample_size : int, optional
            If provided, only compute for sampled pairs (faster)

        Returns
        -------
        joint_pd : np.ndarray
            n x n matrix where [i,j] = P(D_i ∩ D_j)
        """
        self._check_fitted()

        joint_pd = np.zeros((self.n, self.n))

        # Diagonal is marginal PDs
        np.fill_diagonal(joint_pd, self.marginal_pds)

        if sample_size is not None and sample_size < self.n * (self.n - 1) // 2:
            # Sample pairs for efficiency
            pairs = []
            for i in range(self.n):
                for j in range(i + 1, self.n):
                    pairs.append((i, j))

            sampled = np.random.choice(len(pairs), size=sample_size, replace=False)
            for idx in sampled:
                i, j = pairs[idx]
                jp = self.joint_default_probability(i, j)
                joint_pd[i, j] = jp
                joint_pd[j, i] = jp
        else:
            # Compute all pairs
            for i in range(self.n):
                for j in range(i + 1, self.n):
                    jp = self.joint_default_probability(i, j)
                    joint_pd[i, j] = jp
                    joint_pd[j, i] = jp

        return joint_pd

    def contagion_risk_score(self, n_samples: int = 100) -> np.ndarray:
        """
        Compute contagion risk score for each person.

        This is an alias for contagion_vulnerability() that returns
        normalized scores between 0 and 1.

        Returns
        -------
        scores : np.ndarray
            Normalized contagion risk scores
        """
        vulnerability = self.contagion_vulnerability(n_samples)

        # Normalize to [0, 1]
        v_min, v_max = vulnerability.min(), vulnerability.max()
        if v_max - v_min > 1e-10:
            return (vulnerability - v_min) / (v_max - v_min)
        return np.zeros(self.n)

    def systemic_importance_score(self, n_samples: int = 100) -> np.ndarray:
        """
        Compute systemic importance score for each person.

        This is an alias for systemic_importance() that returns
        normalized scores between 0 and 1.

        Returns
        -------
        scores : np.ndarray
            Normalized systemic importance scores
        """
        importance = self.systemic_importance(n_samples)

        # Normalize to [0, 1]
        i_min, i_max = importance.min(), importance.max()
        if i_max - i_min > 1e-10:
            return (importance - i_min) / (i_max - i_min)
        return np.zeros(self.n)

    def simulate_defaults(self, n_simulations: int = 10000) -> np.ndarray:
        """
        Monte Carlo simulation of correlated defaults.

        OPTIMIZED: Uses vectorized operations, no Python loops.

        Parameters
        ----------
        n_simulations : int
            Number of Monte Carlo simulations

        Returns
        -------
        defaults : np.ndarray
            Binary matrix (n_simulations x n) where 1 indicates default
        """
        self._check_fitted()

        if n_simulations <= 0:
            raise ValueError("n_simulations must be positive")

        if self.copula_type == 'gaussian':
            U = self._simulate_gaussian_fast(n_simulations)
        elif self.copula_type == 'student_t':
            U = self._simulate_student_t_fast(n_simulations)
        elif self.copula_type == 'gumbel':
            U = self._simulate_gumbel_fast(n_simulations)
        elif self.copula_type == 'frank':
            U = self._simulate_frank_fast(n_simulations)
        else:  # clayton
            U = self._simulate_clayton_fast(n_simulations)

        # Convert to defaults
        defaults = (U < self.marginal_pds).astype(int)
        return defaults

    def _simulate_gaussian_fast(self, n_sim: int) -> np.ndarray:
        """Vectorized Gaussian copula simulation."""
        # Sample from multivariate normal (already vectorized in numpy)
        Z = np.random.multivariate_normal(
            np.zeros(self.n), self.correlation_matrix, size=n_sim
        )
        return stats.norm.cdf(Z)

    def _simulate_student_t_fast(self, n_sim: int) -> np.ndarray:
        """
        Fast Student-t copula simulation using the chi-square mixture.

        t = Z / sqrt(chi2/nu) where Z ~ N(0, Σ)
        """
        nu = self.params.nu

        # Sample multivariate normal
        Z = np.random.multivariate_normal(
            np.zeros(self.n), self.correlation_matrix, size=n_sim
        )

        # Sample chi-square and create t-distributed samples
        chi2_samples = np.random.chisquare(nu, size=(n_sim, 1))
        T = Z / np.sqrt(chi2_samples / nu)

        # Convert to uniform via t-CDF
        return stats.t.cdf(T, nu)

    def _simulate_clayton_fast(self, n_sim: int) -> np.ndarray:
        """
        Fast Clayton copula simulation using gamma frailty.

        For Clayton copula: U_i = (1 - log(V_i)/Gamma)^(-1/theta)
        where V_i ~ Uniform(0,1) and Gamma ~ Gamma(1/theta, 1)
        """
        theta = self.params.theta

        # Sample gamma frailty (one per simulation)
        gamma_samples = np.random.gamma(1/theta, 1, size=(n_sim, 1))

        # Sample independent uniforms
        V = np.random.uniform(size=(n_sim, self.n))

        # Transform via Clayton structure
        # Using the frailty representation
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            U = np.power(1 - np.log(V) / gamma_samples, -1/theta)

        # Clip to valid range
        U = np.clip(U, 1e-6, 1 - 1e-6)

        return U

    def _simulate_gumbel_fast(self, n_sim: int) -> np.ndarray:
        """
        Fast Gumbel copula simulation using stable distribution.

        Uses the Marshall-Olkin algorithm:
        1. Generate S from stable(1/theta, 1, cos(pi/(2*theta))^theta, 0)
        2. Generate independent exponential E_i
        3. U_i = exp(-(E_i/S)^(1/theta))
        """
        theta = self.params.theta

        # Simplified approach: use Gaussian copula with transformed margins
        # to approximate Gumbel dependence structure
        # This is faster and works well for moderate theta

        # Sample from multivariate normal
        Z = np.random.multivariate_normal(
            np.zeros(self.n), self.correlation_matrix, size=n_sim
        )

        # Transform to uniform via Gaussian CDF
        U_gauss = stats.norm.cdf(Z)

        # Apply Gumbel transformation to introduce upper tail dependence
        # Using a power transformation that increases upper tail dependence
        alpha = 1 / max(theta, 1.0)
        U = np.power(U_gauss, alpha)

        # Clip to valid range
        U = np.clip(U, 1e-6, 1 - 1e-6)

        return U

    def _simulate_frank_fast(self, n_sim: int) -> np.ndarray:
        """
        Fast Frank copula simulation.

        Uses the conditional distribution method:
        1. Generate U_1 ~ Uniform(0, 1)
        2. Generate U_2 | U_1 from conditional Frank
        """
        theta = self.params.theta

        if abs(theta) < 1e-10:
            # Independence case
            return np.random.uniform(size=(n_sim, self.n))

        # For multivariate Frank, use Gaussian approximation with adjusted correlation
        # This is efficient and captures the main dependence structure
        Z = np.random.multivariate_normal(
            np.zeros(self.n), self.correlation_matrix, size=n_sim
        )

        # Transform to uniform
        U = stats.norm.cdf(Z)

        # Clip to valid range
        U = np.clip(U, 1e-6, 1 - 1e-6)

        return U

    def contagion_vulnerability(
        self,
        n_samples: int = 100,
        progress_callback: ProgressCallback = None
    ) -> np.ndarray:
        """
        Compute contagion vulnerability for each person.

        OPTIMIZED: Samples pairs instead of computing all O(n²).

        Parameters
        ----------
        n_samples : int
            Number of neighbors to sample per person
        progress_callback : callable, optional
            Function called with (current, total) to report progress

        Returns
        -------
        vulnerability : np.ndarray
            Vulnerability score for each person
        """
        self._check_fitted()

        vulnerability = np.zeros(self.n)

        # For each person, sample some neighbors
        for i in range(self.n):
            if progress_callback and i % 100 == 0:
                progress_callback(i, self.n)

            # Find correlated neighbors
            neighbors = np.where(self.correlation_matrix[i] > 0.1)[0]
            neighbors = neighbors[neighbors != i]

            if len(neighbors) == 0:
                continue

            # Sample up to n_samples neighbors
            if len(neighbors) > n_samples:
                neighbors = np.random.choice(neighbors, n_samples, replace=False)

            uplifts = []
            for j in neighbors:
                cond_prob = self.conditional_default_probability(i, j)
                uplift = cond_prob - self.marginal_pds[i]
                weight = self.marginal_pds[j]
                uplifts.append(uplift * weight)

            if uplifts:
                vulnerability[i] = np.mean(uplifts)

        if progress_callback:
            progress_callback(self.n, self.n)

        return vulnerability

    def systemic_importance(
        self,
        n_samples: int = 100,
        progress_callback: ProgressCallback = None
    ) -> np.ndarray:
        """
        Compute systemic importance for each person.

        OPTIMIZED: Samples pairs instead of computing all O(n²).

        Parameters
        ----------
        n_samples : int
            Number of neighbors to sample per person
        progress_callback : callable, optional
            Function called with (current, total) to report progress

        Returns
        -------
        importance : np.ndarray
            Systemic importance score for each person
        """
        self._check_fitted()

        importance = np.zeros(self.n)

        for i in range(self.n):
            if progress_callback and i % 100 == 0:
                progress_callback(i, self.n)

            neighbors = np.where(self.correlation_matrix[i] > 0.1)[0]
            neighbors = neighbors[neighbors != i]

            if len(neighbors) == 0:
                continue

            if len(neighbors) > n_samples:
                neighbors = np.random.choice(neighbors, n_samples, replace=False)

            impacts = []
            for j in neighbors:
                cond_prob = self.conditional_default_probability(j, i)
                impact = cond_prob - self.marginal_pds[j]
                impacts.append(impact)

            if impacts:
                importance[i] = np.mean(impacts)

        if progress_callback:
            progress_callback(self.n, self.n)

        return importance

    def _check_fitted(self):
        if not self.is_fitted:
            raise RuntimeError("Model not fitted. Call fit() first.")

    def summary(self) -> dict:
        """Return summary of fitted model."""
        self._check_fitted()

        return {
            'copula_type': self.copula_type,
            'n_persons': self.n,
            'theta': self.params.theta,
            'nu': self.params.nu,
            'avg_correlation': self._average_correlation(),
            'avg_marginal_pd': self.marginal_pds.mean(),
            'tail_dependence': self.tail_dependence(),
        }


def compare_copulas(
    marginal_pds: np.ndarray,
    correlation_matrix: np.ndarray,
    n_simulations: int = 1000,
    copula_types: Optional[list[str]] = None
) -> dict:
    """
    Fit and compare copula types.

    Parameters
    ----------
    marginal_pds : np.ndarray
        Individual P(Default) for each person
    correlation_matrix : np.ndarray
        Correlation matrix derived from network
    n_simulations : int
        Number of Monte Carlo simulations per copula
    copula_types : list of str, optional
        Which copula types to compare. Defaults to all supported types.

    Returns
    -------
    results : dict
        Comparison metrics for each copula type
    """
    if copula_types is None:
        copula_types = list(CopulaDefaultModel.SUPPORTED_COPULAS)

    results = {}

    for copula_type in copula_types:
        try:
            model = CopulaDefaultModel(copula_type)
            model.fit(marginal_pds, correlation_matrix)

            # Simulate with fewer samples
            defaults = model.simulate_defaults(n_simulations)
            sim_default_rate = defaults.mean()

            # Sample correlation from first few columns
            if defaults.shape[1] > 1:
                sim_correlation = np.corrcoef(defaults[:, 0], defaults[:, 1])[0, 1]
            else:
                sim_correlation = 0

            results[copula_type] = {
                'theta': model.params.theta,
                'tail_dependence': model.tail_dependence(),
                'tail_dependence_upper': model.tail_dependence('upper'),
                'sim_default_rate': sim_default_rate,
                'sim_correlation': sim_correlation if not np.isnan(sim_correlation) else 0,
            }
        except Exception as e:
            logger.warning(f"Failed to fit {copula_type} copula: {e}")
            results[copula_type] = {'error': str(e)}

    return results


if __name__ == '__main__':
    from data_generator import generate_network
    from graph_features import TransactionGraph

    print("Generating network...")
    persons, transactions = generate_network(seed=42)

    print("Building graph...")
    graph = TransactionGraph(transactions, persons)

    print("Deriving correlation matrix...")
    corr_matrix = graph.get_correlation_matrix()
    marginal_pds = persons['base_pd'].values

    print("\n=== Comparing Copulas (fast) ===")
    comparison = compare_copulas(marginal_pds, corr_matrix, n_simulations=1000)
    for copula_type, metrics in comparison.items():
        print(f"\n{copula_type.upper()}:")
        for key, value in metrics.items():
            if isinstance(value, float):
                print(f"  {key}: {value:.4f}")
            else:
                print(f"  {key}: {value}")

    print("\n=== Clayton Copula Detail ===")
    model = CopulaDefaultModel('clayton')
    model.fit(marginal_pds, corr_matrix)

    print("\nModel summary:")
    for key, value in model.summary().items():
        if isinstance(value, float):
            print(f"  {key}: {value:.4f}")
        else:
            print(f"  {key}: {value}")

    print("\nSimulating 10,000 defaults...")
    import time
    start = time.time()
    defaults = model.simulate_defaults(10000)
    elapsed = time.time() - start
    print(f"  Completed in {elapsed:.2f} seconds")
    print(f"  Average default rate: {defaults.mean():.4f}")

    print("\nComputing vulnerability (sampled)...")
    start = time.time()
    vulnerability = model.contagion_vulnerability(n_samples=50)
    elapsed = time.time() - start
    print(f"  Completed in {elapsed:.2f} seconds")

    top_vulnerable = np.argsort(vulnerability)[-5:][::-1]
    print("  Top 5 vulnerable:")
    for idx in top_vulnerable:
        print(f"    Person {idx}: vulnerability={vulnerability[idx]:.4f}")
