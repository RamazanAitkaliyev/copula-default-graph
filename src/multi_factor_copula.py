"""
multi_factor_copula.py — Multi-factor Vasicek copula (geo ⟂ transfer, ...).

The single-factor `FactorCopula` lets each borrower load on ONE systematic
factor. Real correlation in this framework comes from MULTIPLE, independent
sources that the user considers equally important:

  * a GEO factor   (geo_cluster_id)      — shared regional economy / shocks
  * a TRANSFER factor (transfer_cluster_id) — shared money-flow community

This module generalizes the Vasicek model to K independent systematic factors:

    A_i = Σ_k β_{i,k} · Y_{k, f_k(i)}  +  sqrt(1 - Σ_k β_{i,k}²) · ε_i

  - Y_{k,·} are independent standard-normal systematic factors (one set per
    dimension k; within a dimension, only borrowers sharing the same factor id
    f_k(i) load on the same Y).
  - ε_i is the idiosyncratic standard normal.
  - default_i  ⇔  A_i ≤ Φ⁻¹(PD_i).
  - **Constraint:** Σ_k β_{i,k}² < 1  (positive idiosyncratic variance).

Implied asset correlation (the quantity the copula block needs):

    corr(A_i, A_j) = Σ_k β_{i,k} · β_{j,k} · 1[f_k(i) == f_k(j) and f_k(i) >= 0]

So two borrowers in the SAME geo cluster but different transfer clusters have
corr = β_geo², in BOTH share corr = β_geo² + β_transfer², and in NEITHER corr = 0.
Setting β_geo = β_transfer makes the two sources "equally important", exactly as
requested.

Storage is O(n · K) (here K = 2) — never the n×n matrix — so it scales to 10M+
exactly like the single-factor copula. The joint-default block reuses
FactorCopula's verified bivariate-normal CDF (`_bvn_cdf`), so this is a true
drop-in for `RiskRatioCalculator` (same duck-typed interface:
`marginal_pds`, `is_fitted`, `simulate_defaults`, `simulate_default_rate`,
`joint_default_probability_block`).

Gaussian variant implemented and tested; the Student-t variant is intentionally
deferred (the single-factor `FactorCopula(student_t=True)` covers heavy-tail
needs for now).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Sequence, Union

import numpy as np
from scipy import stats

from .factor_copula import FactorCopula  # reuse verified _bvn_cdf, etc.


@dataclass
class MultiFactorCopulaParams:
    n: int
    n_factors: int
    betas_per_factor: List[float] = field(default_factory=list)  # mean loading per dim
    factor_cardinality: List[int] = field(default_factory=list)  # #distinct ids per dim
    copula_type: str = "gaussian_multifactor"


class MultiFactorCopula:
    """
    K-factor Gaussian Vasicek copula. Drop-in for RiskRatioCalculator.

    Parameters
    ----------
    The model is configured entirely through `fit`.

    Example
    -------
        mfc = MultiFactorCopula().fit(
            marginal_pds=pds,
            factor_matrix=np.column_stack([geo_cluster_id, transfer_cluster_id]),
            betas=np.column_stack([beta_geo, beta_transfer]),   # or scalars
        )
        calc = RiskRatioCalculator(mfc, persons, exposures=ead, lgd=lgd)
    """

    def __init__(self) -> None:
        self.copula_type = "gaussian_multifactor"
        self.marginal_pds: Optional[np.ndarray] = None
        self.factor_matrix: Optional[np.ndarray] = None     # (n, K) raw ids (-1 = none)
        self.betas: Optional[np.ndarray] = None             # (n, K) loadings
        self.params: Optional[MultiFactorCopulaParams] = None
        self.is_fitted: bool = False
        self.n: int = 0
        self.n_factors: int = 0

        self._thresholds: Optional[np.ndarray] = None       # Φ⁻¹(PD)
        self._factor_index: Optional[np.ndarray] = None     # (n, K) compressed ids
        self._sum_beta_sq: Optional[np.ndarray] = None      # (n,) Σ_k β²

    # ── fit ─────────────────────────────────────────────────────────────────
    def fit(
        self,
        marginal_pds: np.ndarray,
        factor_matrix: np.ndarray,
        betas: Union[float, Sequence[float], np.ndarray] = 0.3,
    ) -> "MultiFactorCopula":
        pds = np.asarray(marginal_pds, dtype=float)
        n = len(pds)
        fm = np.asarray(factor_matrix)
        if fm.ndim == 1:
            fm = fm.reshape(-1, 1)
        if fm.shape[0] != n:
            raise ValueError(
                f"factor_matrix has {fm.shape[0]} rows != {n} marginal_pds."
            )
        K = fm.shape[1]

        # Broadcast betas to (n, K).
        betas_arr = self._broadcast_betas(betas, n, K)
        if np.any(betas_arr < 0):
            raise ValueError("betas (factor loadings) must be >= 0.")

        sum_beta_sq = (betas_arr ** 2).sum(axis=1)
        if np.any(sum_beta_sq >= 1.0):
            bad = int(np.argmax(sum_beta_sq))
            raise ValueError(
                f"Σ_k β² must be < 1 for positive idiosyncratic variance; "
                f"borrower index {bad} has Σβ²={sum_beta_sq[bad]:.3f}. "
                f"Lower the loadings."
            )
        if not np.all(np.isfinite(pds)) or pds.min() < 0 or pds.max() > 1:
            raise ValueError("marginal_pds must be finite and within [0, 1].")

        # Compress factor ids per dimension to 0..C-1, keeping -1 as -1 ("none").
        factor_index = np.empty_like(fm, dtype=np.int64)
        cardinalities = []
        for k in range(K):
            col = fm[:, k]
            comp, card = self._compress_with_none(col)
            factor_index[:, k] = comp
            cardinalities.append(card)

        self.marginal_pds = pds
        self.factor_matrix = fm
        self.betas = betas_arr
        self.n = n
        self.n_factors = K
        self._sum_beta_sq = sum_beta_sq
        self._factor_index = factor_index
        self._thresholds = stats.norm.ppf(np.clip(pds, 1e-12, 1 - 1e-12))
        self.params = MultiFactorCopulaParams(
            n=n, n_factors=K,
            betas_per_factor=[float(betas_arr[:, k].mean()) for k in range(K)],
            factor_cardinality=cardinalities,
        )
        self.is_fitted = True
        return self

    # ── implied correlation ──────────────────────────────────────────────────
    def implied_correlation_block(self, idx: np.ndarray) -> np.ndarray:
        """corr(A_i, A_j) = Σ_k β_ik β_jk · 1[same factor k]. Diagonal = 1."""
        self._check_fitted()
        idx = np.asarray(idx, dtype=int)
        m = len(idx)
        if m == 0:
            return np.zeros((0, 0))
        b = self.betas[idx]            # (m, K)
        fi = self._factor_index[idx]   # (m, K)
        corr = np.zeros((m, m))
        for k in range(self.n_factors):
            same = (fi[:, k][:, None] == fi[:, k][None, :]) & (fi[:, k][:, None] >= 0)
            corr += np.where(same, b[:, k][:, None] * b[:, k][None, :], 0.0)
        np.fill_diagonal(corr, 1.0)
        return corr

    # ── joint default block (drop-in for metrics) ────────────────────────────
    def joint_default_probability_block(self, idx: np.ndarray) -> np.ndarray:
        """
        m×m block of pairwise joint default probabilities P(D_a ∩ D_b), using the
        summed multi-factor correlation and the (verified) bivariate-normal CDF.
        Diagonal = marginal PD. Never materializes the full n×n.
        """
        self._check_fitted()
        idx = np.asarray(idx, dtype=int)
        m = len(idx)
        if m == 0:
            return np.zeros((0, 0))
        z = self._thresholds[idx]
        corr = self.implied_correlation_block(idx)

        Phi_z = stats.norm.cdf(z)
        out = np.outer(Phi_z, Phi_z)                  # independence baseline
        iu, ju = np.triu_indices(m, k=1)
        c = corr[iu, ju]
        mask = np.abs(c) > 1e-12
        if mask.any():
            ai, bj = iu[mask], ju[mask]
            cc = np.clip(c[mask], -0.999, 0.999)
            vals = FactorCopula._bvn_cdf(z[ai], z[bj], cc)   # reuse verified CDF
            out[ai, bj] = vals
            out[bj, ai] = vals
        np.fill_diagonal(out, self.marginal_pds[idx])
        return out

    def joint_default_probability(self, i: int, j: int) -> float:
        """Scalar joint default probability for a single pair."""
        block = self.joint_default_probability_block(np.array([i, j]))
        return float(block[0, 1])

    # ── simulation (streamed, O(n·K) memory) ─────────────────────────────────
    def simulate_defaults(self, n_simulations: int, seed: Optional[int] = None) -> np.ndarray:
        """
        Return an (n_simulations, n) int8 default matrix.
        Draws K systematic normals per scenario plus per-borrower idiosyncratic.
        """
        self._check_fitted()
        rng = np.random.default_rng(seed)
        n, K = self.n, self.n_factors
        betas = self.betas                          # (n, K)
        idio = np.sqrt(np.maximum(1.0 - self._sum_beta_sq, 0.0))  # (n,)
        thr = self._thresholds                      # (n,)

        # systematic draw: for each dim, one normal per distinct factor id.
        cardinalities = self.params.factor_cardinality
        A = np.zeros((n_simulations, n))
        for k in range(K):
            card = cardinalities[k]
            fidx = self._factor_index[:, k]         # (n,) in 0..card-1 or -1
            if card > 0:
                Yk = rng.standard_normal((n_simulations, card))   # (S, card)
                # map each borrower to its factor draw; -1 → 0 contribution
                has = fidx >= 0
                contrib = np.zeros((n_simulations, n))
                if has.any():
                    contrib[:, has] = Yk[:, fidx[has]] * betas[has, k]
                A += contrib
        A += rng.standard_normal((n_simulations, n)) * idio[None, :]
        return (A <= thr[None, :]).astype(np.int8)

    def simulate_default_rate(
        self, n_simulations: int, seed: Optional[int] = None, batch_size: int = 2000
    ) -> np.ndarray:
        """
        Memory-bounded portfolio default-RATE distribution over scenarios.
        Returns an array of length n_simulations (fraction defaulting each scenario).
        """
        self._check_fitted()
        rng = np.random.default_rng(seed)
        n, K = self.n, self.n_factors
        betas = self.betas
        idio = np.sqrt(np.maximum(1.0 - self._sum_beta_sq, 0.0))
        thr = self._thresholds
        cardinalities = self.params.factor_cardinality

        # cap cells to keep memory bounded (mirror FactorCopula behaviour)
        max_cells = 32_000_000
        safe_batch = max(1, int(max_cells // max(n, 1)))
        batch_size = min(batch_size, safe_batch)

        rates = np.empty(n_simulations)
        done = 0
        while done < n_simulations:
            b = min(batch_size, n_simulations - done)
            A = np.zeros((b, n))
            for k in range(K):
                card = cardinalities[k]
                fidx = self._factor_index[:, k]
                if card > 0:
                    Yk = rng.standard_normal((b, card))
                    has = fidx >= 0
                    if has.any():
                        A[:, has] += Yk[:, fidx[has]] * betas[has, k]
            A += rng.standard_normal((b, n)) * idio[None, :]
            defaults = A <= thr[None, :]
            rates[done:done + b] = defaults.mean(axis=1)
            done += b
        return rates

    # ── internals ─────────────────────────────────────────────────────────────
    def _check_fitted(self) -> None:
        if not self.is_fitted:
            raise RuntimeError("MultiFactorCopula.fit() must be called first.")

    @staticmethod
    def _broadcast_betas(betas, n: int, K: int) -> np.ndarray:
        if np.isscalar(betas):
            return np.full((n, K), float(betas))
        arr = np.asarray(betas, dtype=float)
        if arr.ndim == 1:
            if len(arr) == K:            # per-dimension loading
                return np.tile(arr, (n, 1))
            if len(arr) == n:            # per-borrower, same across dims
                return np.tile(arr.reshape(-1, 1), (1, K))
            raise ValueError(
                f"1D betas length {len(arr)} matches neither K={K} nor n={n}."
            )
        if arr.shape == (n, K):
            return arr
        raise ValueError(f"betas shape {arr.shape} != (n={n}, K={K}).")

    @staticmethod
    def _compress_with_none(col: np.ndarray):
        """Map ids to 0..C-1 keeping negatives as -1 ('no factor'). Returns (codes, C)."""
        col = np.asarray(col)
        out = np.full(len(col), -1, dtype=np.int64)
        valid = col >= 0
        if valid.any():
            uniq, inv = np.unique(col[valid], return_inverse=True)
            out[valid] = inv
            card = len(uniq)
        else:
            card = 0
        return out, card


__all__ = ["MultiFactorCopula", "MultiFactorCopulaParams"]
