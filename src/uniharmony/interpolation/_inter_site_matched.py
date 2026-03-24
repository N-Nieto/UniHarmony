"""Provide InterSiteMatchedInterpolation transformer."""

import itertools
import warnings
from typing import Any, Literal

import numpy as np
import structlog
from imblearn.base import SamplerMixin
from numpy.typing import ArrayLike, NDArray
from sklearn.base import BaseEstimator
from sklearn.utils import check_random_state
from sklearn.utils.validation import check_array, check_X_y

from uniharmony.interpolation._utils import (
    sites_sanity_checks,
    validate_covariates,
)


logger = structlog.get_logger()


class InterSiteMatchedInterpolation(SamplerMixin, BaseEstimator):
    """Inter-Site Matched Interpolation (ISMI) Harmonization.

    Parameters
    ----------
    alpha : float or tuple[float, float], default=0.3
        Interpolation weight(s). If float, constant. If tuple (min, max),
        sampled uniformly. Must be in [0, 1].
    target_tolerance : float or None, default=None
        Tolerance for target matching. None means exact match for classification,
        1% of range for regression.
    covariate_tolerance : ArrayLike or None, default=None
        Tolerance for continuous covariates. None means exact match.
    k : int or "max" or "average", default=1
        Number of matches: int for specific count, "max" for all, "average"
        to interpolate with mean of matches.
    mode : "pairwise" or "base_to_others", default="pairwise"
        Interpolation mode. "base_to_others" forces k="average".
    concatenate : bool, default=True
        If True, return original + synthetic. If False, synthetic only.
    random_state : int or RandomState or None, default=None
        Random seed for reproducibility.
    verbose : bool, default=False
        If True, log progress information.

    Attributes
    ----------
    sites_resampled_ : ndarray
        Site labels for output samples.
    unmatched_samples_ : dict
        Count of unmatched samples per direction.
    alpha_min_, alpha_max_ : float
        Validated alpha range.
    k_ : int or str
        Validated k parameter.
    use_average_ : bool
        Whether using average mode.

    """

    def __init__(
        self,
        alpha: float | tuple[float, float] = 0.3,
        target_tolerance: float | None = None,
        covariate_tolerance: ArrayLike | None = None,
        k: int | Literal["max", "average"] = 1,
        mode: Literal["pairwise", "base_to_others"] = "pairwise",
        *,
        concatenate: bool = True,
        random_state: int | np.random.RandomState | None = None,
        verbose: bool = False,
    ) -> None:
        self.alpha = alpha
        self.target_tolerance = target_tolerance
        self.covariate_tolerance = covariate_tolerance
        self.k = k
        self.mode = mode
        self.concatenate = concatenate
        self.random_state = random_state
        self.verbose = verbose

        # Validate parameters immediately (sklearn convention allows basic validation)
        self._validate_init_params()

    def _validate_init_params(self) -> None:
        """Validate constructor parameters."""
        # Validate alpha
        if isinstance(self.alpha, (int, float)):
            alpha_val = float(self.alpha)
            if not (0 <= alpha_val <= 1):
                raise ValueError(f"alpha={self.alpha} outside [0, 1]")
        elif isinstance(self.alpha, (tuple, list)) and len(self.alpha) == 2:
            a_min, a_max = self.alpha
            if not isinstance(a_min, (int, float)) or not isinstance(a_max, (int, float)):
                raise ValueError(f"alpha must be float or tuple (min, max), got {self.alpha}")
            a_min, a_max = float(a_min), float(a_max)
            if not (0 <= a_min <= a_max <= 1):
                raise ValueError(f"alpha must satisfy 0 <= min <= max <= 1, got {self.alpha}")
        else:
            raise ValueError(f"alpha must be float or tuple (min, max), got {self.alpha}")

        # Validate k
        if isinstance(self.k, str):
            k_lower = self.k.lower()
            if k_lower not in ("max", "average"):
                raise ValueError(f"k must be int >= 1, 'max', or 'average', got '{self.k}'")
        else:
            if int(self.k) < 1:
                raise ValueError(f"k must be >= 1, got {self.k}")

    def fit_resample(
        self,
        X: ArrayLike,
        y: ArrayLike,
        sites: ArrayLike,
        *,
        categorical_covariate: ArrayLike | None = None,
        continuous_covariate: ArrayLike | None = None,
    ) -> tuple[NDArray[np.float64], NDArray[Any]]:
        """Fit and resample using cross-site matched interpolation.

        Returns
        -------
        X_resampled : ndarray of shape (n_samples_new, n_features)
        y_resampled : ndarray of shape (n_samples_new,)

        """
        # Validate inputs
        X_arr, y_arr, sites_arr, cat_cov, cont_cov = self._validate_inputs(
            X, y, sites, categorical_covariate, continuous_covariate
        )

        # Initialize tracking
        self.unmatched_samples_: dict[tuple[Any, Any], int] = {}
        self._unique_sites = np.unique(sites_arr)
        self._n_sites = len(self._unique_sites)

        if self._n_sites < 2:
            raise ValueError(f"Need at least 2 sites, got {self._n_sites}")

        # Setup parameters
        self.random_state_ = check_random_state(self.random_state)
        self._setup_parameters()

        self._log_configuration(cat_cov, cont_cov)

        # Generate samples
        synthetic_X, synthetic_y, synthetic_sites = self._generate_samples(X_arr, y_arr, sites_arr, cat_cov, cont_cov)

        # Combine results
        if synthetic_X:
            if self.concatenate:
                X_out = np.vstack([X_arr, *synthetic_X])
                y_out = np.concatenate([y_arr, *synthetic_y])
                sites_out = np.concatenate([sites_arr, *synthetic_sites])
            else:
                X_out = np.vstack(synthetic_X)
                y_out = np.concatenate(synthetic_y)
                sites_out = np.concatenate(synthetic_sites)
        else:
            X_out, y_out, sites_out = X_arr, y_arr, sites_arr

        self.sites_resampled_ = sites_out

        if self.verbose:
            self._log_completion(y_arr, synthetic_y)

        return X_out, y_out

    def _validate_inputs(
        self,
        X: ArrayLike,
        y: ArrayLike,
        sites: ArrayLike,
        categorical_covariate: ArrayLike | None,
        continuous_covariate: ArrayLike | None,
    ) -> tuple[
        NDArray[np.float64],
        NDArray[Any],
        NDArray[Any],
        NDArray[Any] | None,
        NDArray[np.float64] | None,
    ]:
        """Validate and convert input arrays."""
        X_arr, y_arr = check_X_y(X, y)
        sites_arr = check_array(sites, ensure_2d=False, dtype=None)

        # Wrap sites_sanity_checks to match expected error message
        try:
            sites_sanity_checks(X_arr, sites_arr)
        except ValueError as e:
            if "At least two sites required" in str(e):
                raise ValueError(f"Need at least 2 sites, got {len(np.unique(sites_arr))}") from e
            raise

        # Check for NaN in categorical covariates (check_array with dtype=object may miss NaN)
        if categorical_covariate is not None:
            cat_arr = np.asarray(categorical_covariate)
            # Check for NaN in object arrays
            for i in range(cat_arr.shape[0]):
                for j in range(cat_arr.shape[1]):
                    val = cat_arr[i, j]
                    if isinstance(val, float) and np.isnan(val):
                        raise ValueError("Input contains NaN, infinity or a value too large for dtype('float64').")

        cat_cov, cont_cov, cov_tol = validate_covariates(
            X_arr.shape[0],
            categorical_covariate,
            continuous_covariate,
            self.covariate_tolerance,
            allow_nan=False,
        )

        self.covariate_tolerance_ = cov_tol
        self._problem_type = "classification" if y_arr.dtype.kind in "biu" else "regression"

        # Set default target_tolerance for regression if not specified
        if self._problem_type == "regression" and self.target_tolerance is None:
            y_range = np.ptp(y_arr)
            self.target_tolerance_ = y_range * 0.1 if y_range > 0 else 1.0
        else:
            self.target_tolerance_ = self.target_tolerance

        return X_arr, y_arr, sites_arr, cat_cov, cont_cov

    def _setup_parameters(self) -> None:
        """Validate and setup alpha, k parameters."""
        # Alpha already validated in __init__, just set attributes
        if isinstance(self.alpha, (int, float)):
            self.alpha_min_ = self.alpha_max_ = float(self.alpha)
        else:
            a_min, a_max = self.alpha
            self.alpha_min_, self.alpha_max_ = float(a_min), float(a_max)

        # Validate k
        if isinstance(self.k, str):
            k_lower = self.k.lower()
            self.k_ = k_lower
            self.use_average_ = k_lower == "average"
        else:
            self.k_ = int(self.k)
            self.use_average_ = False

        # Force average mode for base_to_others
        if self.mode == "base_to_others" and not self.use_average_:
            logger.warning("mode='base_to_others' requires k='average', overriding")
            self.k_ = "average"
            self.use_average_ = True

    def _log_configuration(
        self,
        cat_cov: NDArray[Any] | None,
        cont_cov: NDArray[np.float64] | None,
    ) -> None:
        """Log setup information."""
        if not self.verbose:
            return

        n_pairs = self._n_sites * (self._n_sites - 1) // 2
        alpha_str = (
            f"constant={self.alpha_min_}"
            if self.alpha_min_ == self.alpha_max_
            else f"uniform[{self.alpha_min_}, {self.alpha_max_}]"
        )

        logger.info(f"[ISMI] Mode: {self.mode}")
        logger.info(f"[ISMI] Sites: {self._unique_sites} ({self._n_sites} sites, {n_pairs} pairs)")
        logger.info(f"[ISMI] Alpha: {alpha_str}")

        if self.use_average_:
            logger.info("[ISMI] Behavior: average across all matches")
        elif self.k_ == "max":
            logger.info("[ISMI] Behavior: use all matches")
        else:
            logger.info(f"[ISMI] Behavior: k={self.k_} matches per sample")

        logger.info(f"[ISMI] Target tolerance: {self.target_tolerance_}")
        if cat_cov is not None:
            logger.info(f"[ISMI] Categorical covariates: {cat_cov.shape[1]}")
        if cont_cov is not None:
            tol_str = f" (tol: {self.covariate_tolerance_})" if self.covariate_tolerance_ is not None else ""
            logger.info(f"[ISMI] Continuous covariates: {cont_cov.shape[1]}{tol_str}")

    def _log_completion(
        self,
        y_orig: NDArray[Any],
        synthetic_y: list[NDArray[Any]],
    ) -> None:
        """Log completion statistics."""
        n_orig = len(y_orig)
        n_synth = sum(len(s) for s in synthetic_y)
        n_unmatched = sum(self.unmatched_samples_.values())
        increase = (n_synth / n_orig * 100) if n_orig > 0 else 0.0

        logger.info(
            f"[ISMI] Complete: {n_orig} original + {n_synth} synthetic "
            f"= {n_orig + n_synth} total ({increase:.1f}% increase), "
            f"{n_unmatched} unmatched"
        )

    def _generate_samples(
        self,
        X: NDArray[np.float64],
        y: NDArray[Any],
        sites: NDArray[Any],
        cat_cov: NDArray[Any] | None,
        cont_cov: NDArray[np.float64] | None,
    ) -> tuple[
        list[NDArray[np.float64]],
        list[NDArray[Any]],
        list[NDArray[Any]],
    ]:
        """Generate synthetic samples."""
        if self.mode == "pairwise":
            return self._generate_pairwise(X, y, sites, cat_cov, cont_cov)
        return self._generate_base_to_others(X, y, sites, cat_cov, cont_cov)

    def _generate_pairwise(
        self,
        X: NDArray[np.float64],
        y: NDArray[Any],
        sites: NDArray[Any],
        cat_cov: NDArray[Any] | None,
        cont_cov: NDArray[np.float64] | None,
    ) -> tuple[list[NDArray[np.float64]], list[NDArray[Any]], list[NDArray[Any]]]:
        """Generate samples for all site pairs."""
        all_X: list[NDArray[np.float64]] = []
        all_y: list[NDArray[Any]] = []
        all_sites: list[NDArray[Any]] = []

        pairs = list(itertools.combinations(self._unique_sites, 2))

        if self.verbose:
            logger.info(f"[ISMI] Processing {len(pairs)} pairs")

        for s1, s2 in pairs:
            m1 = sites == s1
            m2 = sites == s2

            X1, y1 = X[m1], y[m1]
            X2, y2 = X[m2], y[m2]
            c1 = cat_cov[m1] if cat_cov is not None else None
            c2 = cat_cov[m2] if cat_cov is not None else None
            v1 = cont_cov[m1] if cont_cov is not None else None
            v2 = cont_cov[m2] if cont_cov is not None else None

            if self.verbose:
                logger.info(f"[ISMI] Pair: {s1} ({len(X1)}) ↔ {s2} ({len(X2)})")

            # Forward: s1 → s2
            matches_1to2 = _find_matches(y1, y2, c1, c2, v1, v2, self.target_tolerance_, self.covariate_tolerance_)
            X_synth_1, y_synth_1, n_un_1 = self._interpolate(X1, y1, X2, y2, matches_1to2, s1, self.alpha_min_, self.alpha_max_)
            self.unmatched_samples_[(s1, s2)] = n_un_1

            # Reverse: s2 → s1 (with reversed alpha)
            matches_2to1 = _reverse_matches(matches_1to2, len(X2))
            rev_min, rev_max = 1.0 - self.alpha_max_, 1.0 - self.alpha_min_
            X_synth_2, y_synth_2, n_un_2 = self._interpolate(X2, y2, X1, y1, matches_2to1, s2, rev_min, rev_max)
            self.unmatched_samples_[(s2, s1)] = n_un_2

            # Collect
            if X_synth_1.size > 0:
                all_X.append(X_synth_1)
                all_y.append(y_synth_1)
                all_sites.append(np.full(len(y_synth_1), s1))

            if X_synth_2.size > 0:
                all_X.append(X_synth_2)
                all_y.append(y_synth_2)
                all_sites.append(np.full(len(y_synth_2), s2))

            if self.verbose:
                total = len(y_synth_1) + len(y_synth_2)
                logger.info(f"[ISMI]   Generated {total} samples ({n_un_1 + n_un_2} unmatched)")

        return all_X, all_y, all_sites

    def _generate_base_to_others(
        self,
        X: NDArray[np.float64],
        y: NDArray[Any],
        sites: NDArray[Any],
        cat_cov: NDArray[Any] | None,
        cont_cov: NDArray[np.float64] | None,
    ) -> tuple[list[NDArray[np.float64]], list[NDArray[Any]], list[NDArray[Any]]]:
        """Generate samples for each site vs all others."""
        all_X: list[NDArray[np.float64]] = []
        all_y: list[NDArray[Any]] = []
        all_sites: list[NDArray[Any]] = []

        for base_site in self._unique_sites:
            base_mask = sites == base_site
            other_mask = ~base_mask

            X_base, y_base = X[base_mask], y[base_mask]
            X_other, y_other = X[other_mask], y[other_mask]

            c_base = cat_cov[base_mask] if cat_cov is not None else None
            c_other = cat_cov[other_mask] if cat_cov is not None else None
            v_base = cont_cov[base_mask] if cont_cov is not None else None
            v_other = cont_cov[other_mask] if cont_cov is not None else None

            if self.verbose:
                logger.info(f"[ISMI] {base_site} ({len(X_base)}) → others ({len(X_other)})")

            matches = _find_matches(
                y_base, y_other, c_base, c_other, v_base, v_other, self.target_tolerance_, self.covariate_tolerance_
            )
            X_synth, y_synth, n_un = self._interpolate(
                X_base, y_base, X_other, y_other, matches, base_site, self.alpha_min_, self.alpha_max_
            )
            self.unmatched_samples_[(base_site, "others")] = n_un

            if X_synth.size > 0:
                all_X.append(X_synth)
                all_y.append(y_synth)
                all_sites.append(np.full(len(y_synth), base_site))

            if self.verbose:
                logger.info(f"[ISMI]   Generated {len(y_synth)} samples ({n_un} unmatched)")

        return all_X, all_y, all_sites

    def _interpolate(
        self,
        X_src: NDArray[np.float64],
        y_src: NDArray[Any],
        X_dst: NDArray[np.float64],
        y_dst: NDArray[Any],
        matches: list[list[int]],
        src_site: Any,
        alpha_min: float,
        alpha_max: float,
    ) -> tuple[NDArray[np.float64], NDArray[Any], int]:
        """Generate synthetic samples from matches."""
        n_src = len(X_src)

        if n_src == 0 or len(X_dst) == 0:
            return np.array([]).reshape(0, X_src.shape[1] if X_src.size > 0 else 0), np.array([]), n_src

        # Find matched/unmatched
        has_match = np.array([len(m) > 0 for m in matches])
        n_unmatched = n_src - int(np.sum(has_match))
        matched_idx = np.where(has_match)[0]

        if len(matched_idx) == 0:
            return np.array([]).reshape(0, X_src.shape[1]), np.array([]), n_unmatched

        # Sample alphas
        n_matched = len(matched_idx)
        if alpha_min == alpha_max:
            alphas = np.full(n_matched, alpha_min)
        else:
            alphas = self.random_state_.uniform(alpha_min, alpha_max, n_matched)

        # Dispatch to appropriate method
        if self.use_average_:
            X_synth, y_synth = self._interp_average(X_src, y_src, X_dst, y_dst, matches, matched_idx, alphas)

        elif self.k_ == "max":
            X_synth, y_synth = self._interp_max(X_src, y_src, X_dst, y_dst, matches, matched_idx, alphas)

        else:
            X_synth, y_synth = self._interp_k(X_src, y_src, X_dst, y_dst, matches, matched_idx, alphas, self.k_)

        return X_synth, y_synth, n_unmatched

    def _interp_average(
        self,
        X_src: NDArray[np.float64],
        y_src: NDArray[Any],
        X_dst: NDArray[np.float64],
        y_dst: NDArray[Any],
        matches: list[list[int]],
        matched_idx: NDArray[np.int64],
        alphas: NDArray[np.float64],
    ) -> tuple[NDArray[np.float64], NDArray[Any]]:
        """Interpolate to average of all matches."""
        X_m = X_src[matched_idx]
        y_m = y_src[matched_idx]

        # Average of destinations for each source
        X_avg = np.array([np.mean(X_dst[matches[i]], axis=0) for i in matched_idx])

        X_synth = X_m + alphas[:, np.newaxis] * (X_avg - X_m)

        if self._problem_type == "classification":
            y_synth = y_m
        else:
            y_avg = np.array([np.mean(y_dst[matches[i]]) for i in matched_idx])
            y_synth = y_m + alphas * (y_avg - y_m)

        return X_synth, y_synth

    def _interp_max(
        self,
        X_src: NDArray[np.float64],
        y_src: NDArray[Any],
        X_dst: NDArray[np.float64],
        y_dst: NDArray[Any],
        matches: list[list[int]],
        matched_idx: NDArray[np.int64],
        alphas: NDArray[np.float64],
    ) -> tuple[NDArray[np.float64], NDArray[Any]]:
        """Interpolate to each match individually."""
        src_exp: list[int] = []
        dst_exp: list[int] = []
        alpha_exp: list[float] = []

        for idx, src_i in enumerate(matched_idx):
            for dst_j in matches[src_i]:
                src_exp.append(src_i)
                dst_exp.append(dst_j)
                alpha_exp.append(alphas[idx])

        src_arr = np.array(src_exp)
        dst_arr = np.array(dst_exp)
        alpha_arr = np.array(alpha_exp)

        X_s = X_src[src_arr]
        X_d = X_dst[dst_arr]
        X_synth = X_s + alpha_arr[:, np.newaxis] * (X_d - X_s)

        if self._problem_type == "classification":
            y_synth = y_src[src_arr]
        else:
            y_s = y_src[src_arr]
            y_d = y_dst[dst_arr]
            y_synth = y_s + alpha_arr * (y_d - y_s)

        return X_synth, y_synth

    def _interp_k(
        self,
        X_src: NDArray[np.float64],
        y_src: NDArray[Any],
        X_dst: NDArray[np.float64],
        y_dst: NDArray[Any],
        matches: list[list[int]],
        matched_idx: NDArray[np.int64],
        alphas: NDArray[np.float64],
        k: int,
    ) -> tuple[NDArray[np.float64], NDArray[Any]]:
        """Interpolate to k randomly selected matches."""
        src_exp: list[int] = []
        dst_exp: list[int] = []
        alpha_exp: list[float] = []

        for idx, src_i in enumerate(matched_idx):
            n_match = len(matches[src_i])
            if n_match < k:
                warnings.warn(
                    f"Sample has {n_match} matches but k={k} requested, using all",
                    UserWarning,
                    stacklevel=3,
                )
            n_use = min(k, n_match)
            selected = self.random_state_.choice(matches[src_i], size=n_use, replace=False)

            for dst_j in selected:
                src_exp.append(src_i)
                dst_exp.append(dst_j)
                alpha_exp.append(alphas[idx])

        src_arr = np.array(src_exp)
        dst_arr = np.array(dst_exp)
        alpha_arr = np.array(alpha_exp)

        X_s = X_src[src_arr]
        X_d = X_dst[dst_arr]
        X_synth = X_s + alpha_arr[:, np.newaxis] * (X_d - X_s)

        if self._problem_type == "classification":
            y_synth = y_src[src_arr]
        else:
            y_s = y_src[src_arr]
            y_d = y_dst[dst_arr]
            y_synth = y_s + alpha_arr * (y_d - y_s)

        return X_synth, y_synth

    def _fit_resample(self, X: ArrayLike, y: ArrayLike, **params: Any) -> None:
        """For SamplerMixin compatibility."""
        pass


def _find_matches(
    y_src: NDArray[Any],
    y_dst: NDArray[Any],
    cat_src: NDArray[Any] | None,
    cat_dst: NDArray[Any] | None,
    cont_src: NDArray[np.float64] | None,
    cont_dst: NDArray[np.float64] | None,
    target_tol: float | None,
    cov_tol: NDArray[np.float64] | None,
) -> list[list[int]]:
    """Find matches with target priority."""
    n_src = len(y_src)

    # Target match
    if target_tol is None or target_tol == 0:
        match = y_src[:, np.newaxis] == y_dst[np.newaxis, :]
    else:
        match = np.abs(y_src[:, np.newaxis] - y_dst[np.newaxis, :]) <= target_tol

    # Categorical match
    if cat_src is not None and cat_dst is not None:
        for c in range(cat_src.shape[1]):
            match &= cat_src[:, c][:, np.newaxis] == cat_dst[:, c][np.newaxis, :]

    # Continuous match
    if cont_src is not None and cont_dst is not None and cov_tol is not None:
        for c in range(cont_src.shape[1]):
            diff = np.abs(cont_src[:, c][:, np.newaxis] - cont_dst[:, c][np.newaxis, :])
            match &= diff <= cov_tol[c]

    return [np.where(match[i])[0].tolist() for i in range(n_src)]


def _reverse_matches(matches_fwd: list[list[int]], n_dst: int) -> list[list[int]]:
    """Reverse match list direction."""
    rev: list[list[int]] = [[] for _ in range(n_dst)]
    for src_i, dst_list in enumerate(matches_fwd):
        for dst_j in dst_list:
            rev[dst_j].append(src_i)
    return rev
