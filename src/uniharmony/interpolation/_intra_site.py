"""Intra-site interpolation-based harmonization.

This module provides the ``IntraSiteInterpolation`` transformer, a sampler
designed to mitigate site-induced bias by enforcing class balance within each
site independently.

Key features
------------
- Site-wise class balancing using interpolation-based oversampling.
- Optional stratification via categorical and/or continuous covariates.
- Support for both classification and regression problems.
- Regression targets are discretized into bins for balancing purposes.
- Compatible with imbalanced-learn samplers.

Design principles
-----------------
- Preserve covariate distributions when requested.
- Guarantee exact class balance per site (or globally).
- Provide robust fallbacks when interpolation is insufficient.
"""

from collections import Counter
from typing import Literal

import numpy as np
import numpy.typing as npt
import structlog
from imblearn.base import SamplerMixin
from sklearn.base import BaseEstimator, clone
from sklearn.utils import Tags, check_random_state
from sklearn.utils.validation import (
    check_array,
    check_consistent_length,
    check_X_y,
)

from uniharmony._utils import validate_sites
from uniharmony.interpolation._utils import (
    create_interpolator,
    validate_class_representation,
    validate_covariates,
)


__all__ = ["IntraSiteInterpolation"]

logger = structlog.get_logger()


class IntraSiteInterpolation(SamplerMixin, BaseEstimator):
    """Intra-Site Interpolation (ISI) Harmonization.

    This sampler performs **site-wise class balancing** to reduce spurious
    correlations between site membership and class labels.

    For each site independently:
    - The target class count is determined by ``balance_strategy``.
    - All minority classes are oversampled to match the target count.
    - Any imblearn-compatible oversampling strategy may be used.
    - Alternatively, all classes in the smaller sites are oversampled to matched the biggest site.

    When covariates are provided, balancing is performed within each
    covariate stratum (unique combination of covariate values) within
    each site, preserving the joint distribution of covariates and
    target labels.

    For regression tasks, the continuous target is binned into discrete
    intervals and each bin is treated as a class for balancing purposes.

    Parameters
    ----------
    interpolator : str or SamplerMixin instance, optional (default "smote")
        The interpolator to use. Can be a str specifying a built-in method or
        an instance of SamplerMixin.
        Supported str methods are:

          - "smote": Synthetic Minority Over-sampling Technique
          - "borderline-smote": Borderline-SMOTE
          - "svm-smote": SVM-SMOTE
          - "adasyn": Adaptive Synthetic Sampling
          - "kmeans-smote": KMeans-SMOTE
          - "random": Random Over-Sampling

    interpolator_kwargs : dict or None, optional (default None)
        Additional keyword arguments passed to ``interpolator``.

    random_state : int or RandomState instance or None, optional (default None)
        The seed of the pseudo random number generator or RandomState for
        reproducibility.

    balance_strategy : {"per_site", "global_max"}, optional (default "per_site")
        Strategy to determine the target count for oversampling:

        - "per_site": Each site is balanced independently to its own majority
          class count.
        - "global_max": All sites are balanced to the global maximum class
          count across all sites. Both minority and majority classes are samples
          to match the N for the majority class across sites.

    n_bins : int or None, optional (default 10)
        Number of bins for regression target binning.

    binning_strategy : {"uniform", "quantile"}, optional (default "quantile")
        Strategy for creating bins when the task is regression:

        - "uniform": Bins of equal width covering the target range.
        - "quantile": Bins with approximately equal number of samples.

    task : {"auto", "classification", "regression"}, optional (default "auto")
        Task type. If ``"auto"``, inferred from ``y`` dtype (integer types
        imply classification, floating types imply regression).
        A regression problem is treated as a multi-class classification problem.

    Attributes
    ----------
    sites_resampled_ : ndarray of shape (n_samples_new,)
        Site identifiers for the resampled dataset.

    samples_created_ : dict
        A nested dictionary mapping ``{site: {class_label: n_created}}``,
        where ``n_created`` is the number of synthetic samples generated
        for that class in that site. For regression, ``class_label`` is
        the bin index.

    target_count_ : int or None
        The target sample count per class used for balancing. Set to the
        global maximum when ``balance_strategy="global_max"``, otherwise
        ``None`` (targets are per-site).

    bins_ : ndarray or None
        Bin edges used for regression target binning. ``None`` for
        classification tasks.

    task_ : str
        Inferred or specified task type ("classification" or "regression").

    """

    def __init__(
        self,
        interpolator: str
        | Literal["smote", "borderline-smote", "svm-smote", "adasyn", "kmeans-smote", "random"]
        | SamplerMixin = "smote",
        interpolator_kwargs: dict | None = None,
        random_state: int | np.random.RandomState | None = None,
        balance_strategy: str | Literal["per_site", "global_max"] = "per_site",
        n_bins: int | None = 10,
        binning_strategy: str | Literal["uniform", "quantile"] = "quantile",
        task: str | Literal["auto", "classification", "regression"] = "auto",
    ) -> None:
        self.interpolator = interpolator
        self.interpolator_kwargs = interpolator_kwargs
        self.random_state = random_state
        self.balance_strategy = balance_strategy
        self.n_bins = n_bins
        self.binning_strategy = binning_strategy
        self.task = task

    def fit_resample(
        self,
        X: npt.ArrayLike,
        y: npt.ArrayLike,
        sites: npt.ArrayLike,
        *,
        categorical_covariate: npt.ArrayLike | None = None,
        continuous_covariate: npt.ArrayLike | None = None,
        n_bins_cont_cov: int | None = None,
        binning_strategy_cont_cov: str | Literal["uniform", "quantile"] = "quantile",
    ) -> tuple[npt.NDArray, npt.NDArray]:
        """Fit and resample the dataset using site-wise interpolation.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Feature matrix containing the input samples.

        y : array-like of shape (n_samples,)
            Target values. Integer labels for classification, continuous
            values for regression.

        sites : array-like of shape (n_samples,)
            Site or domain identifiers indicating the origin of each sample.
            Resampling is performed independently within each site.

        categorical_covariate : array-like of shape (n_samples, n_categorical), default=None
            Categorical covariates used for stratified balancing. When
            provided, classes are balanced within each unique covariate
            combination within each site.

        continuous_covariate : array-like of shape (n_samples, n_continuous), default=None
            Continuous covariates used for stratified balancing. Samples
            are grouped by approximate matching within ``covariate_tolerance``.

        covariate_tolerance : array-like of shape (n_continuous,), default=None
            Maximum allowed absolute difference for continuous covariate
            grouping. Must have one value per continuous covariate column.
            If ``None``, exact matching is required.

        n_bins_cont_cov : int or None, default=None
            Number of bins to use for continuous covariates when creating
            groups. If None, no binning is applied and exact matching is
            used for grouping.

        binning_strategy_cont_cov : {"uniform", "quantile"}, default="quantile"
            Strategy for binning continuous covariates when creating groups:
            - "uniform": Bins of equal width covering the covariate range.
            - "quantile": Bins with approximately equal number of samples.

        Returns
        -------
        X_resampled : numpy.ndarray of shape (n_samples_new, n_features)
            The feature matrix after site-wise oversampling.

        y_resampled : numpy.ndarray of shape (n_samples_new,)
            The corresponding targets after resampling.

        Raises
        ------
        ValueError
            If ``X``, ``y``, and ``sites`` have incompatible shapes, if fewer
            than two unique sites are present, if any site is missing any
            class, or if ``balance_strategy`` is invalid.


        Notes
        -----
        Sites can be retrieved from IntraSiteInterpolation.sites_resampled_

        """
        logger.info("[ISI] Starting fit_resample")

        X, y, sites, y_binnarized, cat_cov, cont_cov = self._validate_input(
            X,
            y,
            sites,
            categorical_covariate,
            continuous_covariate,
        )

        interpolator_template = self._resolve_interpolator()
        self.random_state = check_random_state(self.random_state)

        # For continuos covariates
        if continuous_covariate is not None:
            if n_bins_cont_cov is None:
                raise ValueError("n_bins_cont_cov must be provided when continuous_covariate are also provided.")
            if binning_strategy_cont_cov not in ["uniform", "quantile"]:
                raise ValueError("binning_strategy_cont_cov must be 'uniform' or 'quantile'")
        self.n_bins_cont_cov = n_bins_cont_cov
        self.binning_strategy_cont_cov = binning_strategy_cont_cov

        unique_sites = np.unique(sites)
        # Use y_binnarized, as this already contains the information of the classes if task="regression"
        unique_classes = np.unique(y_binnarized)

        # Global target
        if self.balance_strategy == "global_max":
            self.target_count_ = max(
                np.sum((sites == site) & (y_binnarized == cls)) for site in unique_sites for cls in unique_classes
            )
            logger.debug(f"[ISI] N target for global_max strategy = {self.target_count_}")

        # Initialize variables
        X_out, y_out, sites_out = [], [], []
        self.samples_created_ = {}

        # Main loop, iterate over sites.
        for site in unique_sites:
            logger.info(f"[ISI] Processing site {site}")

            mask = sites == site
            # Get site data
            Xs, ys, yw = X[mask], y[mask], y_binnarized[mask]
            # Get site's covariates
            cat_s = cat_cov[mask] if cat_cov is not None else None
            cont_s = cont_cov[mask] if cont_cov is not None else None

            # Check how many samples we need for each class. If `balance_strategy` = global_max, then use the global N.
            target_N = max(Counter(yw).values()) if self.balance_strategy == "per_site" else self.target_count_
            logger.debug(f"[ISI] For site {site}, N target for per_site strategy = {target_N}")

            Xr, yr = self._resample_site(
                Xs,
                ys,
                yw,
                target_N,
                interpolator_template,
                cat_s,
                cont_s,
            )

            # Check how many samples were created in the site for each class.
            self.samples_created_[site] = {
                c: max(
                    0,
                    np.sum((self._bin_target(yr)[0] if self.task_ == "regression" else yr) == c) - np.sum(yw == c),
                )
                for c in unique_classes
            }

            X_out.append(Xr)
            y_out.append(yr)
            sites_out.append(np.full(len(Xr), site))

        self.sites_resampled_ = np.concatenate(sites_out)

        return np.vstack(X_out), np.concatenate(y_out)

    # ------------------------------------------------------------------ #
    # Validation
    # ------------------------------------------------------------------ #
    def _validate_input(
        self,
        X: npt.ArrayLike,
        y: npt.ArrayLike,
        sites: npt.ArrayLike,
        categorical_covariate: npt.ArrayLike | None,
        continuous_covariate: npt.ArrayLike | None,
    ) -> tuple[
        npt.NDArray,
        npt.NDArray,
        npt.NDArray,
        npt.NDArray,
        npt.NDArray | None,
        npt.NDArray | None,
    ]:
        """Validate and preprocess all inputs for the resampling pipeline.

        Performs comprehensive input validation, task inference, and covariate
        preprocessing. Ensures all inputs are consistent and properly formatted
        before resampling begins.

        Parameters
        ----------
        X : npt.ArrayLike
            Feature matrix (shape: n_samples x n_features).
        y : npt.ArrayLike
            Target labels or values (shape: n_samples,).
        sites : npt.ArrayLike
            Site identifiers for multi-site data (shape: n_samples,).
        categorical_covariate : npt.ArrayLike or None
            Categorical covariates matrix (shape: n_samples x n_cat_features).
        continuous_covariate : npt.ArrayLike or None
            Continuous covariates matrix (shape: n_samples x n_cont_features).

        Returns
        -------
        tuple
            - X: Validated feature matrix (ndarray)
            - y: Validated target array (ndarray)
            - sites: Validated site identifiers (ndarray)
            - y_binnarized: Target array (binned for regression, original for classification)
            - cat_cov: Validated categorical covariates (ndarray or None)
            - cont_cov: Validated continuous covariates (ndarray or None)

        Raises
        ------
        ValueError
            If regression task is detected but n_bins is not provided.
            If site identifiers are invalid (via validate_sites).
            If class representation is insufficient (via validate_class_representation).
            If balance_strategy is not valid.

        Notes
        -----
        Validation steps performed:
            1. Check X and y for consistency and missing values
            2. Validate site identifiers
            3. Ensure all arrays have consistent lengths
            4. Process and validate covariates
            5. Infer task type (classification vs regression)
            6. Bin targets for regression tasks
            7. Validate class distribution across sites

        """
        # Step 1: Validate feature matrix and target array
        # check_X_y ensures no NaN/inf values, consistent shapes, and proper dtypes
        X, y = check_X_y(X, y, estimator=self)

        # Step 2: Validate site identifiers
        # Convert to array, handle 1D data properly, ensure no unexpected dimensions
        sites = check_array(sites, dtype=None, ensure_2d=False, estimator=self)

        # Step 3: Ensure all primary inputs have the same number of samples
        # Critical for downstream operations that assume alignment
        check_consistent_length(X, y, sites)

        # Step 4: Validate site identifiers format (e.g., no empty strings, valid types)
        validate_sites(sites)

        # Step 5: Process and validate covariates
        # This handles:
        #   - Converting to numpy arrays
        #   - Checking consistency with X shape
        #   - Validating tolerance values match continuous covariates
        #   - Handling None/empty inputs appropriately
        cat_cov, cont_cov, _ = validate_covariates(
            X.shape[0],  # n_samples: ensure covariates match sample count
            categorical_covariate,
            continuous_covariate,
            None,
            allow_nan=True,  # Allow missing values in covariates (common in real-world data)
        )

        # Step 6: Infer task type (classification vs regression) from target data
        # Sets self.task_ for use throughout the resampling process
        self.task_ = self._infer_task(y)

        # Step 7: Handle regression-specific preprocessing
        if self.task_ == "regression":
            # Convert continuous targets to bin indices
            # Stores bin edges in self.bins_ for later use (e.g., inverse transform)
            y_binnarized, self.bins_ = self._bin_target(y)
        else:
            # Classification: use original labels as-is
            y_binnarized = y
            self.bins_ = None  # No bins needed for classification

        # Step 8: Validate class representation across sites
        # Ensures each site has sufficient samples of each class for resampling
        # Prevents errors during interpolation (e.g., SMOTE requires at least 2 samples)
        validate_class_representation(y_binnarized, sites)

        # Step 9: Validate balance_strategy
        if self.balance_strategy not in {"per_site", "global_max"}:
            raise ValueError("balance_strategy must be 'per_site' or 'global_max'")

        # Step 10: Return all validated and preprocessed data
        return X, y, sites, y_binnarized, cat_cov, cont_cov

    def _resolve_interpolator(self) -> SamplerMixin:
        """Create or validate an interpolator instance for resampling.

        Converts string identifiers to actual interpolator objects or validates
        that a provided interpolator is compatible with the resampling pipeline.

        Returns
        -------
        SamplerMixin
            Validated interpolator instance ready for resampling.

        Raises
        ------
        ValueError
            If interpolator is neither a string (valid name) nor a SamplerMixin instance.

        Notes
        -----
        Supported interpolator names (case-insensitive):
            - "smote": SMOTE (Synthetic Minority Over-sampling Technique)
            - "borderline-smote": BorderlineSMOTE (focuses on boundary samples)
            - "svm-smote": SVMSMOTE (uses SVM to identify support vectors)
            - "adasyn": ADASYN (adaptive synthetic sampling)
            - "kmeans-smote": KMeansSMOTE (clusters before oversampling)
            - "random": RandomOverSampler (simple random oversampling)

        Examples
        --------
        >>> # Using string identifier
        >>> resampler._resolve_interpolator()
        SMOTE(random_state=42)

        >>> # Using pre-instantiated interpolator
        >>> resampler.interpolator = SMOTE(random_state=42)
        >>> resampler._resolve_interpolator()
        SMOTE(random_state=42)

        """
        # Step 1: Initialize random number generator for reproducibility
        # This ensures consistent synthetic sample generation across runs
        # Handles both integer seeds and RandomState objects
        random_state = check_random_state(self.random_state)

        # Step 2: Handle string-based interpolator specification
        if isinstance(self.interpolator, str):
            # Create interpolator from registered name
            # Example: "smote" -> SMOTE(random_state=42)
            self.interpolator = create_interpolator(
                name=self.interpolator,  # Interpolator name (e.g., "smote", "adasyn")
                random_state=random_state,  # Ensures reproducible synthetic samples
                **(self.interpolator_kwargs or {}),  # Additional parameters (e.g., k_neighbors=5)
            )
            return self.interpolator

        # Step 3: Handle pre-instantiated interpolator objects
        if isinstance(self.interpolator, SamplerMixin):
            # Verify the interpolator implements the required interface
            # SamplerMixin ensures fit_resample method exists and follows conventions
            # This allows users to pass custom interpolators that follow the API
            return self.interpolator

        # Step 4: Invalid interpolator configuration
        # Neither a recognized string name nor a compatible instance
        raise ValueError(
            f"Invalid interpolator: {self.interpolator}. Must be a string (e.g., 'smote', 'adasyn') or a SamplerMixin instance."
        )

    # ------------------------------------------------------------------ #
    # Core function
    # ------------------------------------------------------------------ #

    def _resample_site(
        self,
        X: np.ndarray,
        y: np.ndarray,
        y_binnarized: np.ndarray,
        target_N: int,
        interpolator_template: SamplerMixin,
        cat: np.ndarray | None,
        cont: np.ndarray | None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Resample a single site to achieve balanced class distribution.

        This method handles resampling within a single site by grouping similar
        samples (based on categorical and continuous covariates) and applying
        interpolation to oversample minority classes within each group.

        Parameters
        ----------
        X : np.ndarray
            Feature matrix for the site (shape: n_samples x n_features).
        y : np.ndarray
            Target labels for the site (shape: n_samples,).
        y_binnarized : np.ndarray
            Working target labels (potentially binned for regression tasks,
            shape: n_samples,).
        target_N : int
            Target number of samples per class after resampling. For regression
            tasks, this is the target number per class in the binned target space.
        interpolator_template : SamplerMixin
            Clonable interpolator instance (e.g., SMOTE, ADASYN) that implements
            the fit_resample method.
        cat : np.ndarray or None
            Categorical covariate indices. If None, categorical grouping is disabled.
        cont : np.ndarray or None
            Continuous covariate indices. If None, continuous grouping is disabled.
        cov_tol : np.ndarray or None
            Tolerance values for continuous covariates when creating groups.
            Must have same length as `cont` if provided.

        Returns
        -------
        tuple[np.ndarray, np.ndarray]
            - X_resampled: Resampled feature matrix (shape: N x n_features)
            - y_resampled: Corresponding target labels (shape: N,)

        Notes
        -----
        The resampling strategy depends on `self.balance_strategy`:
            - "per_site": Each site is balanced independently to its majority class
            - Otherwise: All sites are balanced to the global `target_N`

        Groups are created using `_create_group_labels` to ensure samples with
        similar covariate patterns are resampled together, preserving local
        structure.

        """
        # Get the classes again.
        classes = np.unique(y_binnarized)

        # Step 1: Determine grouping strategy - no grouping if no covariates provided
        if cat is None and cont is None:
            # Single group containing all samples
            group_labels = np.zeros(len(X), dtype=int)
        else:
            # Create groups based on similarity in categorical/continuous covariates
            n_bins_cont_cov = self.n_bins_cont_cov
            binning_strategy_cont_cov = self.binning_strategy_cont_cov
            group_labels = self._create_group_labels(cat, cont, n_bins_cont_cov, binning_strategy_cont_cov)

        # Initialize containers for resampled data from all groups
        X_parts, y_parts = [], []

        # Step 2: Process each group independently
        for group in np.unique(group_labels):
            # Extract samples belonging to current group
            mask = group_labels == group
            Xg, yg, y_bin_g = X[mask], y[mask], y_binnarized[mask]

            # Skip empty groups (shouldn't happen, but defensive programming)
            if len(Xg) == 0:
                logger.warning(f"[ISI] Group {group} has no data, skipping interpolation.")
                continue

            # Step 3: Determine target samples per class for this group
            counts = Counter(y_bin_g)

            # Decide group-level target based on balance strategy
            if self.balance_strategy == "per_site":
                # Balance to the majority class count within this site
                group_target = max(counts.values())
            else:
                # Balance to global target (e.g., for cross-site comparison)
                group_target = target_N

            # Step 4: Identify classes that need oversampling
            sampling_strategy = {cls: group_target for cls in classes if counts.get(cls, 0) < group_target}

            # Step 5: Handle case where no oversampling is needed
            if not sampling_strategy:
                # Just take the first `group_target` samples from each class
                # (keeps class distribution but ensures all classes have equal size)
                for cls in classes:
                    mask_cls = y_bin_g == cls
                    if np.any(mask_cls):
                        X_parts.append(Xg[mask_cls][:group_target])
                        y_parts.append(yg[mask_cls][:group_target])
                    else:
                        logger.warning(f"[ISI] samples for class {cls}")
                continue  # Move to next group

            # Step 6: Apply interpolation for groups needing oversampling
            # Clone the template to avoid modifying the original
            interp = clone(interpolator_template)
            interp.set_params(sampling_strategy=sampling_strategy)

            # Generate synthetic samples for minority classes
            X_tmp, y_tmp_binnarized = interp.fit_resample(Xg, y_bin_g)

            y_tmp = self._reconstruct_continuous_y(
                y_bin_new=y_tmp_binnarized,
                y_orig=yg,  # original continuous values in this group
                y_bin_orig=y_bin_g,  # original bins in this group
            )
            # Step 7: Post-process each class to ensure exact group_target size
            for cls in classes:
                # Get samples of current class
                mask_cls = y_tmp_binnarized == cls
                X_cls, y_cls = X_tmp[mask_cls], y_tmp[mask_cls]

                # If we still have fewer samples than target, bootstrap with replacement
                if len(X_cls) < group_target:
                    n_needed = group_target - len(X_cls)
                    idx = np.random.choice(len(X_cls), n_needed, replace=True)
                    X_cls = np.vstack([X_cls, X_cls[idx]])
                    y_cls = np.concatenate([y_cls, y_cls[idx]])

                # Take exactly group_target samples (first N)
                X_parts.append(X_cls[:group_target])
                y_parts.append(y_cls[:group_target])

        # Step 8: Combine results from all groups and return
        return np.vstack(X_parts), np.concatenate(y_parts)

    # ------------------------------------------------------------------ #
    # Utilities
    # ------------------------------------------------------------------ #
    def _infer_task(self, y: np.ndarray) -> str:
        """Infer the machine learning task type from target data.

        Determines whether the problem is classification or regression based on
        either an explicitly set task or automatic inference from target dtype.

        Parameters
        ----------
        y : np.ndarray
            Target/label array (shape: n_samples,).

        Returns
        -------
        str
            Task type: either "classification" or "regression".

        Notes
        -----
        Task inference logic:
            - If self.task != "auto", return self.task (user explicitly specified)
            - Otherwise, infer from y.dtype.kind:
                * 'b' (boolean), 'i' (signed integer), 'u' (unsigned integer)
                -> "classification"
                * Any other dtype kind (float, complex, etc.) -> "regression"

        Examples
        --------
        >>> import numpy as np
        >>> obj.task = "auto"
        >>> obj._infer_task(np.array([0, 1, 0, 1]))  # integer labels
        'classification'
        >>> obj._infer_task(np.array([0.5, 1.2, 3.7]))  # float labels
        'regression'

        """
        # Priority 1: User explicitly specified task (override automatic inference)
        if self.task != "auto":
            return self.task

        # Priority 2: Automatic inference based on target data type
        # Classification: discrete labels (bool, int, uint)
        # Regression: continuous values (float, complex, etc.)
        return "classification" if y.dtype.kind in "biu" else "regression"

    def _bin_target(self, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Bin continuous target values into discrete categories for regression tasks.

        Transforms regression targets into bins to enable class-based sampling
        strategies (e.g., SMOTE) for regression problems.

        Parameters
        ----------
        y : np.ndarray
            Continuous target values (shape: n_samples,).

        Returns
        -------
        tuple[np.ndarray, np.ndarray]
            - yb: Binned target indices (0 to n_bins-1, shape: n_samples,)
            - bins: Bin edges used for discretization (shape: n_bins+1,)

        Notes
        -----
        Two binning strategies (controlled by self.binning_strategy):
            - "uniform": Equal-width bins between min(y) and max(y)
            - otherwise: Equal-frequency bins using quantiles

        The function uses np.digitize with bins[1:-1] to exclude out-of-range edges:
            - Values exactly at min(y) go to bin 0
            - Values exactly at max(y) go to bin n_bins-1

        """
        # Step 1: Define bin edges based on selected strategy
        if self.binning_strategy == "uniform":
            # Strategy A: Equal-width bins spanning the full data range
            # Example: y=[0, 10], n_bins=5 -> edges=[0, 2, 4, 6, 8, 10]
            bins = np.linspace(y.min(), y.max(), self.n_bins + 1)
        elif self.binning_strategy == "quantile":
            # Strategy B: Equal-frequency bins using quantiles
            # Ensures roughly equal number of samples per bin
            # Example: 4 bins -> edges at 0%, 25%, 50%, 75%, 100% quantiles
            bins = np.quantile(y, np.linspace(0, 1, self.n_bins + 1))
        else:
            raise ValueError(f"binning_strategy must be 'uniform' or 'quantile', got {self.binning_strategy}")

        # Step 2: Assign each target value to a bin index
        # bins[1:-1] excludes first and last edges to handle boundary values correctly
        # Values < bins[1] go to bin 0, values >= bins[-2] go to bin n_bins-1
        yb = np.digitize(y, bins[1:-1])

        # Step 3: Ensure all indices are valid (clip to [0, n_bins-1] range)
        # This handles edge cases where digitize might produce -1 or n_bins
        yb_clipped = np.clip(yb, 0, len(bins) - 2)

        return yb_clipped, bins

    def _create_group_labels(
        self,
        cat: np.ndarray | None,
        cont: np.ndarray | None,
        n_bins_cont_cov: int | None,
        binning_strategy_cont_cov: str = "quantile",
    ) -> np.ndarray:
        """Create group labels by combining categorical and continuous covariates.

        Groups samples into homogeneous subgroups based on covariate similarity.
        Continuous covariates are discretized using a binning strategy, similar
        to target binning in regression tasks.

        Parameters
        ----------
        cat : np.ndarray or None
            Categorical covariates of shape (n_samples, n_cat_features).
            Each column represents a categorical variable.

        cont : np.ndarray or None
            Continuous covariates of shape (n_samples, n_cont_features).
            Each column represents a continuous variable.

        n_bins_cont_cov : int or None
            Number of bins used to discretize continuous covariates.
            Required if `cont` is not None.

        binning_strategy_cont_cov : {"quantile", "uniform"}, default="quantile"
            Strategy used to bin continuous covariates:
            - "quantile": equal number of samples per bin
            - "uniform": equal width bins

        Returns
        -------
        np.ndarray
            Integer group labels of shape (n_samples,). Samples with the same label
            belong to the same covariate-defined group.

        Raises
        ------
        ValueError
            If `cont` is provided but `n_bins_cont_cov` is None or < 2.
            If `binning_strategy_cont_cov` is invalid.

        Notes
        -----
        Group construction:
            1. Categorical covariates → exact grouping via unique combinations
            2. Continuous covariates → discretized via binning
            3. Combined via mixed-radix encoding

        Examples
        --------
        >>> # Categorical: [[0],[0],[1]] → [0,0,1]
        >>> # Continuous: [1.2, 1.3, 5.7] with n_bins=2 → [0,0,1]

        """
        # ------------------------------------------------------------------
        # Determine number of samples
        # ------------------------------------------------------------------
        if cat is None and cont is None:
            raise ValueError("At least one of 'cat' or 'cont' must be provided.")

        n_samples = len(cat) if cat is not None else len(cont)

        # FIX: always compute both parts independently
        cat_labels = None
        cont_labels = None

        if cat is not None:
            _, cat_labels = np.unique(cat, axis=0, return_inverse=True)

        if cont is not None:
            cont_labels = self._resolve_continuous_covariate(cont, n_samples, n_bins_cont_cov, binning_strategy_cont_cov)

        # FIX: correct combination logic
        if cat_labels is not None and cont_labels is not None:
            return cat_labels * (cont_labels.max() + 1) + cont_labels
        elif cat_labels is not None:
            return cat_labels
        else:
            return cont_labels

    def _fit_resample(self, X, y, **params) -> None:
        """Unused method required by sklearn."""
        pass

    def __sklearn_tags__(self) -> Tags:
        """Return sklearn compatibility tags."""
        tags = super().__sklearn_tags__()
        tags.estimator_type = "sampler"
        return tags

    def _reconstruct_continuous_y(
        self,
        y_bin_new: np.ndarray,
        y_orig: np.ndarray,
        y_bin_orig: np.ndarray,
    ) -> np.ndarray:
        """Reconstruct continuous targets from bin assignments.

        For each generated sample (defined by its bin), a continuous value
        is synthesized by interpolating between two original samples from
        the same bin.

        Parameters
        ----------
        y_bin_new : np.ndarray
            Binned targets after resampling (shape: n_samples,).
        y_orig : np.ndarray
            Original continuous targets (shape: n_samples,).
        y_bin_orig : np.ndarray
            Original bin assignments (shape: n_samples,).

        Returns
        -------
        np.ndarray
            Reconstructed continuous targets (float, shape: n_samples,).

        Notes
        -----
        - Ensures synthetic values remain within the distribution of each bin
        - Equivalent to 1D SMOTE in target space

        """
        rng = check_random_state(self.random_state)

        y_new = np.empty(len(y_bin_new), dtype=float)

        # Precompute indices per bin for efficiency
        bin_to_indices = {cls: np.where(y_bin_orig == cls)[0] for cls in np.unique(y_bin_orig)}

        for i, cls in enumerate(y_bin_new):
            idx = bin_to_indices.get(cls)

            if idx is None or len(idx) == 0:
                raise RuntimeError(f"Empty bin {cls} during reconstruction.")

            # If only one sample → just copy it
            if len(idx) == 1:
                y_new[i] = y_orig[idx[0]]
                continue

            # SMOTE-like interpolation in target space
            i1, i2 = rng.choice(idx, size=2, replace=True)
            alpha = rng.rand()

            y_new[i] = alpha * y_orig[i1] + (1 - alpha) * y_orig[i2]

        return y_new

    def _resolve_continuous_covariate(
        self, cont: np.ndarray | None, n_samples: int, n_bins_cont_cov: int | None, binning_strategy_cont_cov: str
    ) -> np.ndarray | None:
        """Discretize continuous covariates into categorical group labels.

        Transforms continuous covariates into integer labels by binning each
        feature and combining them using mixed-radix encoding. This enables
        grouping samples with similar covariate patterns.

        Parameters
        ----------
        cont : np.ndarray or None
            Continuous covariates matrix (shape: n_samples x n_cont_features).
            If None, returns None immediately.
        n_samples : int
            Number of samples (used for creating label arrays).
        n_bins_cont_cov : int or None
            Number of bins for discretizing each continuous feature.
            Must be >= 2 when cont is provided.
        binning_strategy_cont_cov : str
            Binning strategy: either "quantile" (equal-frequency) or "uniform" (equal-width).

        Returns
        -------
        np.ndarray or None
            Integer group labels (shape: n_samples,) combining information from
            all continuous covariates, or None if cont is None.

        Raises
        ------
        ValueError
            If n_bins_cont_cov is None or < 2 when continuous covariates are provided.
            If binning_strategy_cont_cov is not 'quantile' or 'uniform'.

        Notes
        -----
        Binning process per feature:
            1. Detect constant columns (all values identical)
            2. Apply uniform or quantile binning based on strategy
            3. Handle edge cases: duplicate edges, full collapse to 1 bin
            4. Convert to bin indices using digitize

        Mixed-radix combination:
            Combines multiple features into a single integer label using
            positional encoding. Example with 2 features:
                Feature1 bins: [0, 1, 0], Feature2 bins: [0, 0, 1]
                Combined: [0*2+0, 1*2+0, 0*2+1] = [0, 2, 1]
            This ensures each unique combination gets a unique integer.

        Examples
        --------
        >>> # Single continuous feature with uniform binning
        >>> cont = np.array([[1.2], [1.5], [2.7], [2.8]])
        >>> labels = _resolve_continuous_covariate(cont, 4, 3, "uniform")
        >>> # Output: [0, 0, 2, 2] (assuming bins: [1.2-1.7, 1.7-2.2, 2.2-2.8])

        """
        # Step 1: Handle missing covariates
        if cont is None:
            return None

        # Step 2: Validate binning parameters
        # Require at least 2 bins to create meaningful groups
        if n_bins_cont_cov is None or n_bins_cont_cov < 2:
            raise ValueError(f"n_bins_cont_cov must be >= 2 when continuous covariates are provided. Got: {n_bins_cont_cov}")

        # Step 3: Validate binning strategy
        if binning_strategy_cont_cov not in {"quantile", "uniform"}:
            raise ValueError(f"binning_strategy_cont_cov must be 'quantile' or 'uniform'. Got: {binning_strategy_cont_cov}")

        # Step 4: Initialize container for combined labels
        # This will accumulate the mixed-radix encoding across features
        cont_labels = np.zeros(n_samples, dtype=int)

        # Step 5: Process each continuous feature independently
        for i in range(cont.shape[1]):
            col = cont[:, i]  # Extract i-th continuous covariate column

            # Step 5a: Handle constant columns (no variation to bin)
            # Constant columns would create empty bins, so assign all to single bin
            if np.all(col == col[0]):
                # All samples get bin 0 for this feature
                bins = np.zeros(n_samples, dtype=int)
            else:
                # Step 5b: Create bin edges based on selected strategy
                if binning_strategy_cont_cov == "quantile":
                    # Equal-frequency binning: same number of samples per bin
                    # Example: n_bins=4 -> edges at 0%, 25%, 50%, 75%, 100%
                    edges = np.percentile(
                        col,
                        np.linspace(0, 100, n_bins_cont_cov + 1),
                    )
                else:  # binning_strategy_cont_cov == "uniform"
                    # Equal-width binning: constant interval size
                    # Example: col range [0, 10], n_bins=5 -> edges [0, 2, 4, 6, 8, 10]
                    edges = np.linspace(col.min(), col.max(), n_bins_cont_cov + 1)

                # Step 5c: Remove duplicate edges
                # Can happen with quantile binning when many samples share same value
                # Example: col=[0,0,0,1] with 3 bins -> percentiles may produce [0,0,1]
                edges = np.unique(edges)

                # Step 5d: Handle complete bin collapse (all edges duplicate)
                # If after deduplication we have <= 2 edges, only 1 bin is possible
                # Example: all values identical already handled above, or binary with quantiles
                if len(edges) <= 2:
                    # Fallback to single bin for this feature
                    # All samples get bin 0
                    bins = np.zeros(n_samples, dtype=int)
                else:
                    # Step 5e: Assign each sample to a bin
                    # edges[1:-1] excludes first and last edges
                    # right=False means bins are [low, high) intervals
                    # digitize returns 0 for values < edges[1], n_bins-1 for >= edges[-2]
                    bins = np.digitize(col, edges[1:-1], right=False)

            # Step 6: Combine with previous features using mixed-radix encoding
            # This creates a unique label for each combination of bins across features
            # Formula: new_label = old_label * n_bins_current + current_bin
            # Example with feature1 bins [0,1,0] and feature2 bins [0,0,1]:
            #   Sample1: 0*2 + 0 = 0
            #   Sample2: 1*2 + 0 = 2
            #   Sample3: 0*2 + 1 = 1
            # Result: unique integer for each unique combination
            cont_labels = cont_labels * (bins.max() + 1) + bins

        # Step 7: Return the combined group labels
        # Note: Currently returns after first feature (potential bug)
        # Should be outside the loop to process all features
        return cont_labels
