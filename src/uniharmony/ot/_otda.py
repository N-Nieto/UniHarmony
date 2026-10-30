"""Optimal Transport for Domain Adaptation (OTDA) implementation."""

from typing import Any, Literal

import numpy as np
import numpy.typing as npt
import structlog
from ot.da import BaseTransport
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils import Tags
from sklearn.utils.validation import FLOAT_DTYPES, check_array, check_consistent_length, check_is_fitted

from uniharmony._utils import validate_sites
from uniharmony.ot._utils import create_ot_object, data_consistency_check


__all__ = ["OptimalTransportDomainAdaptation"]

logger = structlog.get_logger(src="OptimalTransportDomainAdaptation")

OTMethodType = Literal["emd", "sinkhorn", "s", "sinkhorn_gl", "s_gl", "emd_laplace", "emd_l"]
# Type aliases for clarity
MetricType = Literal[
    "sqeuclidean",
    "euclidean",
    "braycurtis",
    "canberra",
    "chebyshev",
    "cityblock",
    "correlation",
    "cosine",
    "dice",
    "hamming",
    "jaccard",
    "kulsinski",
    "mahalanobis",
    "matching",
    "minkowski",
    "rogerstanimoto",
    "russellrao",
    "seuclidean",
    "sokalmichener",
    "sokalsneath",
    "wminkowski",
    "yule",
]
NormalizationType = Literal["median", "max", "log", "loglog"] | None


class OptimalTransportDomainAdaptation(BaseTransport, TransformerMixin, BaseEstimator):
    """Optimal Transport for Domain Adaptation with reference site handling.

    This class extends POT's BaseTransport to provide a harmonization interface
    where data from multiple sites is aligned to a reference site(s) using optimal
    transport. The implementation supports both string-based OT method selection
    and direct injection of pre-configured OT instances.

    Parameters
    ----------
    ot_method : str or BaseTransport instance, optional (default "emd")
        Optimal transport method to use. Can be either:

        - A string: "emd", "sinkhorn"/"s", "sinkhorn_gl"/"s_gl", "emd_laplace"/"emd_l"
        - A pre-configured BaseTransport instance (e.g., ot.da.SinkhornTransport(reg_e=0.1))

    metric : str, optional (default "euclidean")
        Distance metric for cost matrix computation. Supports all metrics from
        scipy.spatial.distance.cdist and POT's backend implementations.

    reg : float or None, optional (default 1.0)
        Entropic regularization parameter. Used for Sinkhorn-based methods.

    eta : float or None, optional (default 0.1)
        Regularization parameter for Laplace or group Lasso regularization.

    max_iter : int or None, optional (default 10)
        Maximum number of iterations for iterative solvers.

    cost_norm : str or None, optional (default None)
        Cost matrix normalization method: "median", "max", "log", "loglog".

    limit_max : int or None, optional (default 10)
        Semi-supervised mode control. Sets infinite cost (10 * max(cost))
        for transport between different classes.

    Attributes
    ----------
    ot_obj_ : BaseTransport
        The fitted underlying OT object (set during fit).

    ref_site_ : str or list of str
        The reference site(s) used for alignment.

    sites_ : ndarray, shape (n_sites,)
        Sites seen during fit.

    coupling_ : array, shape (n_source_samples, n_target_samples)
        The optimal coupling matrix (forwarded from ot_obj_).

    cost_ : array
        The computed cost matrix (forwarded from ot_obj_).

    Examples
    --------
    >>> # Using string-based method selection
    >>> otda = OptimalTransportDomainAdaptation(ot_method="sinkhorn", reg=0.1, metric="sqeuclidean")
    >>> otda.fit(X_train, sites_train, ref_site="site_A", y=labels)
    >>> X_aligned = otda.transform(X_test, sites_test)

    >>> # Using pre-configured OT instance
    >>> from ot.da import SinkhornTransport
    >>> ot_solver = SinkhornTransport(reg_e=0.5, norm="median")
    >>> otda = OptimalTransportDomainAdaptation(ot_method=ot_solver)
    >>> otda.fit(X_train, sites_train, ref_site="site_A")
    >>> X_aligned = otda.transform(X_test, sites_test))


    """

    def __init__(
        self,
        ot_method: str | BaseTransport = "emd",
        metric: MetricType = "euclidean",
        reg: float | None = 1.0,
        eta: float | None = 0.1,
        max_iter: int | None = 10,
        cost_norm: NormalizationType = None,
        limit_max: int | None = 10,
        copy: bool = True,
    ) -> None:
        self.ot_method = ot_method
        self.metric = metric
        self.reg = reg
        self.eta = eta
        self.max_iter = max_iter
        self.cost_norm = cost_norm
        self.limit_max = limit_max
        self.copy = copy

        self._ot_method_aliases: dict[str, str] = {
            "s": "sinkhorn",
            "s_gl": "sinkhorn_gl",
            "emd_l": "emd_laplace",
        }

    def fit(
        self,
        X: npt.ArrayLike,
        sites: npt.ArrayLike,
        ref_site: str | list[str] | int | list[int],
        y: npt.ArrayLike | None = None,
    ) -> "OptimalTransportDomainAdaptation":
        """Fit optimal transport from non-reference sites to reference site(s).

        This method separates the data into reference (target) and non-reference
        (source) domains, then fits the optimal transport plan to map source
        distributions onto the reference distribution.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Input data from all sites.

        sites : array-like, shape (n_samples,)
            Site labels for each sample. Must align with X.

        ref_site : str, int or list of str or list of int
            Site identifier(s) to use as reference (target domain).
            If list, combines all specified sites as the reference distribution.

        y : array-like, shape (n_samples,) or (n_samples, n_classes), optional (default None)
            Labels for supervised/semi-supervised transport. Used for cost
            matrix computation. Must align with X if provided.

        Returns
        -------
        self : OptimalTransportDomainAdaptation
            Fitted instance.

        """
        X = check_array(X, copy=self.copy, dtype=FLOAT_DTYPES, estimator=self)

        if y is not None:
            logger.debug("Validating labels for supervised transport")
            y = check_array(y, copy=self.copy, ensure_2d=False, estimator=self)
            check_consistent_length(X, y)

        sites = check_array(sites, copy=self.copy, dtype=None, ensure_2d=False, estimator=self)
        check_consistent_length(X, sites)
        validate_sites(sites)

        # Store reference site info
        logger.info(f"Using reference site(s): {ref_site}")
        self.ref_site_ = ref_site
        self.sites_ = np.unique(sites)

        # Initialize OT object (string or instance)
        self.ot_obj_ = self._resolve_ot_method()
        logger.info(f"Initialized OT object: {type(self.ot_obj_).__name__}")

        # Split reference vs. harmonization data
        # Data from the reference site(s) is the target distribution
        # data from other sites (harmonization) is the source distribution to be aligned to the reference.
        X_ref, X_harm, y_ref, y_harm = self._split_ref_harm_data(X, sites, ref_site, y)

        # Determine if we should use labels for fitting
        # POT's fit method handles weight initialization and cost matrix internally
        if y_ref is not None and y_harm is not None:
            # Semi-supervised or supervised domain adaptation
            # Labels are used to guide transport (same class -> lower cost)
            logger.info("Fitting OT with supervised cost matrix")
            self.ot_obj_.fit(Xs=X_harm, ys=y_harm, Xt=X_ref, yt=y_ref)
        else:
            logger.info("Fitting OT with unsupervised cost matrix")
            # Unsupervised domain adaptation
            self.ot_obj_.fit(Xs=X_harm, Xt=X_ref)

        # Expose key attributes from underlying OT object for easier inspection
        self.coupling_ = self.ot_obj_.coupling_
        logger.debug(f"Fitted coupling matrix with shape: {self.coupling_.shape}")
        self.cost_ = self.ot_obj_.cost_
        logger.debug(f"Fitted cost matrix with shape: {self.cost_.shape}")

        return self

    def transform(
        self,
        X: npt.ArrayLike,
        sites: npt.ArrayLike | None = None,
        y: npt.ArrayLike | None = None,
        batch_size: int = 128,
    ) -> npt.NDArray:
        """Transform data using the fitted OT plan.

        Transports samples from the source domain (non-reference sites) to the
        target domain (reference site). If sites are provided, only transforms
        non-reference samples; reference samples are returned unchanged.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Input data to transform.

        sites : array-like, shape (n_samples,), optional (default None)
            Site labels for X, all seen during fit (any subset of them; the
            reference site does not need to be present). If provided, only
            samples of non-reference sites are transformed; reference site
            samples are returned as-is. If None, all samples are transformed.

        y : array-like, shape (n_samples,) or (n_samples, n_classes), optional (default None)
            Labels for supervised transport. Must align with X if provided.
             Used to ensure consistent handling of supervised transformations.

        batch_size : int, optional (default 128)
            Batch size for out-of-sample transformation.

        Returns
        -------
        X_transformed : ndarray, shape (n_samples, n_features)
            Transformed data aligned to reference distribution.

        Raises
        ------
        ValueError
            If a site was not seen during fit.

        """
        check_is_fitted(self)
        X, y, harm_mask = self._check_transform_input(X, sites, y)
        if harm_mask is None:
            # Transform all samples assuming they are from source domain
            logger.info(
                "Transforming all samples without site-based masking, some of which may be from reference site."
                "Ensure this is intended."
            )
            return self.ot_obj_.transform(Xs=X, ys=y, batch_size=batch_size)

        X_transformed = X.copy()
        if np.any(harm_mask):
            y_harm = y[harm_mask] if y is not None else None
            X_transformed[harm_mask] = self.ot_obj_.transform(Xs=X[harm_mask], ys=y_harm, batch_size=batch_size)
        else:
            logger.info("All samples are from the reference site(s): they are returned unchanged.")
        return X_transformed

    def fit_transform(
        self,
        X: npt.ArrayLike,
        sites: npt.ArrayLike,
        ref_site: str | list[str] | int | list[int],
        y: npt.ArrayLike | None = None,
        batch_size: int = 128,
    ) -> npt.NDArray:
        """Fit and transform in one step.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Input data.

        sites : array-like, shape (n_samples,)
            Site labels.

        ref_site : str or list of str
            Reference site(s).

        y : array-like, optional (default None)
            Labels for supervised transport.

        batch_size : int, optional (default 128)
            Batch size for transformation.

        Returns
        -------
        X_transformed : ndarray
            Data aligned to reference distribution.

        """
        return self.fit(X, sites, ref_site, y).transform(X, sites, y, batch_size)

    def inverse_transform(
        self,
        X: npt.ArrayLike,
        sites: npt.ArrayLike | None = None,
        y: npt.ArrayLike | None = None,
        batch_size: int = 128,
    ) -> npt.NDArray:
        """Map harmonized data back from the reference domain to the source domain.

        This is the inverse of :meth:`transform`: samples of non-reference
        sites, which :meth:`transform` moved onto the reference distribution,
        are mapped back to the distribution of the non-reference sites.
        Samples of the reference site(s) are returned unchanged.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Harmonized data (output of :meth:`transform`).
        sites : array-like, shape (n_samples,), optional (default None)
            Site labels for X, all seen during fit. If provided, only samples
            of non-reference sites are mapped back; reference samples are
            returned as-is. If None, all samples are mapped back.

        y : array-like, shape (n_samples,) or (n_samples, n_classes), optional (default None)
            Labels for supervised transport. Must align with X if provided.
             Used to ensure consistent handling of supervised transformations.

        batch_size : int, optional (default 128)
            Batch size for transformation.

        Returns
        -------
        X_inv_transformed : ndarray
            Data mapped back to source distribution.

        Notes
        -----
        The non-reference sites are pooled into one source domain when fitting,
        so samples are mapped back to that pooled distribution, not to their own
        site. Samples that are not the transported training samples are mapped
        out-of-sample (displacement of their nearest transported training
        sample), so ``inverse_transform(transform(X))`` recovers ``X`` only
        approximately in general.

        """
        check_is_fitted(self)
        X, y, harm_mask = self._check_transform_input(X, sites, y)
        if harm_mask is None:
            logger.info("Mapping all samples back to the source domain without site-based masking. Ensure this is intended.")
            return self.ot_obj_.inverse_transform(Xt=X, yt=y, batch_size=batch_size)

        X_inverse = X.copy()
        if np.any(harm_mask):
            y_harm = y[harm_mask] if y is not None else None
            X_inverse[harm_mask] = self.ot_obj_.inverse_transform(Xt=X[harm_mask], yt=y_harm, batch_size=batch_size)
        else:
            logger.info("All samples are from the reference site(s): they are returned unchanged.")
        return X_inverse

    def _check_transform_input(
        self,
        X: npt.ArrayLike,
        sites: npt.ArrayLike | None,
        y: npt.ArrayLike | None,
    ) -> tuple[npt.NDArray, npt.NDArray | None, npt.NDArray | None]:
        """Validate the input of (inverse) transform and find the samples to map.

        Unlike :meth:`fit`, any subset of the fitted sites is accepted: the
        transport plan is fixed, so neither the reference site nor a second
        site needs to be present.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Data.
        sites : array-like, shape (n_samples,) or None
            Site labels, or None to map all samples.
        y : array-like, shape (n_samples,) or None
            Labels.

        Returns
        -------
        X : ndarray, shape (n_samples, n_features)
            The validated data.
        y : ndarray, shape (n_samples,) or None
            The validated labels.
        harm_mask : ndarray of bool, shape (n_samples,) or None
            Samples of non-reference sites, or None if ``sites`` is None.

        Raises
        ------
        ValueError
            If a site was not seen during fit.

        """
        X = check_array(X, copy=self.copy, dtype=FLOAT_DTYPES, estimator=self)
        if y is not None:
            y = check_array(y, copy=self.copy, ensure_2d=False, estimator=self)
            check_consistent_length(X, y)
        if sites is None:
            return X, y, None
        sites = check_array(sites, copy=self.copy, dtype=None, ensure_2d=False, estimator=self)
        check_consistent_length(X, sites)
        sites_str = sites.astype(str)
        unseen = sorted(set(sites_str) - set(self.sites_.astype(str)))
        if unseen:
            raise ValueError(f"Sites {unseen} were not seen during fit.")
        ref_mask = np.isin(sites_str, self._reference_sites_as_str(self.ref_site_))
        return X, y, ~ref_mask

    @staticmethod
    def _reference_sites_as_str(ref_site: str | int | list[str] | list[int]) -> list[str]:
        """Return the reference site(s) as a list of strings."""
        if isinstance(ref_site, (str, int, np.integer)):
            return [str(ref_site)]
        return [str(s) for s in ref_site]

    def _resolve_ot_method(self, **kwargs: Any) -> BaseTransport:
        """Resolve ot_method parameter to a BaseTransport instance.

        Parameters
        ----------
        **kwargs : dict
            Additional parameters to pass to OT object creation if using string.

        Returns
        -------
        BaseTransport
            Configured OT transport instance.

        Raises
        ------
        TypeError
            If ot_method is not a string or BaseTransport instance.

        """
        if isinstance(self.ot_method, BaseTransport):
            # User provided pre-configured instance
            return self.ot_method

        if isinstance(self.ot_method, str):
            # Resolve aliases to canonical names
            method_name = self.ot_method.lower()
            method_name = self._ot_method_aliases.get(method_name, method_name)

            # Filter kwargs relevant to the specific OT method
            # Only pass parameters that the specific method supports
            ot_kwargs = {
                "metric": self.metric,
                "norm": self.cost_norm,
                "limit_max": self.limit_max,
            }

            # Add method-specific parameters
            if method_name in ["sinkhorn", "sinkhorn_gl", "emd_laplace"]:
                ot_kwargs["reg_e"] = self.reg
            if method_name in ["sinkhorn_gl", "emd_laplace"]:
                ot_kwargs["reg_cl"] = self.eta
            if method_name in ["sinkhorn", "sinkhorn_gl"]:
                ot_kwargs["max_iter"] = self.max_iter

            # Override with any user-provided kwargs
            ot_kwargs.update(kwargs)

            return create_ot_object(method_name, **ot_kwargs)

        raise TypeError(f"ot_method must be str or BaseTransport instance, got {type(self.ot_method).__name__}")

    def _split_ref_harm_data(
        self, X: npt.ArrayLike, sites: npt.ArrayLike, ref_site: str | int | list[str] | list[int], y: npt.ArrayLike | None = None
    ) -> tuple[npt.NDArray, npt.NDArray, npt.NDArray | None, npt.NDArray | None]:
        """Validate site labels and split data in reference vs. harmonization data.

        Converts all site identifiers to strings for consistent comparison,
        handling both integer and string site labels.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Input data from all sites.

        sites : array-like, shape (n_samples,)
            Site labels for all samples. Can be integers or strings.

        ref_site : str, int, or list of str/int
            Reference site identifier(s). Will be converted to string(s)
            for comparison with converted site labels.

        y : array-like, shape (n_samples,) or (n_samples, n_classes), optional
            Labels for supervised transport. Must align with X if provided.

        Returns
        -------
        ref_mask : ndarray
            Boolean mask for reference samples.
        harm_mask : ndarray
            Boolean mask for samples to be harmonized.

        Raises
        ------
        ValueError
            If ref_site is not found in sites after conversion,
            or if no samples are assigned to reference or harmonization groups.

        """
        # Convert sites to string type for consistent comparison
        sites_arr = np.asarray(sites)
        sites_str = sites_arr.astype(str)
        unique_sites = np.unique(sites_str)

        # Convert ref_site to string(s) for consistent comparison
        if isinstance(ref_site, (str, int, np.integer)):
            ref_site_str = str(ref_site)
            if ref_site_str not in unique_sites:
                raise ValueError(
                    f"Reference site '{ref_site}' (as string: '{ref_site_str}') not found in sites. "
                    f"Available sites: {list(unique_sites)}"
                )
            ref_mask = sites_str == ref_site_str
        else:
            # Handle list/tuple of ref_sites
            ref_sites_str = [str(s) for s in ref_site]
            missing = set(ref_sites_str) - set(unique_sites)
            if missing:
                raise ValueError(
                    f"Reference sites {missing} (original: {list(ref_site)}) not found in sites. "
                    f"Available sites: {list(unique_sites)}"
                )
            ref_mask = np.isin(sites_str, ref_sites_str)

        harm_mask = ~ref_mask

        if not np.any(ref_mask):
            raise ValueError("No samples found for reference site(s)")
        if not np.any(harm_mask):
            raise ValueError("No samples found to harmonize (all samples are from reference site)")

        # Split data
        X_ref = X[ref_mask]  # Target domain (reference)
        X_harm = X[harm_mask]  # Source domain (to harmonize)

        # Handle labels if provided
        y_ref = y[ref_mask] if y is not None else None
        y_harm = y[harm_mask] if y is not None else None

        # Data validation
        data_consistency_check(X_source=X_ref, X_target=X_harm, y_source=y_ref, y_target=y_harm)

        logger.debug(
            f"Data split into reference and harmonization sets: {X_ref.shape[0]} reference samples, "
            f"{X_harm.shape[0]} harmonization samples",
        )

        return X_ref, X_harm, y_ref, y_harm

    def __sklearn_tags__(self) -> Tags:
        tags = super().__sklearn_tags__()
        tags.estimator_type = "transformer"
        tags.target_tags.required = True
        tags.target_tags.two_d_labels = True
        tags.target_tags.positive_only = True
        tags.input_tags.two_d_array = True
        tags.input_tags.allow_nan = True
        return tags
