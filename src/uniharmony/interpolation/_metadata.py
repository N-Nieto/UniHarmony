"""Sites and covariates of the samplers, as columns of ``X`` or as keywords.

A sampler followed by a harmonizer in an imbalanced-learn
:class:`~imblearn.pipeline.Pipeline` must hand the harmonizer the sites and
covariates of the *resampled* data, synthetic samples included. Metadata
routing cannot do this (it routes the metadata of the input rows only), so
with the column path the samplers return ``X`` with its metadata columns,
filled in for the synthetic samples:

* the **site** column holds the site the sample was created in;
* **continuous** covariates (and floating-point pass-through columns) are
  interpolated with the same weights as the features;
* **categorical** covariates (and other pass-through columns) take the value
  of the parent with the largest weight.
"""

from typing import Any, ClassVar

import numpy as np
import numpy.typing as npt
import pandas as pd
from scipy import sparse

from uniharmony._data import HarmonizationData, HarmonizationInputMixin, MetadataSpec
from uniharmony._utils import validate_sites


__all__ = ["SamplerInputMixin", "parent_weights"]

SITES = MetadataSpec("sites", "site_column", single=True, description="the site of every sample")
CATEGORICAL = MetadataSpec("categorical_covariates", "categorical_columns")
CONTINUOUS = MetadataSpec("continuous_covariates", "continuous_columns", numeric=True)
#: Columns carried to the output without being used for resampling (e.g., covariates of a later harmonizer)
PASSTHROUGH = MetadataSpec("passthrough", "passthrough_columns")


def parent_weights(
    rows: npt.NDArray, parents: npt.NDArray, weights: npt.NDArray, n_rows: int, n_samples: int
) -> sparse.csr_array:
    """Return the parent weights of synthetic samples as a sparse matrix.

    Parameters
    ----------
    rows : ndarray of int
        Synthetic sample (row of the matrix) of each entry.
    parents : ndarray of int
        Original sample (column) of each entry.
    weights : ndarray of float
        Weight of each entry; the weights of a row sum to 1.
    n_rows : int
        Number of synthetic samples.
    n_samples : int
        Number of original samples.

    Returns
    -------
    scipy.sparse.csr_array, shape (n_rows, n_samples)
        The weights (duplicate entries are summed).

    """
    return sparse.csr_array((weights, (rows, parents)), shape=(n_rows, n_samples))


class SamplerInputMixin(HarmonizationInputMixin):
    """Read the sites and covariates of a sampler and rebuild the metadata columns of its output."""

    _metadata_specs: ClassVar[tuple[MetadataSpec, ...]] = (SITES, CATEGORICAL, CONTINUOUS, PASSTHROUGH)

    def _sampler_input(
        self,
        X: Any,
        sites: Any,
        categorical_covariates: Any,
        continuous_covariates: Any,
        allow_nan: bool = False,
    ) -> HarmonizationData:
        """Validate the input of ``fit_resample`` (covariates may have missing values with ``allow_nan``)."""
        data = self._validate_harmonization_input(
            X,
            reset=True,
            required=("sites",),
            allow_nan_metadata=allow_nan,
            sites=sites,
            categorical_covariates=categorical_covariates,
            continuous_covariates=continuous_covariates,
        )
        validate_sites(data.sites)
        return data

    @staticmethod
    def _needs_parents(data: HarmonizationData) -> bool:
        """Whether the output has metadata columns other than the site, which need the parents of synthetic samples."""
        return any(data.from_columns.get(k, False) for k in ("categorical_covariates", "continuous_covariates", "passthrough"))

    @staticmethod
    def _resampled_output(
        X: Any,
        data: HarmonizationData,
        X_resampled: npt.NDArray,
        sites_resampled: npt.NDArray,
        sample_indices: npt.NDArray,
        weights: sparse.csr_array | None,
    ) -> Any:
        """Return the resampled data, with the metadata columns of ``X`` if it had any.

        Parameters
        ----------
        X : DataFrame or array
            The input of ``fit_resample``.
        data : HarmonizationData
            The validated input.
        X_resampled : ndarray, shape (n_resampled, n_features)
            The resampled features.
        sites_resampled : ndarray, shape (n_resampled,)
            Site of every resampled sample.
        sample_indices : ndarray of int, shape (n_resampled,)
            Original sample of every real resampled sample, -1 for synthetic ones.
        weights : sparse matrix, shape (n_synthetic, n_samples) or None
            Parent weights of the synthetic samples (in their output order).
            Needed if ``X`` has metadata columns other than the site.

        Returns
        -------
        DataFrame or ndarray
            ``X_resampled`` alone (keyword path), or the resampled data with
            the columns of ``X`` in their original order.

        """
        if not data.column_positions:
            return X_resampled
        frame = X if isinstance(X, pd.DataFrame) else pd.DataFrame(np.asarray(X))
        n_out = len(X_resampled)
        real = sample_indices >= 0
        synthetic = ~real
        if synthetic.any() and weights is None and SamplerInputMixin._needs_parents(data):  # pragma: no cover - internal
            raise RuntimeError("The parents of the synthetic samples are unknown.")
        dominant = None if weights is None or not synthetic.any() else np.asarray(weights.argmax(axis=1)).ravel()
        meta = {p: keyword for keyword, positions in data.column_positions.items() for p in positions}
        features = iter(X_resampled.T)
        columns = {}
        for position in range(frame.shape[1]):
            source = frame.iloc[:, position]
            if position not in meta:
                columns[position] = pd.Series(next(features))
                continue
            if meta[position] == "sites":
                values = np.asarray(sites_resampled)
            else:
                original = source.to_numpy()
                interpolate = meta[position] == "continuous_covariates" or (
                    meta[position] == "passthrough" and pd.api.types.is_float_dtype(source.dtype)
                )
                values = np.empty(n_out, dtype=float if interpolate else original.dtype)
                values[real] = original[sample_indices[real]]
                if synthetic.any():
                    values[synthetic] = weights @ original.astype(float) if interpolate else original[dominant]
            columns[position] = pd.Series(values)
            try:
                columns[position] = columns[position].astype(source.dtype)
            except (TypeError, ValueError):
                pass
        out = pd.concat([columns[p].rename(p) for p in range(frame.shape[1])], axis=1)
        out.columns = frame.columns
        return out if isinstance(X, pd.DataFrame) else out.to_numpy()
