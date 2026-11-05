"""Input handling shared by the harmonization estimators.

A harmonizer needs per-sample metadata (the **sites**, often **covariates**)
at ``fit`` *and* at ``transform``. scikit-learn only moves ``X`` and ``y``
through a :class:`~sklearn.pipeline.Pipeline` and only splits those two in
cross-validation, so the metadata can be given in two ways:

* **Columns of X** (works with every scikit-learn tool): the constructor names
  the columns (``site_column="site"``, ``continuous_columns=["age"]``, ...)
  and ``X`` is a pandas DataFrame (or a NumPy array with integer positions).
  The columns are read from ``X`` and removed from the features.
* **Keywords** (direct calls, NumPy users, metadata routing): the data are
  passed to ``fit``/``transform`` (``sites=...``, ``continuous_covariates=...``).

:func:`extract_metadata` reads both, checks them and returns a
:class:`HarmonizationData` with standardized types and names, so the
estimators only see validated arrays. :class:`HarmonizationInputMixin` wraps
it for estimators whose constructor holds the column parameters.
"""

from collections.abc import Hashable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, ClassVar, Literal, Self, TypeAlias

import numpy as np
import numpy.typing as npt
import pandas as pd
from sklearn.utils.validation import (
    FLOAT_DTYPES,
    _check_feature_names_in,
    check_array,
    check_is_fitted,
    validate_data,
)


__all__ = [
    "JULEARN_TYPE_SEPARATOR",
    "HarmonizationData",
    "HarmonizationInputMixin",
    "HarmonizationOutputMixin",
    "MetadataSpec",
    "TransformOutput",
    "check_sites_seen",
    "extract_metadata",
    "resolve_column",
]

#: julearn renames a column ``name`` of type ``type`` to ``name__:type:__type``.
JULEARN_TYPE_SEPARATOR = "__:type:__"

ColumnSpec = Hashable | Sequence[Hashable] | None

#: Data returned by ``transform``, ``fit_transform`` and ``fit_resample``: a NumPy
#: array by default, a pandas DataFrame with ``set_output(transform="pandas")``
#: (harmonizers) or with the column path (samplers). It depends on run-time
#: settings, so it is typed as ``Any`` to let type checkers accept the methods of
#: either container.
TransformOutput: TypeAlias = Any


@dataclass(frozen=True)
class MetadataSpec:
    """One kind of per-sample metadata accepted by an estimator.

    Parameters
    ----------
    keyword : str
        Data keyword of ``fit``/``transform`` (e.g., ``"sites"``,
        ``"continuous_covariates"``). It is also the key of the metadata in
        :class:`HarmonizationData`.
    column_param : str
        Constructor parameter naming the column(s) of ``X`` that hold it
        (e.g., ``"site_column"``, ``"continuous_columns"``).
    numeric : bool
        Whether the values must be numeric (converted to float64). Otherwise
        any dtype is kept (e.g., site names, categorical levels).
    single : bool
        Whether it is one value per sample (returned 1D, e.g., the sites).
        Otherwise it is returned 2D, shape (n_samples, n_columns).
    description : str
        Short description used in error messages (e.g., "site of every sample").

    """

    keyword: str
    column_param: str
    numeric: bool = False
    single: bool = False
    description: str = ""


@dataclass
class HarmonizationData:
    """Validated input of a harmonization estimator.

    Attributes
    ----------
    X : ndarray, shape (n_samples, n_features)
        The features (float64), without the metadata columns.
    metadata : dict of str to ndarray or None
        The metadata by keyword: 1D for single-column kinds (``"sites"``),
        2D (n_samples, n_columns) otherwise; None if not given.
    feature_names : ndarray of object or None
        Names of the features if ``X`` was a DataFrame, else None.
    index : pandas.Index or None
        Index of ``X`` if it was a DataFrame, else None.
    metadata_positions : tuple of int
        Positions in the input of the columns read as metadata (empty for the
        keyword path).
    from_columns : dict of str to bool
        For each keyword, whether its data came from columns of ``X``.
    column_positions : dict of str to list of int
        For each keyword read from columns, the positions of its columns.

    """

    X: npt.NDArray
    metadata: dict[str, npt.NDArray | None]
    feature_names: npt.NDArray | None = None
    index: pd.Index | None = None
    metadata_positions: tuple[int, ...] = ()
    from_columns: dict[str, bool] = field(default_factory=dict)
    column_positions: dict[str, list[int]] = field(default_factory=dict)

    @property
    def sites(self) -> npt.NDArray | None:
        """The sites (1D), or None."""
        return self.metadata.get("sites")

    def __getitem__(self, keyword: str) -> npt.NDArray | None:
        return self.metadata.get(keyword)


def _as_list(spec: ColumnSpec) -> list[Hashable]:
    """Return a column specification as a list of column names or positions."""
    if spec is None:
        return []
    if isinstance(spec, (str, int, np.integer)) or not isinstance(spec, Sequence):
        return [spec]
    return list(spec)


def resolve_column(columns: Sequence[Hashable], name: Hashable) -> int:
    """Return the position of the column ``name`` in ``columns``.

    The exact name is looked up first. If it is absent, a julearn-renamed
    column ``name__:type:__<type>`` is accepted, so the same estimator works
    inside and outside a julearn pipeline.

    Parameters
    ----------
    columns : sequence of hashable
        Column names of ``X``.
    name : hashable
        Name to find.

    Returns
    -------
    int
        Position of the column.

    Raises
    ------
    KeyError
        If no column, or more than one julearn-renamed column, matches.

    """
    columns = list(columns)
    exact = [i for i, column in enumerate(columns) if column == name]
    if len(exact) == 1:
        return exact[0]
    if len(exact) > 1:
        raise KeyError(f"Column {name!r} appears {len(exact)} times in X.")
    prefix = f"{name}{JULEARN_TYPE_SEPARATOR}"
    renamed = [i for i, column in enumerate(columns) if isinstance(column, str) and column.startswith(prefix)]
    if len(renamed) == 1:
        return renamed[0]
    if len(renamed) > 1:
        raise KeyError(f"Column {name!r} matches several julearn columns: {[columns[i] for i in renamed]}.")
    shown = columns if len(columns) <= 20 else [*columns[:20], "..."]
    raise KeyError(f"Column {name!r} is not in X. Columns of X: {shown}.")


def _column_positions(X: Any, spec: ColumnSpec, param: str) -> list[int]:
    """Return the positions in ``X`` of the columns named by ``spec``."""
    names = _as_list(spec)
    if isinstance(X, pd.DataFrame):
        positions = []
        for name in names:
            if isinstance(name, (int, np.integer)) and not isinstance(name, bool) and name not in X.columns:
                positions.append(int(name) if name >= 0 else X.shape[1] + int(name))
            else:
                positions.append(resolve_column(X.columns, name))
        return positions
    if not all(isinstance(name, (int, np.integer)) and not isinstance(name, bool) for name in names):
        raise TypeError(
            f"{param}={spec!r} names columns, so X must be a pandas DataFrame; got {type(X).__name__}. "
            "With a NumPy array, give integer column positions, or pass the data as keywords instead."
        )
    n_columns = np.shape(X)[1]
    positions = [int(name) if name >= 0 else n_columns + int(name) for name in names]
    out_of_range = [name for name, position in zip(names, positions, strict=True) if not 0 <= position < n_columns]
    if out_of_range:
        raise IndexError(f"{param}: column positions {out_of_range} are out of range for X with {n_columns} columns.")
    return positions


def _take_columns(X: Any, positions: list[int]) -> Any:
    """Return the columns ``positions`` of ``X`` (DataFrame or array)."""
    if isinstance(X, pd.DataFrame):
        return X.iloc[:, positions]
    return np.asarray(X)[:, positions]


def _to_numpy(values: Any) -> npt.NDArray:
    """Convert a DataFrame, Series or array-like to a NumPy array (pandas extension dtypes to object)."""
    if isinstance(values, (pd.DataFrame, pd.Series)):
        dtypes = values.dtypes if isinstance(values, pd.DataFrame) else [values.dtype]
        numpy_dtypes = all(isinstance(dtype, np.dtype) for dtype in dtypes)
        return values.to_numpy() if numpy_dtypes else values.to_numpy(dtype=object)
    return np.asarray(values)


def _check_metadata(
    values: Any, spec: MetadataSpec, n_samples: int, estimator: Any, copy: bool, allow_nan: bool = False
) -> npt.NDArray:
    """Validate one kind of metadata and return it 1D (single) or 2D; missing values only with ``allow_nan`` (not sites)."""
    name = spec.keyword
    array = check_array(
        _to_numpy(values) if not spec.numeric else values,
        dtype=FLOAT_DTYPES if spec.numeric else None,
        ensure_2d=False,
        allow_nd=False,
        ensure_all_finite="allow-nan" if allow_nan and not spec.single else True,
        copy=copy,
        estimator=estimator,
        input_name=name,
    )
    if spec.single:
        if array.ndim == 2 and array.shape[1] == 1:
            array = array.ravel()
        if array.ndim != 1:
            raise ValueError(f"{name} must have one value per sample (1D), got shape {array.shape}.")
    elif array.ndim == 1:
        array = array.reshape(-1, 1)
    if array.shape[0] != n_samples:
        raise ValueError(f"{name} has {array.shape[0]} samples, but X has {n_samples} (inconsistent numbers of samples).")
    return array


def extract_metadata(
    estimator: Any,
    X: npt.ArrayLike | pd.DataFrame,
    specs: Sequence[MetadataSpec],
    columns: Mapping[str, ColumnSpec],
    values: Mapping[str, Any],
    reset: bool,
    copy: bool = False,
    allow_nan_metadata: bool = False,
) -> HarmonizationData:
    """Read the features and the metadata from columns of ``X`` or from keywords.

    Each kind of metadata comes from exactly one place: the columns named in
    the constructor (``columns``) or the keyword data (``values``).

    Parameters
    ----------
    estimator : estimator instance
        The calling estimator: ``n_features_in_`` and ``feature_names_in_``
        are set (``reset=True``) or checked against the input.
    X : array-like or DataFrame, shape (n_samples, n_columns)
        The input, with the metadata columns if any are named.
    specs : sequence of MetadataSpec
        Kinds of metadata accepted by the estimator.
    columns : mapping of str to column specification
        By keyword, the column name(s) or position(s) holding that metadata,
        or None.
    values : mapping of str to array-like or None
        By keyword, the data given directly, or None.
    reset : bool
        True in ``fit``, False in ``transform``.
    copy : bool, optional (default False)
        Whether to copy the arrays.
    allow_nan_metadata : bool, optional (default False)
        Whether the metadata other than the sites may contain missing values.

    Returns
    -------
    HarmonizationData
        The validated features and metadata.

    Raises
    ------
    ValueError
        If some metadata is given both as columns and as a keyword, a column
        is used twice, no feature is left, or the data are invalid (missing
        or infinite values, inconsistent lengths, different columns than in
        ``fit``).

    """
    named = _named_columns(estimator, specs, columns, values)
    index = X.index if isinstance(X, pd.DataFrame) else None
    from_columns = {spec.keyword: spec.keyword in named for spec in specs}
    if named:
        features, column_positions, feature_names, metadata = _split_columns(estimator, X, specs, named, reset, copy)
        positions = tuple(sorted(p for keyword_positions in column_positions.values() for p in keyword_positions))
    else:
        # Keyword path: X holds only features
        features = validate_data(estimator, X, reset=reset, copy=copy, dtype=FLOAT_DTYPES)
        positions, metadata, column_positions = (), {}, {}
        feature_names = np.asarray(X.columns, dtype=object) if isinstance(X, pd.DataFrame) else None

    n_samples = features.shape[0]
    for spec in specs:
        raw = metadata.get(spec.keyword) if from_columns[spec.keyword] else values.get(spec.keyword)
        metadata[spec.keyword] = (
            None if raw is None else _check_metadata(raw, spec, n_samples, estimator, copy, allow_nan_metadata)
        )
    return HarmonizationData(
        X=features,
        metadata=metadata,
        feature_names=feature_names,
        index=index,
        metadata_positions=positions,
        from_columns=from_columns,
        column_positions=column_positions,
    )


def _named_columns(
    estimator: Any, specs: Sequence[MetadataSpec], columns: Mapping[str, ColumnSpec], values: Mapping[str, Any]
) -> dict[str, ColumnSpec]:
    """Return the column specification of the kinds of metadata read from columns; check each has one source."""
    unknown = sorted(set(values) - {spec.keyword for spec in specs})
    if unknown:
        raise TypeError(f"{type(estimator).__name__} does not accept {unknown}.")
    named = {spec.keyword: columns.get(spec.keyword) for spec in specs if _as_list(columns.get(spec.keyword))}
    for keyword, column_spec in named.items():
        if values.get(keyword) is not None:
            raise ValueError(
                f"{keyword} given twice: as columns of X ({_param(specs, keyword)}={column_spec!r}) and as the "
                f"{keyword}= keyword. Give it in one place only."
            )
    return named


def _split_columns(
    estimator: Any, X: Any, specs: Sequence[MetadataSpec], named: dict[str, ColumnSpec], reset: bool, copy: bool
) -> tuple[npt.NDArray, dict[str, list[int]], npt.NDArray | None, dict[str, Any]]:
    """Split ``X`` into the features and the metadata columns.

    Returns
    -------
    tuple
        The validated features, the positions of the metadata columns by
        keyword, the feature names (None without column names) and the raw
        metadata by keyword.

    """
    if not hasattr(X, "shape") or len(np.shape(X)) != 2:
        X = np.asarray(X, dtype=object)
    # n_features_in_ and feature_names_in_ describe the full input, metadata columns included
    validate_data(estimator, X, reset=reset, skip_check_array=True)
    used: dict[int, str] = {}
    metadata, column_positions = {}, {}
    for keyword, column_spec in named.items():
        param = _param(specs, keyword)
        keyword_positions = _column_positions(X, column_spec, param)
        if next(s for s in specs if s.keyword == keyword).single and len(keyword_positions) != 1:
            raise ValueError(f"{param} must name a single column, got {column_spec!r}.")
        for position in keyword_positions:
            if position in used:
                raise ValueError(f"Column {_column_name(X, position)!r} is used both as {used[position]} and as {keyword}.")
            used[position] = keyword
        metadata[keyword] = _take_columns(X, keyword_positions)
        column_positions[keyword] = keyword_positions
    keep = [i for i in range(np.shape(X)[1]) if i not in used]
    if not keep:
        raise ValueError("No feature left in X once the metadata columns are removed.")
    features = check_array(_take_columns(X, keep), dtype=FLOAT_DTYPES, copy=copy, estimator=estimator)
    feature_names = np.asarray(X.columns[keep], dtype=object) if isinstance(X, pd.DataFrame) else None
    return features, column_positions, feature_names, metadata


def _param(specs: Sequence[MetadataSpec], keyword: str) -> str:
    return next(spec.column_param for spec in specs if spec.keyword == keyword)


def _column_name(X: Any, position: int) -> Hashable:
    return X.columns[position] if isinstance(X, pd.DataFrame) else position


def check_sites_seen(sites: npt.NDArray, fitted_sites: npt.NDArray, estimator_name: str) -> None:
    """Raise if ``sites`` contains a site not seen during ``fit``.

    Parameters
    ----------
    sites : ndarray, shape (n_samples,)
        Sites of the data to transform.
    fitted_sites : ndarray, shape (n_sites,)
        Sites seen during ``fit``.
    estimator_name : str
        Name of the estimator, for the message.

    Raises
    ------
    ValueError
        If a site was not seen during ``fit``. The message explains the usual
        cause in cross-validation.

    """
    unseen = np.setdiff1d(np.unique(sites.astype(str)), np.asarray(fitted_sites).astype(str))
    if unseen.size:
        raise ValueError(
            f"{estimator_name}: sites {unseen.tolist()} were not seen during fit, so their site effect is unknown. "
            "In cross-validation this happens when a test fold holds a site absent from the training folds "
            "(e.g., an unshuffled KFold on data sorted by site): use a shuffled splitter, or stratify the folds "
            "on the sites (e.g., StratifiedKFold(shuffle=True).split(X, sites))."
        )


class HarmonizationInputMixin:
    """Read features and metadata from columns of ``X`` or from keywords.

    Subclasses list the metadata they accept in ``_metadata_specs`` and have
    one constructor parameter per kind (``MetadataSpec.column_param``).
    """

    _metadata_specs: ClassVar[tuple[MetadataSpec, ...]] = ()

    # Fitted attributes set by scikit-learn's ``validate_data`` (declared for type checkers)
    n_features_in_: int
    feature_names_in_: npt.NDArray

    def _metadata_columns(self) -> dict[str, ColumnSpec]:
        """Return the column specification of each kind of metadata, from the constructor."""
        return {spec.keyword: getattr(self, spec.column_param, None) for spec in self._metadata_specs}

    def _validate_harmonization_input(
        self,
        X: npt.ArrayLike | pd.DataFrame,
        reset: bool,
        required: Sequence[str] = (),
        copy: bool = False,
        allow_nan_metadata: bool = False,
        **values: Any,
    ) -> HarmonizationData:
        """Read and validate the features and metadata.

        Parameters
        ----------
        X : array-like or DataFrame, shape (n_samples, n_columns)
            The input.
        reset : bool
            True in ``fit`` (stores the metadata columns), False in
            ``transform``.
        required : sequence of str, optional (default ())
            Keywords that must be present.
        copy : bool, optional (default False)
            Whether to copy the arrays.
        allow_nan_metadata : bool, optional (default False)
            Whether the metadata other than the sites may contain missing values.
        **values : array-like or None
            The keyword data.

        Returns
        -------
        HarmonizationData
            The validated input.

        Raises
        ------
        ValueError
            If a required kind of metadata is missing.

        """
        data = extract_metadata(
            self,
            X,
            self._metadata_specs,
            self._metadata_columns(),
            values,
            reset=reset,
            copy=copy,
            allow_nan_metadata=allow_nan_metadata,
        )
        for keyword in required:
            if data[keyword] is None:
                spec = next(s for s in self._metadata_specs if s.keyword == keyword)
                what = f" ({spec.description})" if spec.description else ""
                raise ValueError(
                    f"{type(self).__name__} needs {keyword}{what}: set {spec.column_param}=<column name(s)> in the "
                    f"constructor to read it from a DataFrame X, or pass {keyword}=... to fit and transform."
                )
        if reset:
            self._metadata_positions = data.metadata_positions
            self._metadata_used = {keyword: value is not None for keyword, value in data.metadata.items()}
        return data

    def _check_metadata_as_in_fit(self, data: HarmonizationData, keywords: Sequence[str]) -> None:
        """Check that ``transform`` receives the same kinds of metadata as ``fit``.

        Parameters
        ----------
        data : HarmonizationData
            The input of ``transform``.
        keywords : sequence of str
            Kinds of metadata to check (e.g., covariates).

        Raises
        ------
        ValueError
            If a kind given to ``fit`` is missing, or a kind not given to
            ``fit`` is given.

        """
        used = getattr(self, "_metadata_used", {})
        for keyword in keywords:
            if used.get(keyword, False) and data[keyword] is None:
                raise ValueError(
                    f"{keyword} were given to fit, so they must also be given to transform "
                    "(their effect is preserved for every sample)."
                )
            if not used.get(keyword, False) and data[keyword] is not None:
                raise ValueError(
                    f"{keyword} were not given to fit, so transform cannot use them. "
                    f"Fit again with {keyword} to preserve their effect."
                )

    def get_feature_names_out(self, input_features: npt.ArrayLike | None = None) -> npt.NDArray:
        """Get the names of the output (harmonized) features.

        The metadata columns (sites, covariates) are not part of the output.

        Parameters
        ----------
        input_features : array-like of str or None, optional (default None)
            Names of the input columns. If None, ``feature_names_in_`` is used,
            or ``["x0", "x1", ...]`` if ``X`` had no column names.

        Returns
        -------
        ndarray of str
            Names of the output features.

        """
        check_is_fitted(self, "n_features_in_")
        names = _check_feature_names_in(self, input_features)
        keep = np.ones(len(names), dtype=bool)
        keep[list(getattr(self, "_metadata_positions", ()))] = False
        return np.asarray(names[keep], dtype=object)


class HarmonizationOutputMixin:
    """``set_output`` typed to return the estimator itself.

    scikit-learn's ``set_output`` is untyped, and the scikit-learn stubs bundled
    with Pylance annotate it as returning ``BaseEstimator``, so editors flag
    ``NeuroComBat(...).set_output(transform="pandas").fit_transform(...)``.
    This override only adds the return type. Place it before
    :class:`~sklearn.base.TransformerMixin` in the bases.
    """

    def set_output(self, *, transform: Literal["default", "pandas", "polars"] | None = None) -> Self:
        """Set the output container of ``transform`` and ``fit_transform``.

        Parameters
        ----------
        transform : {"default", "pandas", "polars"} or None, optional (default None)
            ``"pandas"`` returns a DataFrame with the harmonized feature names
            (``get_feature_names_out()``) and the index of ``X``;
            ``"default"`` returns a NumPy array; None leaves the setting
            unchanged.

        Returns
        -------
        self
            The estimator.

        """
        super().set_output(transform=transform)  # type: ignore[misc]
        return self
