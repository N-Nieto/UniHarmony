"""Plotting utilities for UniHarmony examples."""

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
import pandas as pd
import seaborn as sns
import sklearn.linear_model
import structlog
from matplotlib.axes import Axes
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.decomposition import PCA, KernelPCA
from sklearn.manifold import MDS, TSNE, Isomap
from sklearn.utils import check_random_state
from sklearn.utils.validation import (
    FLOAT_DTYPES,
    check_array,
    check_consistent_length,
)

from uniharmony._utils import validate_sites


__all__ = ["plot_2d_components_by_value", "plot_2d_projection", "plot_decision_boundary_2d", "plot_features_by_site"]

logger = structlog.get_logger()


def plot_decision_boundary_2d(
    ax: Axes,
    clf: sklearn.linear_model._base.LinearClassifierMixin,
    linewidths: float = 1.0,
    alpha: float = 0.7,
) -> None:
    """Plot 2D decision boundary.

    Parameters
    ----------
    ax : Axes
        Plot axes.
    clf : LinearClassifierMixin
        Linear classifier model.
    linewidths : float (default 1.0)
        Line widths for the plot.
    alpha : float (default 0.7)
        Alpha for the plot.

    """
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    xx, yy = np.meshgrid(np.linspace(xlim[0], xlim[1], 100), np.linspace(ylim[0], ylim[1], 100))
    Z = clf.decision_function(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    ax.contour(xx, yy, Z, levels=[0], colors="black", linewidths=linewidths, alpha=alpha)


def plot_2d_components_by_value(
    projection_data: pd.DataFrame,
    group_value: str,
    reducer_name: str,
    ax: Axes,
    alpha: float = 0.6,
    s: int = 30,
    fontsize: int = 12,
) -> None:
    """Plot 2D projection components colored by a group value.

    Parameters
    ----------
    projection_data : pd.DataFrame
        DataFrame containing projection components (must have columns 'comp1' and 'comp2').
    group_value : str
        Column name in projection_data to use for coloring.
    reducer_name : str
        Name of the dimensionality reduction method
    ax : Axes
        Matplotlib axes to plot on.
    alpha : float (default 0.6)
        Transparency level for scatter points.
    s : int (default 30)
        Size of scatter points.
    fontsize : int, default (12)
        Font size for title.

    """
    unique_groups = projection_data[group_value].unique()
    logger.info(f"Plotting 2D projection colored by '{group_value}'", n_groups=len(unique_groups), groups=list(unique_groups))

    for group in unique_groups:
        subset = projection_data[projection_data[group_value] == group]
        ax.scatter(subset["comp1"], subset["comp2"], label=str(group), alpha=alpha, s=s)

    ax.set_title(f"2D Projection using {reducer_name} - Colored by {group_value}", fontsize=fontsize, fontweight="bold")
    ax.set_xlabel("Component 1")
    ax.set_ylabel("Component 2")
    ax.legend(title=group_value, loc="best")
    ax.grid(True, alpha=0.3)
    logger.debug(f"Finished plotting for '{group_value}'")


def plot_2d_projection(
    X: npt.ArrayLike,
    sites: npt.ArrayLike,
    y: npt.ArrayLike | None = None,
    dim_reductor: BaseEstimator | TransformerMixin | None = None,
    figsize: tuple[int, int] = (12, 6),
    random_state: int | np.random.RandomState | None = None,
    **dim_reductor_kwargs,
) -> tuple[plt.Figure, np.ndarray]:
    """Plot 2D projection components with site and target.

    Parameters
    ----------
    X : array-like, shape (n_samples, n_features)
        The data samples.
    sites : array-like, shape (n_samples,)
        Samples' site labels.
    y : array-like or None, shape (n_samples,), default (None)
        Samples' target labels.
    dim_reductor : BaseEstimator, TransformerMixin or None, default None
        An instance of a dimensionality reduction technique that has:
        - n_components attribute
        - fit_transform method
        If None, creates a new TSNE instance with n_components=2.
    figsize : tuple[int, int], default (12, 6)
        Figure size (width, height) in inches.
    random_state : int or RandomState instance or None, optional (default None)
        The seed of the pseudo random number generator or RandomState for
        reproducibility.(used if dim_reductor is None).
    **dim_reductor_kwargs
        Additional keyword arguments passed to dim_reductor constructor
        if dim_reductor is None.

    Returns
    -------
    fig : plt.Figure
        The matplotlib figure object.
    axes : np.ndarray
        Array of matplotlib axes objects.

    Raises
    ------
    ValueError
        If the dimensionality reducer's n_components is not 2.
    TypeError
        If dim_reductor doesn't have fit_transform method or n_components attribute.

    Examples
    --------
    >>> from sklearn.manifold import TSNE
    >>> tsne = TSNE(n_components=2, random_state=42)
    >>> fig, axes = plot_2d_projection(X, sites, y, dim_reductor=tsne)

    >>> # Using PCA with custom parameters
    >>> fig, axes = plot_2d_projection(X, sites, y, dim_reductor=None,
    ...                                method='pca', n_components=2)

    """
    logger.info(
        "Starting 2D projection plotting", n_samples=X.shape[0] if hasattr(X, "shape") else len(X), has_target=y is not None
    )
    random_state = check_random_state(random_state)

    # Validate inputs
    X = check_array(X, copy=True, dtype=FLOAT_DTYPES)
    sites = check_array(sites, copy=True, dtype=None, ensure_2d=False)
    check_consistent_length(X, sites)
    validate_sites(sites)

    if y is not None:
        y = check_array(y, copy=True, dtype=None, ensure_2d=False)
        check_consistent_length(X, y)
        logger.info(f"Target variable provided with {len(np.unique(y))} unique classes")

    dim_reductor = _resolve_dimensionality_reductor(X, dim_reductor, random_state, **dim_reductor_kwargs)

    X_2d = dim_reductor.fit_transform(X)
    logger.info(
        "Dimensionality reduction completed",
        output_shape=X_2d.shape,
        explained_variance=getattr(dim_reductor, "explained_variance_ratio_", None),
    )

    # Get reducer name for column labels
    reducer_name = type(dim_reductor).__name__.lower()

    # Create DataFrame with projection results
    df_2d = pd.DataFrame(
        {
            "comp1": X_2d[:, 0],
            "comp2": X_2d[:, 1],
            "site": sites,
        }
    )

    if y is not None:
        df_2d["target"] = y

    # Create plots
    if y is None:
        logger.info("Creating single plot (no target variable)")
        fig, ax = plt.subplots(1, 1, figsize=figsize)
        axes = np.array([ax])
        plot_2d_components_by_value(df_2d, "site", reducer_name, ax)

    else:
        logger.info("Creating two subplots: colored by site and by target")
        fig, axes = plt.subplots(1, 2, figsize=figsize)

        # Plot by site
        plot_2d_components_by_value(df_2d, "site", reducer_name, axes[0])

        # Plot by target
        plot_2d_components_by_value(df_2d, "target", reducer_name, axes[1])

    plt.tight_layout()
    logger.info("2D projection plotting completed")

    return fig, axes


def _resolve_dimensionality_reductor(
    X: npt.ArrayLike, dim_reductor: BaseEstimator | TransformerMixin | None, random_state, **dim_reductor_kwargs
) -> BaseEstimator | TransformerMixin:
    # Handle dimensionality reducer
    if dim_reductor is None:
        logger.info("No dimensionality reducer provided, creating default TSNE", random_state=random_state, **dim_reductor_kwargs)

        # Check if method is specified in kwargs
        method = dim_reductor_kwargs.pop("method", "tsne")

        if method.lower() == "tsne":
            dim_reductor = TSNE(
                n_components=2,
                random_state=random_state,
                perplexity=min(30, X.shape[0] - 1),  # Adjust perplexity for small datasets
                max_iter=1000,
                learning_rate="auto",
                **dim_reductor_kwargs,
            )
        elif method.lower() == "pca":
            dim_reductor = PCA(n_components=2, random_state=random_state, **dim_reductor_kwargs)
        elif method.lower() == "kernelpca":
            dim_reductor = KernelPCA(n_components=2, random_state=random_state, **dim_reductor_kwargs)
        elif method.lower() == "mds":
            dim_reductor = MDS(n_components=2, random_state=random_state, **dim_reductor_kwargs)
        elif method.lower() == "isomap":
            dim_reductor = Isomap(n_components=2, **dim_reductor_kwargs)
        else:
            raise ValueError(f"Unknown method '{method}'. Supported methods as str: 'tsne', 'pca', 'kernelpca', `MDS`, `isomap`")

    # Validate dimensionality reducer
    if not hasattr(dim_reductor, "fit_transform"):
        raise TypeError(f"Dimensionality reducer must have 'fit_transform' method. Got {type(dim_reductor).__name__}")

    if dim_reductor.n_components != 2:
        raise ValueError(
            f"Dimensionality reducer must have exactly 2 components to plot 2D projection. "
            f"Got n_components={dim_reductor.n_components}"
        )

    logger.info(f"Fitting dimensionality reducer: {type(dim_reductor).__name__}")

    return dim_reductor


def prepare_data_for_boxplot(X: np.ndarray, sites: np.ndarray, feature_names: list[str] | None = None) -> pd.DataFrame:
    """Convert numpy array and sites to a tidy DataFrame suitable for seaborn boxplots.

    Parameters
    ----------
    X : np.ndarray
        Feature matrix of shape (n_samples, n_features)
    sites : np.ndarray
        Array of site labels for each sample, shape (n_samples,)
    feature_names : List[str], optional
        Names for each feature column. If None, uses 'Feature_0', 'Feature_1', etc.

    Returns
    -------
    pd.DataFrame
        Tidy DataFrame with columns: 'site', 'feature', 'value'

    """
    n_samples, n_features = X.shape

    # Create default feature names if not provided
    if feature_names is None:
        feature_names = [f"Feature_{i}" for i in range(n_features)]
    else:
        assert len(feature_names) == n_features, (
            f"Number of feature names ({len(feature_names)}) must match n_features ({n_features})"
        )

    # Reshape data to long format
    data_list = []
    for i in range(n_samples):
        for j in range(n_features):
            data_list.append({"site": sites[i], "feature": feature_names[j], "value": X[i, j]})

    return pd.DataFrame(data_list)


def plot_features_by_site(
    X: np.ndarray,
    sites: np.ndarray,
    feature_names: list[str] | None = None,
    figsize: tuple[int, int] = (14, 7),
    rotation: int = 45,
    show_points: bool = False,
    title: str | None = None,
    site_order: list[str] | None = None,
    feature_order: list[str] | None = None,
) -> tuple[plt.Figure, plt.Axes]:
    """Create boxplots with ALL features in the SAME plot, grouped by feature and colored by site.

    Parameters
    ----------
    X : np.ndarray
        Feature matrix of shape (n_samples, n_features)
    sites : np.ndarray
        Array of site labels for each sample
    feature_names : List[str], optional
        Names for each feature
    figsize : Tuple[int, int]
        Figure size (width, height)
    rotation : int
        Rotation angle for x-axis labels
    show_points : bool
        Whether to overlay individual data points on boxplots
    title : str, optional
        Custom title for the plot
    site_order : List[str], optional
        Order of sites in the legend
    feature_order : List[str], optional
        Order of features on x-axis

    Returns
    -------
    Tuple[plt.Figure, plt.Axes]
        Figure and axes objects for further customization

    """
    # Validate inputs
    X = check_array(X, copy=True, dtype=FLOAT_DTYPES)
    sites = check_array(sites, copy=True, dtype=None, ensure_2d=False)
    check_consistent_length(X, sites)
    validate_sites(sites)
    # Prepare data
    df = prepare_data_for_boxplot(X, sites, feature_names)

    # Set orders if provided
    if feature_order:
        df["feature"] = pd.Categorical(df["feature"], categories=feature_order, ordered=True)
    if site_order:
        df["site"] = pd.Categorical(df["site"], categories=site_order, ordered=True)

    # Create plot
    fig, ax = plt.subplots(figsize=figsize)

    # Create boxplot with hue by site - ALL FEATURES TOGETHER
    sns.boxplot(data=df, x="feature", y="value", hue="site", ax=ax, width=0.8)

    # Optionally add individual points
    if show_points:
        sns.stripplot(data=df, x="feature", y="value", hue="site", ax=ax, dodge=True, alpha=0.5, size=2, legend=False)

    # Customize plot
    ax.set_xlabel("Features", fontsize=12, fontweight="bold")
    ax.set_ylabel("Values", fontsize=12, fontweight="bold")
    ax.set_title(title or "Feature Distributions by Site (All Features Together)", fontsize=14, fontweight="bold")
    ax.tick_params(axis="x", rotation=rotation)
    ax.legend(title="Site", bbox_to_anchor=(1.05, 1), loc="upper left")
    ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()

    return fig, ax
