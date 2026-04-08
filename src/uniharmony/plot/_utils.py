"""Plotting utilities for UniHarmony examples."""

import numpy as np
import sklearn.linear_model
from matplotlib.axes import Axes


__all__ = ["plot_decision_boundary_2d"]


def plot_decision_boundary_2d(
    ax: Axes, clf: sklearn.linear_model._base.LinearClassifierMixin, linewidths: float = 1.0, alpha: float = 0.7
) -> None:
    """Plot decision boundary."""
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    xx, yy = np.meshgrid(np.linspace(xlim[0], xlim[1], 100), np.linspace(ylim[0], ylim[1], 100))
    Z = clf.decision_function(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    ax.contour(xx, yy, Z, levels=[0], colors="black", linewidths=linewidths, alpha=alpha)
