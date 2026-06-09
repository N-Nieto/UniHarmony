(otda)=
# Optimal Transport Domain Adaptation (OTDA)

[Optimal Transport](https://en.wikipedia.org/wiki/Optimal_transport) provides a geometric framework for aligning probability distributions by minimizing the cost of moving mass from one distribution to another [^1]. When applied to domain adaptation, OT learns a **transport map** that transforms data from a source domain (e.g., one scanner site) to match the distribution of a target domain (e.g., a reference site).

Unlike ComBat-based methods that assume parametric location-scale shifts, OTDA makes **no distributional assumptions** and can align arbitrarily complex site differences, including non-linear distortions and multi-modal distributions.

## Method

OTDA finds a **coupling matrix** $\mathbf{P}$ that minimizes transport cost between source and target samples:

$$

\min_{\mathbf{P}} \langle \mathbf{P}, \mathbf{C} \rangle + \lambda \cdot \text{regularization}(\mathbf{P})

$$

where $\mathbf{C}$ is the cost matrix (e.g., Euclidean distance) and $\lambda$ controls regularization (e.g., entropic smoothing for Sinkhorn).

After fitting $\mathbf{P}$, source samples are transformed via **barycentric projection**:

$$

\mathbf{X}_{\text{transformed}} = \text{diag}(\mathbf{P} \mathbf{1})^{-1} \mathbf{P} \mathbf{X}_{\text{target}}

$$

## Key features

| Aspect | Detail |
|--------|--------|
| **Flexible transport** | Supports EMD, Sinkhorn, Sinkhorn with group Lasso, and Laplace regularization |
| **Supervised mode** | Uses labels to guide same-class transport (lower cost between matched classes) |
| **Multi-reference** | Can align to multiple reference sites combined |
| **Distribution-free** | No normality or linearity assumptions |

## Parameters

| Parameter | Options | Default | Description |
|-----------|---------|---------|-------------|
| `ot_method` | "emd", "sinkhorn"/"s", "sinkhorn_gl"/"s_gl", "emd_laplace"/"emd_l" | "emd" | Transport algorithm |
| `metric` | Any scipy distance metric | "euclidean" | Cost function |
| `reg` | float | 1.0 | Entropic regularization (Sinkhorn) |
| `eta` | float | 0.1 | Group Lasso regularization |
| `limit_max` | int or None | 10 | Sets infinite cost for different classes (semi-supervised) |

## Example

```python
from uniharmony.ot import OptimalTransportDomainAdaptation

otda = OptimalTransportDomainAdaptation(
    ot_method="sinkhorn",
    metric="sqeuclidean",
    reg=0.1
)

# Fit with reference site
otda.fit(X_train, sites_train, ref_site="site_A", y=labels_train)

# Harmonize new data
X_harmonized = otda.transform(X_test, sites_test)
