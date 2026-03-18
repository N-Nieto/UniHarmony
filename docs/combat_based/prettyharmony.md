# PrettYharmonize


## Cross-site class imbalance

A common challenge in multi-site datasets occurs when **class distributions differ across sites**. For example:

- Site A contains mostly control subjects  
- Site B contains mostly patients

In such cases, **biological signal and site effects become confounded**.

If harmonization removes site effects without accounting for the biological variable, the biological signal may also be partially removed.

To address this, ComBat allows users to include biological variables as **covariates to preserve** during harmonization.

However, this introduces an issue in **machine-learning pipelines**.

If the covariate being preserved is also the **target variable of the ML model**, then the target must be known during harmonization. In real-world prediction scenarios this information is unavailable, creating a new form of **data leakage** ${^6}$.

Even if harmonization parameters are estimated using only the training set, the transformation of test data still requires the target variable.

A suitable alternative in these scenarios is **PrettYharmonize**, which allows ComBat-based harmonization to be integrated into ML pipelines **without requiring the target variable during inference**.
