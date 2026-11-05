Pipelines
---------

Harmonizers need the site (and often covariates) of every sample at ``fit``
*and* at ``transform``. These examples show how to keep this metadata as
columns of a DataFrame so that UniHarmony works with every scikit-learn tool
(pipelines, cross-validation, grid search, scorers), with imbalanced-learn
samplers followed by a harmonizer, and as a julearn pipeline step. See
:doc:`/pipelines` for the reference.
