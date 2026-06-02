# %%

from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from uniharmony import verbosity
verbosity("error")
# %%
from uniharmony.combat import NeuroComBat
from uniharmony.datasets import make_multisite_classification
harm_model =  NeuroComBat().set_fit_request(sites=True).set_transform_request(sites=True)

pipe =  Pipeline([("harm", harm_model), ("pca", PCA())])

X, y, sites =  make_multisite_classification()

pipe.fit_transform(X, y, sites=sites)
