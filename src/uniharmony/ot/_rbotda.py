"""Regularized Backward Optimal Transport for Domain Adaptation (rBOTDA) implementation."""
# # @author: Nieto Nicolás
# # @email: nnieto@sinc.unl.edu.ar /  n.nieto@fz-juelich.de

# import numpy.typing as npt
# from ot.da import BaseTransport
# from sklearn.base import ClassifierMixin

# from ._utils import data_consistency_check, naming_check
# from .balancing_functions import (
#     balance_samples,
#     deal_with_wrong_classified,
#     initialize_uniform_weights,
#     subsample_set,
# )
# from .ot_helper_functions import compute_backward_coupling, compute_cost_matrix, initialize_ot_obj
# from .penalization_functions import compute_penalization


# __all__ = ["rBOTDA"]


# class rBOTDA(BaseTransport, ClassifierMixin):
#     """Regularized Backward optimal Transport (rBOTDA) class.

#     The regularization version of BOTDA has three main improvements:

#     1) The possibility to use the already trained classifier to improve the transport.
#        The points that were wrongly classified can be removed using "wrong_cls"

#        The points that are closer to the classifier decision boundlry are penalized using "k" and "penalized_type"

#     2) Possibility to change the mass for each class in both domains.
#         Use the parameters "balanced_train" and "balanced_val"
#     """

#     def __init__(
#         self,
#         k: int,
#         ot_method: str = "emd",
#         metric: str = "euclidean",
#         penalized_type: str = "probability",
#         wrong_cls: bool = True,
#         balanced_train="auto",
#         balanced_val="auto",
#         train_size="all",
#         reg: float | None = 1,
#         eta: float | None = 0.1,
#         max_iter: int | None = 10,
#         cost_norm: bool | None = None,
#         limit_max: int | None = 10,
#         cost_supervised: bool | None = True,
#     ) -> None:
#         """Initialize the rBOTDA object.

#         Args:
#             k (int): penalization strength. If k=0 no penalization applied.

#             ot_method (str, optional): Optimal transport method applied. Defaults to "emd".
#                 Supported: "emd" / Earth Movers distance
#                            "s" / Sinkhorn
#                            "s_gl" / Sinkhorn Group Lasso
#                            "emd_l" / Laplace Earth Movers distance

#             metric (str, optional): Distance metric. Defaults to "euclidean".

#             From POT:
#                 'sqeuclidean' or 'euclidean' on all backends.
#                 On numpy the function also accepts from
#                 the scipy.spatial.distance.cdist function :
#                 'braycurtis', 'canberra', 'chebyshev', 'cityblock', 'correlation', 'cosine', 'dice', 'euclidean', 'hamming',
#                 'jaccard', 'kulsinski', 'mahalanobis', 'matching', 'minkowski', 'rogerstanimoto', 'russellrao', 'seuclidean',
#                 'sokalmichener','sokalsneath', 'sqeuclidean', 'wminkowski', 'yule'.


#             penalized_type (str, optional): Penalize the samples by a metric. Defaults to "p".
#                 Supported: "p" - Probability. sklearn classifier must have the
#                                               "predict_proba" function
#                                               Only possible for binary
#                                               classification
#                            "d" - Hyperplane distance (only linear classifiers)
#                                  Classifier must have "intercept_" and
#                                  "coef_" attributes

#             wrong_cls (bool, optional): Delete points wrong classified on train. Defaults to True.

#             balanced_train (Any, optional): Balance the train set. Defaults to "auto".

#                 Supported: "auto" - Balance the train domain using train labels
#                                     The mass of the point for each class must sum 1/Number Classes.

#                             None - No balance the train domain. All points will have the same mass

#                             [] - List containing the sum of mass for each class
#                                  Lenght of list = number of classes
#                                  Sum of elements in list must be 1
#                                  For example, if [0.6, 0.4] is provided, the mass of the class 1 will sum 0.6
#                                  while the ones for class 2 will sum 0.4.

#             balanced_val (Any, optional): Balance the validation set. Defaults to "auto".

#                 Supported: "auto" - Balance the val domain using val labels
#                                     The mass of the point for each class must
#                                     sum 1/Number Classes.

#                             None - No balance the val domain.
#                                    All points will have the same mass

#                             [] - List containing the sum of mass for each class
#                                  Lenght of list = number of classes
#                                  Sum of elements in list must be 1
#                                  For example, if [0.6, 0.4] is provided, the mass of the class 1 will sum 0.6
#                                  while the ones for class 2 will sum 0.4.

#             train_size (Any, optional): Control the size of the training set. Default "all"

#                 Supported: "all" - Use all training samples

#                             [] - List containing the number of samples used for each class.

#             reg (float, optional): Regularization Parameter. Defaults to 1.
#                                      Only used when ot_method = "s" or "s_gl"

#             eta (float, optional): Regularization Parameter. Defaults to 1.

#                                    Only used when ot_method = "s_gl" or "emd_l"
#             max_iter (int, optional): (from POT) the maximum numer of iteration before stopping the optimization procedure
#             cost_norm (bool, optional): Normalize the cost matrix
#                                         Defaults to None.

#             From POT: Type of normalization from 'median', 'max', 'log',
#                       'loglog'. Any other value do not normalize

#             limit_max (int, optional):  Controls the semi supervised mode.
#                                         Transport between labeled source and
#                                         target samples of different classes
#                                         will exhibit an infinite cost (10 times
#                                         the maximum value of the cost matrix)
#                                         Defaults to 10.

#             cost_supervised (bool, optional): Supervise the cost matrix. Defaults to True.
#                                               This allows the user to calculate the cost matriz unsupervised even
#                                               if the train and val labels are provided

#         """
#         # Initialize any hyperparameters for OT
#         # Optimal Transport Method. EMD, S, GL
#         self.ot_method = ot_method
#         # Distance metric to compute transport plan
#         self.metric = metric
#         # Parameter for regularized version of OT
#         self.reg = reg
#         self.eta = eta
#         # Cost norm and limit max of OT
#         self.cost_norm = cost_norm
#         self.limit_max = limit_max
#         self.cost_supervised = cost_supervised
#         self.max_iter = max_iter
#         # Initialize any hyperparameters for penalization
#         # Type of penalization (Distance or classifier probability)
#         self.penalized_type = penalized_type
#         # Penalization intensity. Is and hyperparameter to tune
#         # If k=0 no penalization is applied
#         self.k = k
#         # If removing points wrong classified before applied transport.
#         # Poinsts will be assigned with 0 mass.
#         self.wrong_cls = wrong_cls

#         # Initialize any hyperparameters for balance
#         # Balancing the target and source samples.
#         # if "auto" is passed, the target mass will be normalized
#         # so the sum of all point's mass sum 1/n_classes.
#         # A vector with different proportions can be also passed as parameter.
#         # The vector muss sum one, and then the points for the corresponding
#         # class will sum up until the proportion passed.
#         self.balanced_train = balanced_train
#         self.balanced_val = balanced_val

#         # Allows to control the train size. If "all" all samples are used.
#         # if not, a list with the number of samples by class must be provided
#         self.train_size = train_size
#         self.ot_obj = initialize_ot_obj(ot_method)

#     def fit(
#         self,
#         X_train: npt.ArrayLike,
#         X_val: npt.ArrayLike,
#         clf,
#         y_train: npt.ArrayLike | None = None,
#         y_val: npt.ArrayLike | None = None,
#     ):
#         """Fit optimal transport method with penalization.

#         Classifier has to be trained on Xs as in BOTDA method.

#         Parameters
#         ----------
#         X_train : array-like, shape (n_samples, n_features)
#             Training Samples / Domain where the classifier was trained
#             This domain is the source domain for the transport

#         X_val : array-like, shape (n_samples, n_features)
#             Samples to be transported / This domain is the test doman
#             This domain is the target domain for the transport

#         clf : Already trained classifier. Sklearn classifier instance.
#             This is used to compute the penalization of the samples.
#             The samples that are closer to the decision boundary will be more penalized.

#         y_train : array-like, shape (n_samples,) or (n_samples, n_classes), optional
#             Train labels. default (None)
#             This is used for the penalization and balancing of the train domain.
#             If not provided, no penalization or balancing will be applied to the train domain.

#         y_val : array-like, shape (n_samples,) or (n_samples, n_classes), optional
#             Validation labels.  default (None)
#             This is used for the penalization and balancing of the validation domain.
#             If not provided, no penalization or balancing will be applied to the validation domain.

#         Returns
#         -------
#         ot_obj :  Fitted Optimal transport instance

#         """
#         # check naming
#         naming_check(ot_method=self.ot_method, penalized_type=self.penalized_type)

#         # Check consistency
#         data_consistency_check(X_train=X_train, X_val=X_val, y_train=y_train, y_val=y_val)

#         mass_train, mass_val = initialize_uniform_weights(X_train=X_train, X_val=X_val)

#         # If source (train), labels are provided, then enter to the function
#         # that allows to remove the wrongly classified points in train
#         if self.wrong_cls:
#             # Deal with wrong classified point in the target domain
#             X_train, y_train, mass_train = deal_with_wrong_classified(X_train, y_train, mass_train, clf)

#         # Change the weights of the target points with respect a penalization
#         mass_train = compute_penalization(self.penalized_type, self.k, mass_train, X_train, clf)

#         if self.train_size != "all":
#             X_train, y_train, mass_train = subsample_set(X_train, y_train, mass_train, self.train_size)

#         # Change the weights of the target points with respect a penalization
#         mass_train = balance_samples(self.balanced_train, mass_train, y_train)

#         mass_val = balance_samples(self.balanced_val, mass_val, y_val)

#         # Compute cost matrix
#         M = compute_cost_matrix(self, X_train=X_train, y_train=y_train, X_val=X_val, y_val=y_val)

#         # Compute coupling with different OT methods in a backward way
#         G0 = compute_backward_coupling(self, mass_train, mass_val, M, X_train=X_train, X_val=X_val, y_val=y_val)

#         # Replace the coupling with the penalized one
#         self.ot_obj.coupling_ = G0
#         self.ot_obj.cost_ = M

#         self.ot_obj.mu_t = mass_train
#         self.ot_obj.mu_s = mass_val

#         # store arrays of samples
#         self.ot_obj.xt_ = X_train
#         self.ot_obj.xs_ = X_val

#         return self

#     def transform(
#         self,
#         Xs: npt.ArrayLike,
#         Xt: npt.ArrayLike | None = None,
#         ys: npt.ArrayLike | None = None,
#         yt: npt.ArrayLike | None = None,
#         batch_size: int = 128,
#     ) -> npt.ArrayLike:
#         """Transform data using the fitted OT Element.

#         Args:
#             X (ArrayLike): Input data to be transformed.

#         Returns:
#             ArrayLike: Transformed data.

#         """
#         # Transport samples from target to source using the fitted OT Element
#         X_transformed = self.ot_obj.transform(Xs=Xs, ys=ys, Xt=Xt, yt=yt, batch_size=batch_size)
#         return X_transformed


# import numpy as np
# import numpy.typing as npt
# from sklearn.base import ClassifierMixin


# def balance_weights(y: npt.ArrayLike, weights: npt.ArrayLike, balance: list[int | float]) -> np.ndarray:
#     """Balance the weights of samples based on the specified balance strategy for different classes.

#     Parameters
#     ----------
#     y : npt.ArrayLike
#         Target labels for class-specific balancing.
#     weights : npt.ArrayLike
#         Input weights to be balanced.
#     balance : list[int | float]
#         List of weights for
#         balancing different classes. If "auto," uniform relevance for each class.

#     Returns
#     -------
#     np.ndarray: Balanced weights for the samples.

#     """
#     # Get unique classes from labels
#     classes = np.unique(y)
#     if balance == "auto":
#         # All classes sum the same
#         balance = 1 / len(classes) * np.ones(classes.shape)
#     elif not (sum(balance) == 1):
#         raise ValueError("Relevance vector needs to sum to 1")

#     # Initialize the new weights
#     w_final = np.zeros(weights.shape)
#     balance = np.array(balance)

#     for cl in classes:
#         mask = np.array(classes == cl)
#         rel = balance[mask]

#         # Total points
#         w_cl = weights.copy()

#         # Keep only the points for one class
#         w_cl[y != cl] = 0

#         # In the case that all points in one class have 0 weight
#         # (i.e., if all were misclassified)
#         if sum(w_cl) == 0:
#             w_cl = np.ones((len(w_cl),)) / len(w_cl)

#         # Normalize the weights with respect to the balance of
#         # the class and the total mass of the class
#         w_cl = w_cl * rel / (sum(w_cl))

#         # Sum the weight of the particular class to the final weight vector
#         w_final = w_final + w_cl

#     return w_final


# def initialize_uniform_weights(X_train: npt.ArrayLike, X_val: npt.ArrayLike) -> tuple[npt.ArrayLike, npt.ArrayLike]:
#     """Initialize uniform weights for train and validation data.

#     Parameters
#     ----------
#     X_train : array-like, shape (n_samples, n_features)
#         Train data.

#     X_val : array-like, shape (n_samples, n_features)
#         Validation data.

#     Returns
#     -------
#     tuple: Tuple containing train and val uniform weights.

#     """
#     # Start train with uniform weights
#     mass_train = np.ones((X_train.shape[0],)) / X_train.shape[0]

#     # Start target with uniform weights
#     mass_val = np.ones((X_val.shape[0],)) / X_val.shape[0]

#     return mass_train, mass_val


# def balance_samples(balance: str | list[int | float] | None, mass: npt.ArrayLike, y: npt.ArrayLike | None) -> np.ndarray:
#     """Balance the weights of samples based on the specified balance strategy.

#     Parameters
#     ----------
#     balance :  ([str orlList) Balance strategy.
#                 Can be "auto" for uniform relevance for each class
#                 or a list of weights.
#     mass : npt.ArrayLike
#         Input weights to be balanced.
#     y : npt.ArrayLike
#         Target labels for class-specific balancing.

#     Returns
#     -------
#     np.ndarray: Balanced weights for the samples.

#     """
#     if balance is None:
#         balanced_sampes = mass

#     # If "auto" use uniform relevance for each classs
#     elif balance == "auto":
#         if y is not None:
#             # Check the y was provided for this type of balancing
#             balanced_sampes = balance_weights(y, mass, balance)
#         else:
#             raise ValueError("label must be provided for balancing")
#     # If the first element is int or float
#     elif isinstance(balance[0], (int, float)):
#         if y is not None:
#             balanced_sampes = balance_weights(y, mass, balance)
#         else:
#             raise ValueError("label must be provided for balancing")
#     else:
#         raise Exception("Balance target not supported")

#     # In any case, make sure the samples sum is 1.
#     balanced_sampes = balanced_sampes / sum(balanced_sampes)
#     return balanced_sampes


# def deal_with_wrong_classified(
#     X_train: npt.ArrayLike, y_train: npt.ArrayLike, mass_train: npt.ArrayLike, clf: ClassifierMixin
# ) -> tuple[npt.ArrayLike, npt.ArrayLike, npt.ArrayLike]:
#     """Handle wrongly classified points in train data.

#     Parameters
#     ----------
#     X_train : array-like, shape (n_samples, n_features)
#         Train data.
#     y_train : npt.ArrayLike
#         True labels for train data.
#     mass_train : npt.ArrayLike
#         Weights associated with each data point.
#     clf : ClassifierMixin
#         The trained classifier.

#     Returns
#     -------
#     Tuple[ArrayLike, ArrayLike, ArrayLike]: Tuple containing filtered
#                                             X_train, y_train, and
#                                             updated weights (mass_train).

#     """
#     if y_train is None:
#         ValueError("Train labels must be provided to delet wrong classified")
#     # Generate prediction over train data
#     y_pred = clf.predict(X_train)

#     # Check if we do not delete all points for one class
#     if len(np.unique(y_train[y_train == y_pred])) < 2:
#         Warning("All points for one class wrongly classified, continuing without removing wrong classified")
#     else:
#         # Delet the points from the Xs, a and target
#         X_train = X_train[y_train == y_pred]
#         mass_train = mass_train[y_train == y_pred]
#         y_train = y_train[y_train == y_pred]

#         # If all the datapoins were missclassified
#         if np.isnan(np.sum(mass_train)):
#             # Target uniform weights
#             mass_train = np.ones(((X_train.shape[0]),)) / (X_train.shape[0])

#     return X_train, y_train, mass_train


# def subsample_set(
#     X: npt.ArrayLike, y: npt.ArrayLike, mass: npt.ArrayLike, train_size: list[int]
# ) -> tuple[npt.ArrayLike, npt.ArrayLike, npt.ArrayLike]:
#     """Extract data points with the highest mass for each class.

#     Parameters
#     ----------
#     X : npt.ArrayLike
#         Feature matrix.
#     y : npt.ArrayLike
#         Array of class labels.
#     mass : npt.ArrayLike
#         Array of mass values for each data point.
#     train_size : list[int]
#         List specifying the number of data points to be extracted for each class.

#     Returns
#     -------
#     - tuple[npt.ArrayLike, npt.ArrayLike, npt.ArrayLike]: Extracted data (X, y, mass).


#     """
#     # Dimension checks
#     unique_classes = np.unique(y)
#     # Initialization
#     X_f = []
#     y_f = []
#     mass_f = []

#     for class_label, size in zip(unique_classes, train_size, strict=True):
#         mask = y == class_label
#         # Get indices of highest mass values
#         sorted_indices = np.argsort(mass[mask])[::-1]
#         selected_indices = np.where(mask)[0][sorted_indices][:size]

#         X_f.extend(X[selected_indices])
#         y_f.extend(y[selected_indices])
#         mass_f.extend(mass[selected_indices])

#     return np.array(X_f), np.array(y_f), np.array(mass_f)


# import numpy as np
# import numpy.typing as npt
# import ot


# def compute_cost_matrix(
#     self, X_train: npt.ArrayLike, X_val: npt.ArrayLike, y_train: npt.ArrayLike | None = None, y_val: npt.ArrayLike | None = None
# ) -> npt.ArrayLike:
#     """Compute the cost matrix for optimal transport between source (val) and target (train) domains.

#     Args:
#         X_train (npt.ArrayLike): Target domain samples.
#         X_val (npt.ArrayLike): Source domain samples.
#         y_train (Optional[npt.ArrayLike]): Target domain labels.
#         y_val (Optional[npt.ArrayLike]): Source domain labels.

#     Returns:
#         npt.ArrayLike: Cost matrix for optimal transport.

#     """
#     # Pairwise distance computation
#     M = ot.dist(X_val, X_train, metric=self.metric)

#     # Normalize the cost matrix
#     M = ot.utils.cost_normalization(M, self.cost_norm)

#     if (y_train is not None) and (y_val is not None) and (self.cost_supervised):
#         # Apply cost adjustments for supervised training
#         if self.limit_max != np.inf:
#             limit_max = self.limit_max * np.max(M)

#         # Assumes labeled source samples occupy the first rows
#         # and labeled target samples occupy the first columns
#         classes = [c for c in np.unique(y_val) if c != -1]
#         for c in classes:
#             idx_s = np.where((y_val != c) & (y_val != -1))
#             idx_t = np.where(y_train == c)

#             # Set the coefficients corresponding to a source sample
#             # and a target sample with different labels to infinity
#             for j in idx_t[0]:
#                 M[idx_s[0], j] = limit_max

#     return M


# def compute_backward_coupling(
#     self,
#     mass_train: npt.ArrayLike,
#     mass_val: npt.ArrayLike,
#     M: npt.ArrayLike,
#     X_val: npt.ArrayLike,
#     y_val: npt.ArrayLike,
#     X_train: npt.ArrayLike,
# ) -> npt.ArrayLike:
#     """Compute the backward coupling between source and target samples.

#     This methods lears to transport samples from the validation domain (source)
#     to the train domain (target)

#     Args:
#         mass_train (npt.ArrayLike): Source sample weights.
#         mass_val (npt.ArrayLike): Target sample weights.
#         M (npt.ArrayLike): Cost matrix.
#         X_val (npt.ArrayLike): Source samples.
#         yt (npt.ArrayLike): Target sample labels.
#         X_train (npt.ArrayLike): Target samples.


#     Returns:
#         npt.ArrayLike: Backward coupling matrix.

#     """
#     # Check if the OT method is supported
#     if self.ot_method == "emd":
#         # Earth Mover's Distance (EMD) coupling
#         G0 = ot.da.emd(a=mass_val, b=mass_train, M=M)

#     elif self.ot_method in ["sinkhorn", "s"]:
#         # Sinkhorn coupling
#         G0 = ot.da.sinkhorn(a=mass_val, labels_a=y_val, b=mass_train, M=M, reg=self.reg)

#     elif self.ot_method in ["sinkhorn_gl", "s_gl"]:
#         # Sinkhorn coupling with Group L1L2 regularization
#         G0 = ot.da.sinkhorn_l1l2_gl(
#             a=mass_val, labels_a=y_val, b=mass_train, M=M, reg=self.reg, eta=self.eta, numItermax=self.max_iter
#         )

#     elif self.ot_method in ["emd_laplace", "emd_l"]:
#         # EMD coupling with Laplace regularization
#         G0 = ot.da.emd_laplace(a=mass_val, b=mass_train, Xs=X_val, Xt=X_train, M=M, eta=self.eta)

#     else:
#         # Raise an error if the OT method is not supported
#         raise RuntimeError("OT method not supported")

#     return G0


# """Helper functions for penalization in the rBOTDA algorithm."""

# import numpy as np
# import numpy.typing as npt
# from sklearn.linear_model import LinearRegression


# def distance_to_hyperplane(X: npt.ArrayLike, clf: LinearRegression) -> npt.ArrayLike:
#     """Calculate the distance of data points to the hyperplane defined by a classifier.

#     Parameters
#     ----------
#     X : (array-like), shape (samples x features).
#         Input data points for which to compute the distance to the hyperplane.
#     clf : (classifier) The trained classifier.
#         sdads

#     Returns
#     -------
#     array-like: Array of distances from data points to the hyperplane.

#     """
#     # Get the intercept and coefficients from the classifier
#     b = clf.intercept_
#     W = clf.coef_

#     # Calculate the module of the coefficients
#     mod = np.sqrt(np.sum(np.power(W, 2)))

#     # Calculate the distance from data points to the hyperplane
#     d = np.abs(np.dot(X, W.T) + b) / mod

#     return d[:, 0]


# def compute_penalization(penalized_type: str, k: int, mass: npt.ArrayLike, X_train: npt.ArrayLike, clf: object)
# -> npt.ArrayLike:
#     """Compute the penalization with respect to the classifier.

#     The penalization could be inversely proportional to the distance
#     from the samples to the decision hyperplane (only linear classifiers)
#     or with respect to the probability output of the classifier.

#     Parameters
#     ----------
#     penalized_type : str
#         Type of penalization ("distance" or "probability").
#     k : int
#         Power parameter for distance transformation.
#     mass : ArrayLike
#         Input samples weights.
#     X_train : ArrayLike
#         Input training data.
#     clf : Optional[object]
#         The trained classifier.

#     Returns
#     -------
#     np.ndarray: Updated samples weights based on the computed penalization.

#     """
#     # Calculate the distance of each sample to the LDA decision straight
#     if penalized_type in ["distance", "d"]:
#         penalization = dist_matrix(X=X_train, clf=clf, k=k)
#     elif penalized_type in ["probability", "proba", "p"]:
#         penalization = proba_matrix(X=X_train, clf=clf, k=k)
#     else:
#         raise Exception("Penalization not supported")

#     # Change the sample weight proportionally to the computed score
#     mass = np.dot(mass, penalization)

#     return mass


# def dist_matrix(X: npt.ArrayLike, clf: object, k: int) -> npt.ArrayLike:
#     """Compute the matrix Q based on the distance to the hyperplane defined by a classifier.

#     Parameters
#     ----------
#     X : npt.ArrayLike
#         Input data.
#     clf : object
#         The trained classifier.
#     k : int
#         Power parameter for distance transformation.

#     Returns
#     -------
#     np.ndarray: Matrix Q based on the distance to the hyperplane.

#     """
#     # Compute distance
#     d = distance_to_hyperplane(X, clf)
#     d = np.power(d, k)

#     # Normalization term
#     nom = np.prod(np.power(d, 1 / len(d)))

#     # Penalization is proportional to the distance to
#     # the classifier decision boundary
#     penalization = d / nom
#     penalization = np.diag(penalization)

#     return penalization


# def proba_matrix(X: npt.ArrayLike, clf: object, k: int) -> npt.ArrayLike:
#     """Compute the matrix of probabilities based on the output probabilities from a classifier.

#     Parameters
#     ----------
#     X : npt.ArrayLike
#         Input data.
#     clf : object
#         The trained classifier.
#     k : int
#         Power parameter for distance transformation.

#     Returns
#     -------
#     np.ndarray: Matrix of probabilities based on the classifier's output.

#     """
#     # Get the probabilities for each point
#     d = clf.predict_proba(X)

#     # Subtract the mean for each point and compute the absolute value
#     d = np.abs(d - np.mean(d) * np.ones(d.shape))
#     d = d.sum(axis=1)
#     d = np.power(d, k)
#     d = d + 1e-10

#     # Normalization term
#     nom = np.prod(np.power(d, 1 / len(d)))

#     # penalization is proportional to the distance to
#     # the classifier decision boundary
#     penalization = d / nom
#     penalization = np.diag(penalization)

#     return penalization
