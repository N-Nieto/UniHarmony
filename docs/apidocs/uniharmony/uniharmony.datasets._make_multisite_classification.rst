:py:mod:`uniharmony.datasets._make_multisite_classification`
============================================================

.. py:module:: uniharmony.datasets._make_multisite_classification

.. autodoc2-docstring:: uniharmony.datasets._make_multisite_classification
   :parser: rst
   :allowtitles:

Module Contents
---------------

Functions
~~~~~~~~~

.. list-table::
   :class: autosummary longtable
   :align: left

   * - :py:obj:`make_multisite_classification <uniharmony.datasets._make_multisite_classification.make_multisite_classification>`
     - .. autodoc2-docstring:: uniharmony.datasets._make_multisite_classification.make_multisite_classification
          :parser: rst
          :summary:
   * - :py:obj:`_validate_parameters <uniharmony.datasets._make_multisite_classification._validate_parameters>`
     - .. autodoc2-docstring:: uniharmony.datasets._make_multisite_classification._validate_parameters
          :parser: rst
          :summary:
   * - :py:obj:`_get_default_balance_per_site <uniharmony.datasets._make_multisite_classification._get_default_balance_per_site>`
     - .. autodoc2-docstring:: uniharmony.datasets._make_multisite_classification._get_default_balance_per_site
          :parser: rst
          :summary:
   * - :py:obj:`_generate_binary_labels <uniharmony.datasets._make_multisite_classification._generate_binary_labels>`
     - .. autodoc2-docstring:: uniharmony.datasets._make_multisite_classification._generate_binary_labels
          :parser: rst
          :summary:
   * - :py:obj:`_generate_multiclass_labels <uniharmony.datasets._make_multisite_classification._generate_multiclass_labels>`
     - .. autodoc2-docstring:: uniharmony.datasets._make_multisite_classification._generate_multiclass_labels
          :parser: rst
          :summary:
   * - :py:obj:`_generate_signal_component <uniharmony.datasets._make_multisite_classification._generate_signal_component>`
     - .. autodoc2-docstring:: uniharmony.datasets._make_multisite_classification._generate_signal_component
          :parser: rst
          :summary:
   * - :py:obj:`_validate_balance_per_site <uniharmony.datasets._make_multisite_classification._validate_balance_per_site>`
     - .. autodoc2-docstring:: uniharmony.datasets._make_multisite_classification._validate_balance_per_site
          :parser: rst
          :summary:
   * - :py:obj:`_check_balance_for_binary_classification <uniharmony.datasets._make_multisite_classification._check_balance_for_binary_classification>`
     - .. autodoc2-docstring:: uniharmony.datasets._make_multisite_classification._check_balance_for_binary_classification
          :parser: rst
          :summary:
   * - :py:obj:`_check_balance_for_multiclass <uniharmony.datasets._make_multisite_classification._check_balance_for_multiclass>`
     - .. autodoc2-docstring:: uniharmony.datasets._make_multisite_classification._check_balance_for_multiclass
          :parser: rst
          :summary:

Data
~~~~

.. list-table::
   :class: autosummary longtable
   :align: left

   * - :py:obj:`__all__ <uniharmony.datasets._make_multisite_classification.__all__>`
     - .. autodoc2-docstring:: uniharmony.datasets._make_multisite_classification.__all__
          :parser: rst
          :summary:
   * - :py:obj:`logger <uniharmony.datasets._make_multisite_classification.logger>`
     - .. autodoc2-docstring:: uniharmony.datasets._make_multisite_classification.logger
          :parser: rst
          :summary:

API
~~~

.. py:data:: __all__
   :canonical: uniharmony.datasets._make_multisite_classification.__all__
   :value: ['make_multisite_classification']

   .. autodoc2-docstring:: uniharmony.datasets._make_multisite_classification.__all__
      :parser: rst

.. py:data:: logger
   :canonical: uniharmony.datasets._make_multisite_classification.logger
   :value: 'get_logger(...)'

   .. autodoc2-docstring:: uniharmony.datasets._make_multisite_classification.logger
      :parser: rst

.. py:function:: make_multisite_classification(n_classes: int = 2, n_sites: int = 2, n_samples: int = 1000, balance_per_site: list[float] | list[list[float]] | None = None, n_features: int = 10, signal_strength: float = 1.0, noise_strength: float = 1.0, site_effect_strength: float = 3.0, site_effect_homogeneous: bool = True, random_state: int | numpy.random.RandomState = 42) -> tuple[numpy.ndarray, numpy.ndarray, numpy.ndarray]
   :canonical: uniharmony.datasets._make_multisite_classification.make_multisite_classification

   .. autodoc2-docstring:: uniharmony.datasets._make_multisite_classification.make_multisite_classification
      :parser: rst

.. py:function:: _validate_parameters(n_sites: int, n_samples: int, n_features: int, signal_strength: float, noise_strength: float, site_effect_strength: float, n_classes: int) -> None
   :canonical: uniharmony.datasets._make_multisite_classification._validate_parameters

   .. autodoc2-docstring:: uniharmony.datasets._make_multisite_classification._validate_parameters
      :parser: rst

.. py:function:: _get_default_balance_per_site(n_sites: int, n_classes: int) -> list[float] | list[list[float]]
   :canonical: uniharmony.datasets._make_multisite_classification._get_default_balance_per_site

   .. autodoc2-docstring:: uniharmony.datasets._make_multisite_classification._get_default_balance_per_site
      :parser: rst

.. py:function:: _generate_binary_labels(n_samples: int, p_class_1: float, random_state: numpy.random.RandomState) -> numpy.ndarray
   :canonical: uniharmony.datasets._make_multisite_classification._generate_binary_labels

   .. autodoc2-docstring:: uniharmony.datasets._make_multisite_classification._generate_binary_labels
      :parser: rst

.. py:function:: _generate_multiclass_labels(n_samples: int, class_probs: list[float], n_classes: int, random_state: numpy.random.RandomState) -> numpy.ndarray
   :canonical: uniharmony.datasets._make_multisite_classification._generate_multiclass_labels

   .. autodoc2-docstring:: uniharmony.datasets._make_multisite_classification._generate_multiclass_labels
      :parser: rst

.. py:function:: _generate_signal_component(y: numpy.ndarray, n_features: int, signal_strength: float, n_classes: int, random_state: numpy.random.RandomState) -> numpy.ndarray
   :canonical: uniharmony.datasets._make_multisite_classification._generate_signal_component

   .. autodoc2-docstring:: uniharmony.datasets._make_multisite_classification._generate_signal_component
      :parser: rst

.. py:function:: _validate_balance_per_site(balance_per_site: list | list[list] | None, n_sites: int, n_classes: int) -> list | list[list]
   :canonical: uniharmony.datasets._make_multisite_classification._validate_balance_per_site

   .. autodoc2-docstring:: uniharmony.datasets._make_multisite_classification._validate_balance_per_site
      :parser: rst

.. py:function:: _check_balance_for_binary_classification(balance_per_site: list[float]) -> None
   :canonical: uniharmony.datasets._make_multisite_classification._check_balance_for_binary_classification

   .. autodoc2-docstring:: uniharmony.datasets._make_multisite_classification._check_balance_for_binary_classification
      :parser: rst

.. py:function:: _check_balance_for_multiclass(balance_per_site: list | list[list] | tuple, n_classes: int) -> None
   :canonical: uniharmony.datasets._make_multisite_classification._check_balance_for_multiclass

   .. autodoc2-docstring:: uniharmony.datasets._make_multisite_classification._check_balance_for_multiclass
      :parser: rst
