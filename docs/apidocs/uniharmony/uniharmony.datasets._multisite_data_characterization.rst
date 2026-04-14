:py:mod:`uniharmony.datasets._multisite_data_characterization`
==============================================================

.. py:module:: uniharmony.datasets._multisite_data_characterization

.. autodoc2-docstring:: uniharmony.datasets._multisite_data_characterization
   :parser: rst
   :allowtitles:

Module Contents
---------------

Functions
~~~~~~~~~

.. list-table::
   :class: autosummary longtable
   :align: left

   * - :py:obj:`get_site_data_statistics <uniharmony.datasets._multisite_data_characterization.get_site_data_statistics>`
     - .. autodoc2-docstring:: uniharmony.datasets._multisite_data_characterization.get_site_data_statistics
          :parser: rst
          :summary:
   * - :py:obj:`_validate_inputs <uniharmony.datasets._multisite_data_characterization._validate_inputs>`
     - .. autodoc2-docstring:: uniharmony.datasets._multisite_data_characterization._validate_inputs
          :parser: rst
          :summary:
   * - :py:obj:`_validate_array_types <uniharmony.datasets._multisite_data_characterization._validate_array_types>`
     - .. autodoc2-docstring:: uniharmony.datasets._multisite_data_characterization._validate_array_types
          :parser: rst
          :summary:
   * - :py:obj:`_validate_array_shapes <uniharmony.datasets._multisite_data_characterization._validate_array_shapes>`
     - .. autodoc2-docstring:: uniharmony.datasets._multisite_data_characterization._validate_array_shapes
          :parser: rst
          :summary:
   * - :py:obj:`_validate_array_dimensions <uniharmony.datasets._multisite_data_characterization._validate_array_dimensions>`
     - .. autodoc2-docstring:: uniharmony.datasets._multisite_data_characterization._validate_array_dimensions
          :parser: rst
          :summary:
   * - :py:obj:`_validate_array_values <uniharmony.datasets._multisite_data_characterization._validate_array_values>`
     - .. autodoc2-docstring:: uniharmony.datasets._multisite_data_characterization._validate_array_values
          :parser: rst
          :summary:
   * - :py:obj:`_compute_overall_statistics <uniharmony.datasets._multisite_data_characterization._compute_overall_statistics>`
     - .. autodoc2-docstring:: uniharmony.datasets._multisite_data_characterization._compute_overall_statistics
          :parser: rst
          :summary:
   * - :py:obj:`_compute_site_statistics <uniharmony.datasets._multisite_data_characterization._compute_site_statistics>`
     - .. autodoc2-docstring:: uniharmony.datasets._multisite_data_characterization._compute_site_statistics
          :parser: rst
          :summary:
   * - :py:obj:`_compute_class_statistics <uniharmony.datasets._multisite_data_characterization._compute_class_statistics>`
     - .. autodoc2-docstring:: uniharmony.datasets._multisite_data_characterization._compute_class_statistics
          :parser: rst
          :summary:
   * - :py:obj:`_compute_correlation_statistics <uniharmony.datasets._multisite_data_characterization._compute_correlation_statistics>`
     - .. autodoc2-docstring:: uniharmony.datasets._multisite_data_characterization._compute_correlation_statistics
          :parser: rst
          :summary:
   * - :py:obj:`_compute_dataset_entropy <uniharmony.datasets._multisite_data_characterization._compute_dataset_entropy>`
     - .. autodoc2-docstring:: uniharmony.datasets._multisite_data_characterization._compute_dataset_entropy
          :parser: rst
          :summary:
   * - :py:obj:`print_statistics_summary <uniharmony.datasets._multisite_data_characterization.print_statistics_summary>`
     - .. autodoc2-docstring:: uniharmony.datasets._multisite_data_characterization.print_statistics_summary
          :parser: rst
          :summary:

Data
~~~~

.. list-table::
   :class: autosummary longtable
   :align: left

   * - :py:obj:`__all__ <uniharmony.datasets._multisite_data_characterization.__all__>`
     - .. autodoc2-docstring:: uniharmony.datasets._multisite_data_characterization.__all__
          :parser: rst
          :summary:
   * - :py:obj:`logger <uniharmony.datasets._multisite_data_characterization.logger>`
     - .. autodoc2-docstring:: uniharmony.datasets._multisite_data_characterization.logger
          :parser: rst
          :summary:

API
~~~

.. py:data:: __all__
   :canonical: uniharmony.datasets._multisite_data_characterization.__all__
   :value: ['get_site_data_statistics', 'print_statistics_summary']

   .. autodoc2-docstring:: uniharmony.datasets._multisite_data_characterization.__all__
      :parser: rst

.. py:data:: logger
   :canonical: uniharmony.datasets._multisite_data_characterization.logger
   :value: 'get_logger(...)'

   .. autodoc2-docstring:: uniharmony.datasets._multisite_data_characterization.logger
      :parser: rst

.. py:function:: get_site_data_statistics(x: numpy.ndarray, y: numpy.ndarray, site_labels: numpy.ndarray, feature_names: list[str] | None = None, compute_comprehensive: bool = True) -> dict[str, typing.Any]
   :canonical: uniharmony.datasets._multisite_data_characterization.get_site_data_statistics

   .. autodoc2-docstring:: uniharmony.datasets._multisite_data_characterization.get_site_data_statistics
      :parser: rst

.. py:function:: _validate_inputs(x: numpy.ndarray, y: numpy.ndarray, site_labels: numpy.ndarray) -> None
   :canonical: uniharmony.datasets._multisite_data_characterization._validate_inputs

   .. autodoc2-docstring:: uniharmony.datasets._multisite_data_characterization._validate_inputs
      :parser: rst

.. py:function:: _validate_array_types(x: numpy.ndarray, y: numpy.ndarray, site_labels: numpy.ndarray) -> None
   :canonical: uniharmony.datasets._multisite_data_characterization._validate_array_types

   .. autodoc2-docstring:: uniharmony.datasets._multisite_data_characterization._validate_array_types
      :parser: rst

.. py:function:: _validate_array_shapes(x: numpy.ndarray, y: numpy.ndarray, site_labels: numpy.ndarray) -> None
   :canonical: uniharmony.datasets._multisite_data_characterization._validate_array_shapes

   .. autodoc2-docstring:: uniharmony.datasets._multisite_data_characterization._validate_array_shapes
      :parser: rst

.. py:function:: _validate_array_dimensions(x: numpy.ndarray, y: numpy.ndarray, site_labels: numpy.ndarray) -> None
   :canonical: uniharmony.datasets._multisite_data_characterization._validate_array_dimensions

   .. autodoc2-docstring:: uniharmony.datasets._multisite_data_characterization._validate_array_dimensions
      :parser: rst

.. py:function:: _validate_array_values(x: numpy.ndarray, y: numpy.ndarray, site_labels: numpy.ndarray) -> None
   :canonical: uniharmony.datasets._multisite_data_characterization._validate_array_values

   .. autodoc2-docstring:: uniharmony.datasets._multisite_data_characterization._validate_array_values
      :parser: rst

.. py:function:: _compute_overall_statistics(x: numpy.ndarray, y: numpy.ndarray, site_labels: numpy.ndarray, feature_names: list[str] | None = None) -> dict[str, typing.Any]
   :canonical: uniharmony.datasets._multisite_data_characterization._compute_overall_statistics

   .. autodoc2-docstring:: uniharmony.datasets._multisite_data_characterization._compute_overall_statistics
      :parser: rst

.. py:function:: _compute_site_statistics(x: numpy.ndarray, y: numpy.ndarray, site_labels: numpy.ndarray, unique_sites: numpy.ndarray, feature_names: list[str] | None = None) -> dict[str, typing.Any]
   :canonical: uniharmony.datasets._multisite_data_characterization._compute_site_statistics

   .. autodoc2-docstring:: uniharmony.datasets._multisite_data_characterization._compute_site_statistics
      :parser: rst

.. py:function:: _compute_class_statistics(x: numpy.ndarray, y: numpy.ndarray, unique_classes: numpy.ndarray, feature_names: list[str] | None = None) -> dict[str, typing.Any]
   :canonical: uniharmony.datasets._multisite_data_characterization._compute_class_statistics

   .. autodoc2-docstring:: uniharmony.datasets._multisite_data_characterization._compute_class_statistics
      :parser: rst

.. py:function:: _compute_correlation_statistics(x: numpy.ndarray, y: numpy.ndarray, site_labels: numpy.ndarray, unique_sites: numpy.ndarray, unique_classes: numpy.ndarray) -> dict[str, typing.Any]
   :canonical: uniharmony.datasets._multisite_data_characterization._compute_correlation_statistics

   .. autodoc2-docstring:: uniharmony.datasets._multisite_data_characterization._compute_correlation_statistics
      :parser: rst

.. py:function:: _compute_dataset_entropy(labels: numpy.ndarray) -> float
   :canonical: uniharmony.datasets._multisite_data_characterization._compute_dataset_entropy

   .. autodoc2-docstring:: uniharmony.datasets._multisite_data_characterization._compute_dataset_entropy
      :parser: rst

.. py:function:: print_statistics_summary(stats: dict[str, typing.Any], max_features: int = 5) -> None
   :canonical: uniharmony.datasets._multisite_data_characterization.print_statistics_summary

   .. autodoc2-docstring:: uniharmony.datasets._multisite_data_characterization.print_statistics_summary
      :parser: rst
