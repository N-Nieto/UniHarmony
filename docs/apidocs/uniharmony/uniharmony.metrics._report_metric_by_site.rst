:py:mod:`uniharmony.metrics._report_metric_by_site`
===================================================

.. py:module:: uniharmony.metrics._report_metric_by_site

.. autodoc2-docstring:: uniharmony.metrics._report_metric_by_site
   :parser: rst
   :allowtitles:

Module Contents
---------------

Functions
~~~~~~~~~

.. list-table::
   :class: autosummary longtable
   :align: left

   * - :py:obj:`report_metric_by_site <uniharmony.metrics._report_metric_by_site.report_metric_by_site>`
     - .. autodoc2-docstring:: uniharmony.metrics._report_metric_by_site.report_metric_by_site
          :parser: rst
          :summary:
   * - :py:obj:`_input_checks <uniharmony.metrics._report_metric_by_site._input_checks>`
     - .. autodoc2-docstring:: uniharmony.metrics._report_metric_by_site._input_checks
          :parser: rst
          :summary:

Data
~~~~

.. list-table::
   :class: autosummary longtable
   :align: left

   * - :py:obj:`__all__ <uniharmony.metrics._report_metric_by_site.__all__>`
     - .. autodoc2-docstring:: uniharmony.metrics._report_metric_by_site.__all__
          :parser: rst
          :summary:

API
~~~

.. py:data:: __all__
   :canonical: uniharmony.metrics._report_metric_by_site.__all__
   :value: ['report_metric_by_site']

   .. autodoc2-docstring:: uniharmony.metrics._report_metric_by_site.__all__
      :parser: rst

.. py:function:: report_metric_by_site(y_true: numpy.ndarray, y_pred: numpy.ndarray, sites: numpy.ndarray, metric: collections.abc.Callable, overall_performance: bool = False) -> dict[str | int, float]
   :canonical: uniharmony.metrics._report_metric_by_site.report_metric_by_site

   .. autodoc2-docstring:: uniharmony.metrics._report_metric_by_site.report_metric_by_site
      :parser: rst

.. py:function:: _input_checks(y_true: numpy.ndarray, y_pred: numpy.ndarray, sites: numpy.ndarray, metric: collections.abc.Callable, overall_performance: bool) -> None
   :canonical: uniharmony.metrics._report_metric_by_site._input_checks

   .. autodoc2-docstring:: uniharmony.metrics._report_metric_by_site._input_checks
      :parser: rst
