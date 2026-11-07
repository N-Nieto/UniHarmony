

.. _sphx_glr_auto_examples_08-pipelines:

Pipelines
---------

Harmonizers need the site (and often covariates) of every sample at ``fit``
*and* at ``transform``. These examples show how to keep this metadata as
columns of a DataFrame so that UniHarmony works with every scikit-learn tool
(pipelines, cross-validation, grid search, scorers), with imbalanced-learn
samplers followed by a harmonizer, and as a julearn pipeline step. See
:doc:`/pipelines` for the reference.


.. raw:: html

  <div id='sg-tag-list' class='sphx-glr-tag-list'></div>


.. raw:: html

    <div class="sphx-glr-thumbnails">

.. thumbnail-parent-div-open

.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="julearn builds scikit-learn pipelines from typed columns: every column of the data gets a type (``continuous``, confound, site, ...) and every step is applied to some types. A UniHarmony harmonizer is a julearn step: it receives the feature, site and covariate columns and returns the harmonized features.">

.. only:: html

  .. image:: /auto_examples/08-pipelines/images/thumb/sphx_glr_03-plot_julearn_harmonization_thumb.png
    :alt:

  :doc:`/auto_examples/08-pipelines/03-plot_julearn_harmonization`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Harmonization as a julearn pipeline step</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="When the class balance differs across sites, ComBat-type harmonizers face a dilemma: without the class as a covariate they remove part of the class signal with the site effect; with it they need the class of the test subjects (leakage). Balancing the classes within every site first, with IntraSiteInterpolation (ISI), removes the dependence between site and class, so the harmonizer can be fitted without the class.">

.. only:: html

  .. image:: /auto_examples/08-pipelines/images/thumb/sphx_glr_02-plot_imblearn_sampler_then_harmonizer_thumb.png
    :alt:

  :doc:`/auto_examples/08-pipelines/02-plot_imblearn_sampler_then_harmonizer`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Sampler followed by a harmonizer (imbalanced-learn)</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="A harmonizer must know the site of every sample, both when it is fitted on the training data and when it transforms the test data: the site effect of a test subject can only be removed if its site is known. Harmonizers that keep biological effects also need the covariates (e.g., age) of every sample.">

.. only:: html

  .. image:: /auto_examples/08-pipelines/images/thumb/sphx_glr_01-plot_sklearn_pipeline_thumb.png
    :alt:

  :doc:`/auto_examples/08-pipelines/01-plot_sklearn_pipeline`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Harmonization in scikit-learn pipelines</div>
    </div>


.. thumbnail-parent-div-close

.. raw:: html

    </div>


.. toctree::
   :hidden:

   /auto_examples/08-pipelines/03-plot_julearn_harmonization
   /auto_examples/08-pipelines/02-plot_imblearn_sampler_then_harmonizer
   /auto_examples/08-pipelines/01-plot_sklearn_pipeline

