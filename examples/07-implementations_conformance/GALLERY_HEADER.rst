Implementations conformance
---------------------------

UniHarmony re-implements harmonization methods that were first published as
standalone code, often unmaintained or hard to reuse. The examples in this
gallery compare each UniHarmony implementation with the original code:

* where the results are identical, and to what numerical precision;
* where they differ, the mathematical reason, the exact lines of code in both
  implementations, and a simulation with a known ground truth that shows which
  version is correct;
* runtime and memory use.

Every deviation from an original implementation is deliberate, documented in
the estimator's docstring, and covered by a test.
