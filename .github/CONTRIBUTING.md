# How To Contribute

## Setting up the local development environment

#. Fork the https://github.com/N-Nieto/UniHarmony repository on GitHub. If you
   have never done this before,
   [follow the official guide](https://guides.github.com/activities/forking/).

#. Clone your fork locally as described in the same guide.

#. Install [uv](https://docs.astral.sh/uv/getting-started/installation/) and [just](https://just.systems/man/en/packages.html).
   Then, in the project root, run:

   ```console
   $ just install-dev
   $ just install-prek
   $ just install-hooks
   ```

#. Create a branch for local development using the `main` branch as a
   starting point.

   ```console
   $ git checkout main
   $ git checkout -b <prefix>/<name-of-your-branch>
   ```

   `<prefix>` can be any of the following:

   - `feat` : feature addition
   - `update` : code update
   - `fix` : bug fix
   - `refactor` : code restructure
   - `chore` : housekeeping changes not affecting functionality

   Now you can make your changes locally.

## Best practices for making changes

#. When making changes locally, it is helpful to `git commit` your work
   regularly. On one hand to save your work and on the other hand, the smaller
   the steps, the easier it is to review your work later. Please use
   [semantic commit messages](http://karma-runner.github.io/2.0/dev/git-commit-msg.html).

   ```console
   $ git add .
   $ git commit -m "<prefix>: <summary of changes>"
   ```

   In case, you want to commit some WIP (work-in-progress) code, please indicate
   that in the commit message and use the flag `--no-verify` with
   `git commit` like so:

   ```console
   $ git commit --no-verify -m "WIP: <summary of changes>"
   ```

#. When you're done making changes, check that your changes pass our linting by running:

   ```console
   $ just lint
   ```

#. If you are updating the documentation, you can run:

   ```console
   $ just serve-docs
   ```

   to see the changes before pushing them.

#. Check that the tests pass and coverage is good enough by running:

   ```console
   $ just coverage
   ```

## Submitting your changes

#. Push your branch to GitHub.

   ```console
   $ git push origin <prefix>/<name-of-your-branch>
   ```

#. Open the link displayed in the message when pushing your new branch in order
   to submit a pull request. Please follow the template presented to you in the
   web interface to complete your pull request.

## Adding examples

#. If you are adding examples, add Jupyter notebooks to the `examples/` directory in the project root.

#. To check your changes, you can either run:

   ```console
   $ just serve-docs
   ```

   to start the docs server or run:

   ```console
   $ just convert-notebooks
   ```

   to convert the `.ipynb` files to respective `.md` files under `docs/examples`. "Serving" the docs
   automatically converts the notebooks, but if you only want to check the generated `.md` files, running
   the latter command will be easier.

#. To make the docs generator aware of the `.md` example file, add it under `[project] > nav > "Examples"`
   in `zensical.toml`.

## GitHub Pull Request guidelines

Before you submit a pull request, check that it meets these guidelines:

#. If the pull request adds functionality, the documentation should be
   updated accordingly.

#. Note the pull request ID assigned after completing the previous step and
   create a *newsfragment* by running:

   ```console
   $ just add-news '<change>' <pr-id> <type>
   ```

   `<type>` can be any of the following:

   - `security` : code security related changes
   - `removed` : code or feature removal
   - `deprecated` : feature deprecation
   - `added` : code or feature addition
   - `changed` : code or feature changes
   - `fixed` : bug fix

#. Someone from the core team will review your work and guide you to a successful
   contribution.
