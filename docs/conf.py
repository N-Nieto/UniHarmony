"""Provide configuration for Sphinx."""

import datetime
from functools import partial
from pathlib import Path

from myst_sphinx_gallery import GalleryConfig
from setuptools_scm import get_version


# Path setup

PROJECT_ROOT_DIR = Path(__file__).parents[1].resolve()
get_scm_version = partial(get_version, root=PROJECT_ROOT_DIR)

# Project information

github_url = "https://github.com"
github_repo_org = "N-Nieto"
github_repo_name = "uniharmony"
github_repo_slug = f"{github_repo_org}/{github_repo_name}"
github_repo_url = f"{github_url}/{github_repo_slug}"

project = github_repo_name
author = f"{project} Contributors"
copyright = f"{datetime.date.today().year}, {author}"

# The version along with dev tag
release = get_scm_version(
    version_scheme="guess-next-dev",
    local_scheme="no-local-version",
)

# General configuration

extensions = [
    # Built-in extensions:
    "sphinx.ext.doctest",  # test snippets in the documentation
    "sphinx.ext.extlinks",  # markup to shorten external links
    "sphinx.ext.intersphinx",  # link to other projects` documentation
    "sphinx.ext.napoleon",  # parse numpy-style docstrings
    "sphinx.ext.githubpages",  # publish to github-pages
    # Third-party extensions:
    "sphinx_copybutton",  # copy button for code blocks
    "myst_nb",  # md + ipynb to rst parser
    "autodoc2",  # include documentation from docstrings
    "myst_sphinx_gallery",  # HTML gallery of examples
]

source_suffix = {
    ".rst": "restructuredtext",
    ".md": "myst-nb",
    ".myst": "myst-nb",
}

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

# Options for HTML output

html_theme = "sphinx_book_theme"
html_theme_options = {
    "repository_url": github_repo_url,
    "use_repository_button": True,
}
html_static_path = ["_static"]

# sphinx.ext.intersphinx configuration

intersphinx_mapping = {
    "python": ("https://docs.python.org/3/", None),
    "sklearn": ("https://scikit-learn.org/stable", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "pandas": ("https://pandas.pydata.org/pandas-docs/dev", None),
}

# sphinx.ext.extlinks configuration

extlinks = {
    "gh": (f"{github_repo_url}/issues/%s", "#%s"),
}

# myst_parser (myst_nb) configuration

myst_enable_extensions = [
    "dollarmath",
    "linkify",
]

# autodoc2 configuration

autodoc2_packages = ["../src/uniharmony"]
autodoc2_docstring_parser_regexes = [
    (".*", "rst"),
]

# myst_sphinx_gallery configuration

myst_sphinx_gallery_config = GalleryConfig(
    examples_dirs="../examples",
    gallery_dirs="auto_examples",
    root_dir=Path(__file__).parent,
    notebook_thumbnail_strategy="code",
)
