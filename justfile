default:
    just --list

install-prek:
    uv tool install prek

install-hooks:
    prek install --install-hooks

run-hooks:
    prek run --all-files

install-dev:
    uv pip install -e ".[dev,docs]"

format-lint:
    uv run -- ruff format . && ruff check --fix .

convert-notebooks:
    uv run -- jupyter nbconvert --to markdown --output-dir=docs/examples/ examples/*

serve-docs: convert-notebooks
    uv run -- zensical serve

lint:
    uv run -- tox -e ruff

coverage:
    uv run -- tox -e coverage

changelog:
    uv run -- tox -e changelog

add-news news id type:
    uv run -- towncrier create -c "{{news}}" {{id}}.{{type}}.md
