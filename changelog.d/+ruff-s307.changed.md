Replace the regex-based `python-no-eval` pre-commit hook with ruff rule `S307`, which flags only the builtin `eval()` and allows method calls such as PyTorch's `model.eval()`
