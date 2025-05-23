<p align="center">
    <img src="docs/assets/logo-diagram.png" width="600px" alt='mxlpy-logo'>
</p>

# mxlpy

[![pypi](https://img.shields.io/pypi/v/mxlpy.svg)](https://pypi.python.org/pypi/mxlpy)
[![docs][docs-badge]][docs]
![License](https://img.shields.io/badge/license-GPL--3.0-blue?style=flat-square)
![Coverage](https://img.shields.io/badge/dynamic/json?url=https%3A%2F%2Fgist.github.com%2Fmarvinvanaalst%2F98ab3ce1db511de42f9871e91d85e4cd%2Fraw%2Fcoverage.json&query=%24.message&label=Coverage&color=%24.color&suffix=%20%25)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![security: bandit](https://img.shields.io/badge/security-bandit-yellow.svg)](https://github.com/PyCQA/bandit)
[![PyPI Downloads](https://static.pepy.tech/badge/mxlpy)](https://pepy.tech/projects/mxlpy)

[docs-badge]: https://img.shields.io/badge/docs-main-green.svg?style=flat-square
[docs]: https://computational-biology-aachen.github.io/mxlpy/

## Installation

You can install mxlpy using pip: `pip install mxlpy`

If you want access to the sundials solver suite via the [assimulo](https://jmodelica.org/assimulo/) package, we recommend setting up a virtual environment via [pixi](https://pixi.sh/) or [mamba / conda](https://mamba.readthedocs.io/en/latest/) using the [conda-forge](https://conda-forge.org/) channel.

```bash
pixi init
pixi add python assimulo
pixi add --pypi mxlpy[torch]
```

## How to cite

If you use this software in your scientific work, please cite [this article](...):

- [doi](https://doi.org/)
- [bibtex file](https://fillme.out)


## Development setup

You have two choices here, using `uv` (pypi-only) or using `pixi` (conda-forge, including assimulo)

### uv

- Install `uv` as described in [the docs](https://docs.astral.sh/uv/getting-started/installation/).
- Run `uv sync --extra dev --extra torch` to install dependencies locally

### pixi

- Install `pixi` as described in [the docs](https://pixi.sh/latest/#installation)
- Run `pixi install --frozen`


## Notes

- `uv add $package`
- `uv add --optional dev $package`
