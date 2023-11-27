<h1 align="center"> Epseon Neural Networks </h1>

## Installation

Epseon Neural Networks can be installed with `pip` from PyPI:

```
pip install epseon_nn
```

Alternatively, it is also possible to install it directly from repository:

```
pip install git+https://github.com/UniversityOfGdanskTeamPython/epseon_nn.git
```

## Development

To quickly set up development environment, first you have to install `poetry` globally:

```
pip install poetry
```

Afterwards you will be able to create development virtual environment:

```
poetry shell
```

Then You have to install dependencies into this environment:

```
poetry install --with=docs
```

And pre-commit hooks:

```
poe install-hooks
```

Now you are good to go. Whenever you commit changes, pre-commit hooks will be invoked.
If they fail or change files, you will have to re-add changes and commit again.

## Build from source

To build Epseon Neural Networks from source make sure you have `poetry` environment
activated with:

```
poetry shell
```

With environment active it should be possible to build wheel and source distribution
with:

```
poetry build
```

## Build documentation

To locally build documentation site, first you will need to install all documentation
related dependencies. This can be achieved with following command:

```
poetry install --with docs
```

Afterwards you can invoke `mkdocs` to generate documentation in form of HTML website:

```
mkdocs build
```

**Important** this is not how CI builds documentation, do not use this approach to
upload documentation to GitHub pages.
