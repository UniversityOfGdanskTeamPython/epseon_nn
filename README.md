<h1 align="center"> PEC Framework Data Consumer </h1>

## Installation

PEC Framework Data Consumer can be installed using the HTTPS address of the
[repository](https://github.com/UniversityOfGdanskTeamPython/pec_framework_data_consumer.git):

```
git clone https://github.com/UniversityOfGdanskTeamPython/pec_framework_data_consumer.git
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

To build PEC Framework Data Consumer from source make sure you have `poetry` environment
activated with:

```
poetry shell
```

With environment active it should be possible to build wheel and source distribution
with:

```
poetry build
```
