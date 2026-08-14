# Contributing

Thank you for improving Cloud QDM Climate.

## Development setup

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[cloud,test,docs]"
pre-commit install  # optional
```

Before submitting a pull request:

```bash
ruff check .
ruff format --check .
pytest --cov=cloud_qdm
mkdocs build --strict
```

## Scientific changes

Changes to units, calendars, regridding, wet-day handling, QDM parameters, or
physical constraints must include:

- a focused synthetic test;
- an explanation of the scientific assumption;
- a note in `CHANGELOG.md`; and
- updated methodology or technical-manual text.

Never add proprietary or access-controlled data to tests. Use small synthetic
xarray objects generated inside the test suite.

## Pull requests

Keep pull requests focused. Describe the motivation, user impact, validation,
and any change to output compatibility. By contributing, you agree that your
contribution is licensed under Apache-2.0.
