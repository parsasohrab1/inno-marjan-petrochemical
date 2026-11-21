# GitHub Actions workflow

The CI pipeline file `ci.yml` was not pushed because the Personal Access Token
used for the initial push lacked the `workflow` scope (GitHub refuses to create
or update `.github/workflows/*` via a token without that scope).

To add it, run locally with a token that has the `workflow` scope, or use the
GitHub CLI:

```bash
gh workflow add ci.yml   # if you have the file
# or simply:
git add .github/workflows/ci.yml
git commit -m "Add CI workflow"
git push origin main
```

The workflow runs the full test suite with coverage:
- Sets up Python 3.11
- Installs backend/requirements.txt
- Runs `pytest` with `--cov=mrto --cov-report=xml`
- Uploads coverage to Codecov
