# Linear Regression Health Costs Calculator

freeCodeCamp Machine Learning with Python project solution for predicting medical insurance costs.

## Approach

- Loads the public `insurance.csv` dataset used by the FCC boilerplate.
- Converts categorical fields (`sex`, `smoker`, `region`) to numeric features.
- Adds simple, interpretable interaction features for smoking, age, BMI, children, and obesity.
- Fits a NumPy closed-form linear/ridge regression model against actual dollar charges.
- Verifies the FCC requirement: final test-set mean absolute error must be below `3500`.

## Files

- `fcc_health_costs.py` — reproducible training and verification script.
- `.gitignore` — excludes downloaded CSV caches and generated model/log artifacts.
- `AUDITAGENT_TRANSFER.md` — concise transfer note for AuditAgent/Bachelorarbeit usage.

## Run locally

From this folder:

```bash
python3 fcc_health_costs.py
```

The script downloads `insurance.csv` on first run and caches it locally. The cached file is ignored by git.

## Local verification

Verified on the local macOS environment with Python 3 and NumPy:

```text
Test MAE: 2491.13
FCC threshold: < 3500
PASSED: final MAE is below the freeCodeCamp requirement.
```

Preprocessing, train/test split, and the closed-form regression solution are deterministic.
