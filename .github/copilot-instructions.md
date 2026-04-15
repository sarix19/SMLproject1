# SML Project 1 - AI Assistant Instructions

## Project Overview

**SML (Statistical Machine Learning) Project 1** is a supervised learning task to predict minimum distance from image data using scikit-learn.

- **Framework**: scikit-learn (regression task)
- **Primary Model**: GradientBoostingRegressor (SVRs are NOT allowed)
- **Pipeline**: Load images → Preprocess (StandardScaler) → Dimensionality reduction (PCA 95% variance) → Hyperparameter tuning (GridSearchCV) → Predict test set distances

## Environment Setup

### Local Development (Recommended)
- **Python Version**: 3.10+
- **Package Manager**: Conda + pip
- **Environment Location**: `/envs/project_1SML`
- **Activation**: `conda activate project_1SML`
- **Setup Commands**:
  ```bash
  pip install --upgrade pip
  pip install -r requirements.txt
  ```

### Key Limitations
- **JupyterHub RAM**: 4GB per user (downsample_factor < 5 with load_rgb=True causes crashes)
- **JupyterHub Time**: 50-hour total limit for entire semester
- **Recommendation**: Develop locally; use JupyterHub only when necessary

## Running Code

### Local
```bash
python sml-2026-project-1/main.py
```

### Expected Output
- Training/validation MAE and R² scores
- Best GridSearchCV parameters
- `prediction.csv` generated in working directory

## Architecture & Key Files

| File | Purpose |
|------|---------|
| `sml-2026-project-1/main.py` | Main ML pipeline (data loading → preprocessing → model tuning → prediction) |
| `sml-2026-project-1/utils.py` | Helper functions (load_config, load_dataset, load_test_dataset, print_results, save_results) |
| `sml-2026-project-1/config.yaml` | Configuration (data_dir, load_rgb, downsample_factor) |
| `sml-2026-project-1/data/` | Train/test images and labels |
| `requirements.txt` | Dependencies (scikit-learn, pandas, numpy, PIL, OpenCV, PyYAML, joblib) |

## Configuration (config.yaml)

```yaml
data_dir: ./data                    # Path to data folder
load_rgb: false                     # Load color images (false = grayscale)
downsample_factor: 5                # Image downsampling (smaller = more detail, higher memory)
```

### Critical Config Notes
- Change `downsample_factor` to ≥5 if memory is an issue
- Set `load_rgb: true` only if necessary (increases feature dimensionality by 3x)
- **Never commit large changes to config.yaml without testing on JupyterHub first**

## Development Workflow

1. **Modify/test code** in `main.py` or `utils.py`
2. **Run locally** first: `python sml-2026-project-1/main.py`
3. **Check outputs**: MAE/R² scores and `prediction.csv`
4. **Test on JupyterHub** if approaching final submission (resource constraints)
5. **Save predictions** via `save_results(test_pred)`

## Common Patterns & Conventions

- **Data handling**: Use `utils.load_config()` and `utils.load_dataset(config)` consistently
- **Preprocessing order**: StandardScaler → PCA (fit only on train data)
- **Model evaluation**: Use `print_results(y_true, y_pred)` for consistent MAE/R² reporting
- **Hyperparameter tuning**: GridSearchCV with cv=3 on training data only
- **Prediction format**: ID (zero-padded 3 digits), Distance (float)

## Constraint Reminder

⚠️ **SVRs are NOT allowed** in this project. Use allowed regressors:
- GradientBoostingRegressor (current)
- RandomForestRegressor
- LinearRegression
- Ridge/Lasso with appropriate regularization
- Neural networks (if sklearn-compatible)

## When Asking for Help

Provide:
1. Current config.yaml settings
2. Error message and stderr output
3. Which step failed (loading? training? evaluation?)
4. Whether testing locally or on JupyterHub
5. Current best MAE/R² scores

---

**Last Updated**: April 2026
**Python**: 3.10+
**Status**: Active Development
