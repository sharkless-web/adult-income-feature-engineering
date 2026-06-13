# Adult Income Feature Engineering Assignment

This project implements a complete machine learning feature engineering pipeline
for the UCI Adult Income dataset.

## Dataset

- Dataset: Adult / Census Income
- Source: UCI Machine Learning Repository
- Task: Binary classification, predicting whether income is over 50K USD.
- Official page: https://archive.ics.uci.edu/dataset/2/adult

## What Is Included

- Data download and loading
- EDA tables and visualizations
- Missing-value treatment comparison
- Encoding comparison
- Scaling comparison
- Derived feature generation
- Pipeline-based model training
- Feature selection with SelectKBest and Random Forest importance
- Logistic Regression and Random Forest comparison
- Small GridSearchCV experiment
- Korean PDF report

## Run

```powershell
python src/adult_feature_engineering_pipeline.py --project-root .
```

Generated files are saved under:

- `data/raw/`
- `outputs/figures/`
- `outputs/tables/`
- `outputs/adult_income_feature_engineering_report.pdf`

## Main Source File

- `src/adult_feature_engineering_pipeline.py`
