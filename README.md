# Machine Learning Take-Home Assignment: Loan Risk Modeling

This repo contains my solution for the loan risk modeling take-home assignment for Bree Technologies.

## Directory Structure

data/
loan_applications.csv # synthetic dataset generated from the provided script

notebooks/
eda.ipynb data exploration + understanding the dataset
model.ipynb feature engineering, model training, evaluation
plot\_\*.png plots generated during analysis


requirements.txt # python dependencies

---

## How To Run This

Install dependencies:

```bash
pip install -r requirements.txt
```

Then run the notebooks in order:

### 1. EDA Notebook (`notebooks/eda.ipynb`)

Explores the raw dataset, looks at class imbalance, checks missing values, and generates plots showing relationships between features and default risk.

### 2. Model Notebook (`notebooks/model.ipynb`)

Builds features, trains the XGBoost model, compares it against the rule-based baseline, generates evaluation metrics and plots, and uses gain-based feature importance to explain model predictions.

---

## Approach & Key Decisions

### Data Exploration

**Missing Data:** The first thing that stood out was missing `documented_monthly_income`. Instead of just filling those values with a mean, I treated the missingness itself as information. Applicants without documentation default at a much higher rate, so I created a binary feature called `has_documentation`.

**Ongoing Loans:** Loans without outcomes were removed from training. Including them would introduce label noise since we don't know whether those applicants will repay or default.

### Feature Engineering

A lot of the useful signals ended up being ratios rather than raw values:

- `loan_to_income` — how large the loan is relative to monthly income
- `withdrawal_ratio` — monthly withdrawals relative to deposits
- `income_ratio` — stated income relative to documented income

These capture financial stress in a way that raw numbers don't. For missing documented income, I filled it with `stated_monthly_income` since that's the best available proxy, but kept the `has_documentation` flag so the model knows documentation was absent.

### Model Choice

I used XGBoost for a few reasons:

- Handles non-linear relationships between features naturally
- Works well on tabular data
- Deals with missing values without special treatment
- Consistently strong performance on datasets like this

The dataset is imbalanced (~85% repaid, ~15% defaulted), so I used `scale_pos_weight` to give the default class more importance during training. Rather than using a fixed 0.5 threshold, I tuned it to maximise F1 on the default class using the precision-recall curve. That landed at **0.62**.

### Evaluation

The model was evaluated on:

- Precision, Recall, F1 (default class)
- AUC-ROC
- Confusion matrix for both XGBoost and the rule-based baseline
- False positive rate (good applicants wrongly denied)
- False negative rate (defaults that slipped through)

Compared to the rule-based system, XGBoost catches significantly more defaults but also rejects more good applicants. That becomes a business trade-off: fewer financial losses versus approving more customers. The threshold is a dial the business can tune depending on which cost matters more.

For interpretability, gain-based feature importance from the booster shows which features matter globally. For individual decisions, the top contributing features and their actual values are surfaced per applicant — important in lending since denials need to be explainable.

---

## Fairness Analysis

The rule-based system penalises self-employed applicants despite them having nearly identical default rates to employed applicants (25.9% vs 26.5%). That 21-point gap in approval rates has no justification in the outcome data.

XGBoost learns from actual repayment outcomes rather than employment category, so approval rates align more closely to true underlying risk. That said, `employment_status` is correlated with protected characteristics, so its inclusion carries disparate-impact risk even when used correctly. This is flagged as an open tension rather than a resolved one.

---

## What I Would Do With More Time

**Fairness checks** — look more closely at how false positives and false negatives are distributed across employment groups, and test whether removing `employment_status` from the feature set meaningfully changes default rates.

**Handling ongoing loans** — instead of dropping them, treat them as right-censored data and use survival analysis to model default probability over time.

**Hyperparameter tuning** — the current model works well but wasn't heavily tuned. A systematic search with Optuna over `max_depth`, `min_child_weight`, and `gamma` would likely squeeze out more performance.

**Monitoring drift** — in production, the biggest risk is silent degradation as economic conditions change. Feature distribution monitoring comparing live data against the training distribution would catch this early.
