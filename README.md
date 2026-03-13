# Machine Learning Take-Home Assignment: Loan Risk Modeling

This repository contains my solution for the Loan Risk Modeling take-home assignment. It focuses on exploring a synthetic dataset, training an XGBoost model, establishing an optimal threshold based on precision-recall trade-offs, and explaining the model using SHAP.

## Directory Structure

- `data/`
  - `loan_applications.csv`: The generated synthetic dataset.
- `notebooks/`
  - `eda.ipynb`: Exploratory Data Analysis, missingness analysis, correlational checks.
  - `model.ipynb`: Feature engineering, model building, and evaluation code.
  - `plot_*.png`: Visualization outputs generated during analysis and evaluation.
- `slides.pdf`: The slide presentation used for the video walkthrough.
- `requirements.txt`: List of dependencies needed to reproduce this environment.

---

## How to Run & Reproduce Results

### 1. Set Up the Environment
You will need a Python environment (Python 3.10+ recommended) to run these scripts. 

Install the required packages from `requirements.txt`:
```bash
pip install -r requirements.txt
```

### 2. Run the Notebooks
You can run the notebooks in the `notebooks/` directory sequentially to see the logical flow:
1. `eda.ipynb`: Examines the data, handles missingness, visualizes risk factors, and examines target imbalance.
2. `model.ipynb`: Performs feature engineering, splits the data, trains the XGBoost model, compares against the rule-based baseline, and generates SHAP explanations.

---

## Approach and Key Decisions

### 1. Data Exploration & Missing Values
- Discovered that missing `documented_monthly_income` strongly correlated with higher default rates. I mapped this to a new binary feature `has_documentation` rather than blindly imputing it with the mean. 
- For the `ongoing` applicants, I excluded them from training to avoid confusing the model with indeterminate ground-truth outcomes, keeping the target robust.

### 2. Feature Engineering
- Created ratio-based features such as `loan_to_income` and `withdrawal_ratio` to give the model normalized financial indicators, which are often stronger predictors of risk than raw dollar amounts.
- Filled missing `documented_monthly_income` values with `stated_monthly_income` based on the assumption that without documentation, stated income is the only anchor point we have, while explicitly tagging that assumption using `has_documentation`. 

### 3. Model Choice: XGBoost
- Selected **XGBoost** because it natively handles non-linear relationships, interacting features, and missing data smoothly.
- Since the dataset was imbalanced (~30% default rate), I relied on the `scale_pos_weight` argument to correctly weigh the minority class constraints.
- Optimized for the max **F1-score** threshold (found at `0.621`) from the Precision-Recall curve to weigh both False Positives and False Negatives contextually, rather than rigidly adhering to a 0.5 threshold. 

### 4. Evaluation 
- Used **SHAP values** to assess global feature importance and provide robust local interpretability for individual loan decisions (crucial in financial compliance). `has_documentation` and `loan_to_income` proved to be major drivers.
- Visualized confusion matrices and ROC curves showing that the ML model heavily outperforms the old rule-based baseline logic on crucial metrics (AUC of 0.72 vs 0.72, but much better F1 and False Negative footprint at the chosen threshold).

---

## What I'd Do With More Time

If given more time to develop this model before production, I would focus on:

1. **Fairness & Bias Analysis:** Evaluate the model's False Positive and False Negative rates sliced across demographic features. Given that there are flags like `employment_status`, we must ensure the model isn't unfairly penalizing self-employed or unemployed groups if it isn't strictly driven by financial capacity indicators.
2. **Handle Interminable Outcomes ('ongoing'):** Instead of dropping the 'ongoing' loans, we could use survival analysis or right-censored modeling to glean early signals from them to predict default probability over time.
3. **Hyperparameter Tuning:** Conduct a systematic grid or random search (e.g., using `Optuna`) over a larger parameter space (focusing on `max_depth`, `min_child_weight`, and `gamma`) to marginally improve ROC-AUC without overfitting.
4. **Data Drift Monitoring:** Implement a pipeline to compare live incoming feature distributions to our training set distributions to combat concept drift over time.
