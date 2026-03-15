# Credit Risk Scorecard Development

## Overview
End-to-end development of a **logistic regression credit scorecard** using the
Home Credit Default Risk dataset (307,511 applicants). Covers the complete
model lifecycle from raw data to a deployable integer scorecard — following
industry standards used by retail banks globally.

---

## Final Model Results

| Metric | Train | OOT | Benchmark | Status |
|--------|-------|-----|-----------|--------|
| **Gini** | 44.8% | 45.4% | > 40% |  Pass |
| **Gini Drop** | — | -0.6% | < 5% | Pass |
| **Score Range** | — | 481–640 | — | — |
| **Mean Score** | — | 566 | — | — |
| **Variables in Scorecard** | — | 11 | 8–15 | Pass |
| **Scaling** | PDO=20, Base=600, Odds=50:1 | Industry standard | — |  |

> **Note on OOT Gini > Train Gini:** A small positive difference (-0.6%) indicates
> the model generalises well and is not overfitted to training data.

---

##  Notebook Structure

| Notebook | Description | Key Output |
|----------|-------------|------------|
| [01_EDA](./notebooks/01_EDA.ipynb) | Exploratory Data Analysis | 9 charts, data quality findings |
| [02_preprocessing](./notebooks/02_preprocessing.ipynb) | Data Cleaning & Feature Engineering | train_clean.csv, oot_clean.csv |
| [woe_iv_binning](./notebooks/03_woe_iv_binning.ipynb) | WOE/IV Analysis & Variable Selection | IV rankings, WOE bin plots |
| [04_logistic_regression_scorecard](./notebooks/04_logistic_regression_scorecard.ipynb) | Model Development | Gini, KS, AUROC, decile table |
| [05_model_validation](./notebooks/05_model_validation.ipynb) | Formal Model Validation | PSI, CSI, HL test, validation report |
| [06_scorecard_scaling](./notebooks/06_scorecard_scaling.ipynb) | PDO Scaling & Score Bands | Final scorecard points table |

---

## Dataset

**Source:** [Home Credit Default Risk — Kaggle](https://www.kaggle.com/competitions/home-credit-default-risk/data)

| Detail | Value |
|--------|-------|
| Total Applicants | 307,511 |
| Features (Raw) | 122 |
| Target Variable | `TARGET` (1 = Default, 0 = Non-Default) |
| Overall Default Rate | ~8.1% |
| Train Split | 246,009 (80%) |
| OOT Split | 61,502 (20%) |

> **Note on Test Set:** `application_test.csv` (Kaggle submission file) was not used
> as it contains no target labels. Instead, a stratified 80/20 OOT split was created
> from labelled training data — this is the correct approach for professional
> model validation and allows rigorous Gini, KS and PSI measurement.

---

##  Methodology

### 1. Exploratory Data Analysis
- Target distribution and class imbalance analysis (8.1% default rate, 1:11 ratio)
- Missing value severity classification — high (>80%), medium (30-80%), low (<30%)
- Default rate heatmap by education level × income type
- Age band risk profiling — younger applicants (<30) show meaningfully higher default rates
- Outlier detection using IQR method on key financial variables

### 2. Data Preprocessing
- Dropped 41 columns exceeding 40% missing threshold
- Fixed `DAYS_EMPLOYED` anomaly — code 365243 representing unemployed replaced with NaN
- Fixed `CODE_GENDER` XNA values replaced with NaN
- Created 8 derived features including `CREDIT_INCOME_RATIO`, `ANNUITY_INCOME_RATIO`, `EMPLOYMENT_AGE_RATIO`
- Winsorization at 1st/99th percentile for financial variables
- Stratified 80/20 train/OOT split preserving target rate

### 3. WOE / IV Binning
- Information Value calculated for all variables using `scorecardpy`
- Variable selection threshold: IV between 0.10 and 0.50
- Monotonicity check on WOE trends — regulatory requirement
- PSI check on each variable — stability between train and OOT
- Final shortlist: **11 variables** passing all three criteria

### 4. Logistic Regression
- Built using `statsmodels` — provides p-values for regulatory documentation
- VIF analysis to detect and remove multicollinear variables (threshold VIF < 10)
- All coefficients verified positive — correct direction in WOE-encoded model
- Decile analysis: top 3 deciles capture majority of defaults

### 5. Model Validation (SR 11-7 Aligned)
| Test | Result | Status |
|------|--------|--------|
| Gini OOT | 45.4% |  Pass |
| Gini Drop | -0.6% |  Pass |
| Bootstrap Gini 95% CI | Computed |  |
| PSI (Score) | Calculated |  Stable |
| CSI (Per Variable) | All variables |  Stable |
| Hosmer-Lemeshow Calibration | p-value computed |  |
| Rank Ordering (20 ventiles) | Violations checked |  Pass |

### 6. Scorecard Scaling
- PDO method: Points to Double the Odds
- Parameters: PDO = 20, Base Score = 600, Base Odds = 50:1
- Score formula: `Score = Offset + Factor × (−intercept − Σ coef × WOE)`
- Gini preserved after integer rounding: Yes (difference < 0.001)
- Score bands defined with business recommendations:

| Band | Score Range | Default Rate | Decision |
|------|-------------|--------------|----------|
| 1 | < 500 | Very High | Decline |
| 2 | 500–540 | High | Decline |
| 3 | 540–580 | Medium High | Manual Review |
| 4 | 580–620 | Medium | Conditional Approve |
| 5 | 620–660 | Medium Low | Conditional Approve |
| 6 | 660–700 | Low | Auto Approve |
| 7 | > 700 | Very Low | Auto Approve |

---

## Technical Stack

```
Python 3.10       pandas, numpy, matplotlib, seaborn
scorecardpy       WOE/IV binning, PSI calculation
optbinning        Optimal monotonic binning
statsmodels       Logistic regression with p-values
scikit-learn      train_test_split, roc_auc_score, roc_curve
scipy             Hosmer-Lemeshow chi-square test
```

---

##  Key Charts Generated

| Chart | Description |
|-------|-------------|
| Target distribution | Class imbalance analysis |
| Missing value severity | Column-level missingness |
| Default rate heatmap | Education × Income type |
| IV ranking | Top 25 variables by predictive power |
| WOE bin plots | Risk profile per bin for top 6 variables |
| ROC curve | Train vs OOT discrimination |
| KS chart | Maximum separation point |
| Calibration plot | Predicted vs actual default rate |
| Decile analysis | Top decile lift |
| PSI chart | Score distribution stability |
| CSI chart | Variable-level stability |
| Score band chart | Risk and volume by band |

---

##  Regulatory Alignment

This project follows industry model risk management standards:

- **SR 11-7** (Federal Reserve) — Model validation framework
- **SS1/23** (PRA UK) — Model risk management principles
- **Basel IRB** — Internal ratings-based approach for PD estimation
- **IFRS 9** — Calibration testing relevant for ECL PD models

---

>  Dataset sourced from Kaggle (public). Code written to production standards
> with full inline documentation. All validation tests documented with pass/fail verdicts.
