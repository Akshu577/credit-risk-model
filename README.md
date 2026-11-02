# Credit Risk Model

**Live app:** https://akshu-credit-risk.streamlit.app · **Full report:** https://akshu577.github.io/credit-risk-model/

Predicts whether a loan applicant is a bad credit risk, using the public German Credit dataset (1,000 applicants, UCI / OpenML `credit-g`). The model is judged on what errors cost the bank, not only on accuracy.

## Why cost matters

The dataset comes with a cost matrix: approving a borrower who defaults costs **5x** more than turning away a good one. So the project:

1. Compares an interpretable baseline (logistic regression) with gradient boosting.
2. Picks the approval threshold that minimises expected cost, using out-of-fold predictions on the training set only. The test set is never used for tuning.
3. Uses permutation importance to show which applicant features drive each decision.

## Results (25% stratified hold-out)

| Model | CV ROC-AUC | Test ROC-AUC | Test PR-AUC |
|---|---|---|---|
| Logistic Regression | 0.765 ± 0.039 | **0.807** | 0.643 |
| Gradient Boosting | 0.757 ± 0.015 | 0.798 | 0.653 |

The strongest risk drivers are **checking account status**, **credit amount**, **loan purpose**, **credit history** and **loan duration**. These match standard credit-underwriting practice.

**What I learned:** On a dataset this small (1,000 rows), the simple linear model did as well as gradient boosting and is easier to explain to a credit committee. The threshold tuned for cost lowered cost for gradient boosting (155 → 144). For logistic regression it did *not* lower cost on the test set (123 → 146). With only 250 test rows, a tuned threshold carries a lot of variance, and that should be checked before it goes into production.

The full table is in [`reports/metrics.md`](reports/metrics.md) and the importances are in [`reports/feature_importance.csv`](reports/feature_importance.csv).

## Run it

```bash
pip install -r requirements.txt
python src/train.py
```

The dataset downloads from OpenML on the first run.
