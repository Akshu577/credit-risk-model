# Credit Risk Model - Results

Dataset: German Credit (1,000 applicants, 30% labelled bad risk). Test set: 25% stratified hold-out.

| Model | CV ROC-AUC (train) | Test ROC-AUC | Test PR-AUC | Threshold | Cost @0.5 | Cost @tuned |
|---|---|---|---|---|---|---|
| Logistic Regression | 0.765 ± 0.039 | 0.807 | 0.643 | 0.20 | 123 | 146 |
| Gradient Boosting | 0.757 ± 0.015 | 0.798 | 0.653 | 0.22 | 155 | 144 |

Best model by test ROC-AUC: **Logistic Regression**

Cost = 5 x (bad borrowers approved) + 1 x (good borrowers rejected).

## Top 8 drivers (permutation importance, ROC-AUC drop)

- checking_status: 0.084
- credit_amount: 0.030
- purpose: 0.023
- credit_history: 0.018
- duration: 0.016
- installment_commitment: 0.011
- savings_status: 0.008
- other_parties: 0.007
