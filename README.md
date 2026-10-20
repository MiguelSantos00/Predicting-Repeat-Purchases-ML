# Predicting Repeat Purchases - Machine Learning

Group 06 homework: predict whether a customer purchases again within 90 days of
the current purchase. One observation represents a purchase occasion.

## Run the project

Open this folder in VS Code, select a Python kernel with the dependencies in
`environment.yml`, and run these notebooks in order. Start each with a fresh
kernel and its working directory set to `notebooks/`.

1. [Business Understanding](notebooks/01_Business_Understanding.ipynb)
2. [Data Understanding](notebooks/02_Data_Understanding.ipynb)
3. [Data Preparation](notebooks/03_Data_Preparation.ipynb)
4. [Modeling](notebooks/04_Modeling.ipynb)
5. [Evaluation](notebooks/05_Evaluation.ipynb)
6. [Deployment](notebooks/06_Deployment.ipynb)

The saved notebooks include executed results. Three small functions in
`notebooks/preparation.py` reuse the learned preparation steps for inner folds
and final fitting; the operations are shown individually in Data Preparation.
Modeling presents each classifier separately and uses one evaluation function
to repeat the same three folds. Evaluation loads the fitted forest and selected
column CSV directly. Deployment explicitly rebuilds the selected forest + L1.

Git keeps the source notebooks, raw data, reports, result tables and submission.
Clean/prepared data, fitted model files and the delivery ZIP are generated locally
and excluded from version control. Running notebooks 03 to 06 recreates them.

The notebooks were executed with the existing Anaconda Python 3.11.5 installation,
pandas 2.0.3, NumPy 1.24.3 and scikit-learn 1.3.0. The environment file records the
core versions used. A fresh Conda installation of that file has not been tested.
To create a separate environment:

```text
conda env create -f environment.yml
conda activate machine_learning
```

## Homework result

The official Kaggle metric is **binary F1**, for `repeat_purchase_90d = 1`.
Three expanding temporal folds inside development compare logistic regression,
a decision tree and random forest, with and without L1 feature selection.
Every fold learns its own imputation, scaling, encoding and selection.

Random forest with L1 is selected by mean inner F1 (0.7384, against baseline
0.7044). The L1 choice reduces development inputs from 276 to 54. Its advantage
over forest without selection (0.7368) is small and not conclusive.

On the later 1,298-row holdout, the fixed solution achieves:

| Solution | F1 | Precision | Recall | Accuracy |
|---|---:|---:|---:|---:|
| Majority baseline | 0.7525 | 0.6032 | 1.0000 | 0.6032 |
| Random forest + L1 | 0.7778 | 0.7869 | 0.7688 | 0.7350 |

There are 163 false positives and 181 false negatives. Performance is weaker
for customers unseen in development (101 of their 145 purchases already have
history); none of the 12 positive cases among 44 first purchases are
identified. These limitations are discussed in Evaluation.

The earlier group notebook explored the full training file, so the local holdout
is not completely untouched. No Kaggle score has been observed or reported.

## Files and reproducibility

- `data/raw/`: the four original files, unchanged.
- `data/clean/`: two cleaned, unscaled tables for fold preparation and full refitting.
- `data/prepared/`: development, holdout and test matrices for the original fixed split.
- `notebooks/`: the six phases and the three preparation helper functions.
- `results/`: fold scores, selected features, fitted development model and holdout results.
- `submissions/ML_Group_06_Version01.csv`: 986 original test IDs and binary predictions.
- `reports/`: the two-page homework PDF and its editable Markdown text.
- `assigment instructions/`: the professor's original briefs.

Local validation starts on 17 June 2023: 5,024 development purchases, a 147-row
90-day gap and 1,298 holdout purchases. Ten exact duplicates and 55 distinct
undated/incomplete rows are excluded. IDs and target are excluded from inputs.

Cleaning reconstructs historical averages from total/count and treats impossible
maxima and annual spend as missing. This assumes totals and counts are reliable.
Coherent high values are retained; monetary inputs receive log1p before standard
scaling. Missing-value flags are kept and categories occurring fewer than 20
times in the current training split are pooled.

Deployment refits the same selected method on all 6,469 eligible labelled rows,
including the former gap. The first test date is 98 days after the last training
purchase, so the training targets are mature. Final preparation yields 302 inputs,
of which L1 selects 58. Test contributes no fitted parameters. Fixed random seeds
make the versioned CSV reproducible; it contains only `ID` and `repeat_purchase_90d`.

## Delivery status

This is the homework version, due **26 October 2026 at 17:59**. The final project,
due **22 December 2026 at 17:59**, still requires at least five taught algorithms,
further optimisation and an additional investigation.

The local CSV and two-page report are prepared. Kaggle upload, final submission
selection and course delivery have not been performed. Use team `ML_Group_06`
and the versioned filename. The public leaderboard uses 70% of test rows; it
should not replace the local assessment protocol.

## Credits

The original repository credits Miguel Santos, Miguel Nunes and Ricardo Lourenço.
The group will confirm the full member list and contribution percentages for the
final-project version. Original group notebooks remain in Git history. Work is
on `codex/crisp-dm-preparation`, based on `homework`; `feature/pipeline` was not
used as the implementation base.
