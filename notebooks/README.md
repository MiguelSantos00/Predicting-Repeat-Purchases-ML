# Notebooks - CRISP-DM

Run the six notebooks in order from this folder, top to bottom in fresh kernels.

| Phase | Notebook | Main work |
|---|---|---|
| Business Understanding | `01_Business_Understanding.ipynb` | Task, objectives and protocol. |
| Data Understanding | `02_Data_Understanding.ipynb` | Quality, dates, missingness and relationships. |
| Data Preparation | `03_Data_Preparation.ipynb` | Cleaning, split, missing values, log, scale and encoding. |
| Modeling | `04_Modeling.ipynb` | Temporal folds, baseline, three classifiers and L1 selection. |
| Evaluation | `05_Evaluation.ipynb` | Fixed later holdout, F1, error counts and groups. |
| Deployment | `06_Deployment.ipynb` | Full eligible refit and versioned Kaggle CSV. |

`preparation.py` has three short functions for repeated fitting of the learned
steps. They reproduce the preparation shown explicitly in notebook 03.

The inner folds start from `data/clean`, not globally prepared matrices, so each
fold learns its own parameters. Notebook 05 uses the fixed holdout matrices.
Notebook 06 refits preparation and selection on all 6,469 eligible labelled rows.

Main metric: binary F1. Random forest with L1 wins the inner comparison; the
later holdout F1, with a 0.40 cutoff chosen on the folds, is 0.7915 versus baseline 0.7525. First-purchase prediction is
a weakness, discussed in Evaluation. No model changes follow that holdout analysis.

The CSV is generated in `submissions/`, with all 986 IDs and no extra index.
Uploading to Kaggle and course delivery remain separate actions. The final
project extends this homework with five taught algorithms, optimisation and
an additional investigation.
