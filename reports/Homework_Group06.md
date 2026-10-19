# Predicting Repeat Purchases
Machine Learning 2026/27 - Homework - Group 06 - 19 October 2026

## 1. Goal and workflow
We predict whether a customer buys again within 90 days of a purchase occasion.
The observation is a purchase, not a unique customer. The target is
repeat_purchase_90d; class 1 means a subsequent purchase within that window.
Binary F1 is the official Kaggle metric. The six CRISP-DM notebooks cover the
business question, exploration, preparation, modeling, evaluation and batch prediction.

Workflow: understand the task -> inspect quality and chronology -> clean and split
-> learn missing-value treatment, log, scale and encoding -> L1 selection and
classifier comparison -> assess the later holdout -> refit and export predictions.

## 2. Data and preparation decisions
The supplied files contain 6,534 training purchases, 986 test purchases and
1,143 customer records. A left join through customer_id covers every purchase.
We remove 10 exact duplicates excluding ID and 55 distinct undated records with
no purchase information. This leaves 6,469 dated training purchases. All original
files and all test IDs are preserved.

ID and customer_id are tracking fields, not model inputs. Empty customer_type,
company name, address and selected granular geography are excluded. Broader
geography is treated as categorical: the pseudonymous codes do not measure distance.

Invalid negative history values and suspected exact 1e9 placeholders become missing values.
Historical averages are reconstructed as total/count, assuming those supplied
totals and counts are reliable; a zero count leaves the average undefined.
Impossible historical maxima and annual spend are marked missing. Coherent high
amounts are retained. Remaining numeric gaps receive training medians and flags;
text gaps become Missing. Monetary inputs use log1p followed by standard scaling.
One-hot encoding pools categories with fewer than 20 training purchases.
Year, month and weekday are candidate calendar inputs; dates are anonymised.

## 3. Assessment protocol
The latest approximately 20% of dated purchases form a chronological holdout:
1,298 purchases from 17 June 2023. Development contains 5,024 purchases ending
18 March 2023. The intervening 147 purchases form a 90-day label-maturity gap.
Purchases on the same day remain together. Repeated customers are allowed across
periods because future purchases by existing customers are part of the task.

Inside development, three expanding temporal folds each have their own 90-day gap.
Medians, scaling, category frequencies and supervised selection are learned only
from each training fold. The prepared full-development CSV is not used for inner CV.
The classifier and feature strategy are selected by mean inner F1 before assessing
the later holdout. No threshold is adjusted after viewing that holdout.

## 4. Feature selection and classifiers
L1 logistic regression with C=0.1 selects nonzero coefficients (tolerance 1e-6),
separately within each fold. We compare all inputs with the selected subset for
logistic regression (C=1), a depth-5 tree (minimum leaf 20), and a 200-tree random
forest (minimum leaf 5). The majority-class baseline uses the same folds.
Random seeds are fixed. This homework uses a compact comparison, not a broad
hyperparameter search.

The forest with L1 has mean F1 0.7384 (fold SD 0.0096); baseline F1 is 0.7044.
Forest without selection scores 0.7368. The difference of 0.0016 is small and not
conclusive. L1 reduces full-development inputs from 276 to 54. random_noise passes
the fixed rule and remains included; this is not proof of stable predictive value.
Its targeted ablation is a useful final-project extension.

## 5. Fixed holdout result
The selected forest achieves F1 0.7778, precision 0.7869,
recall 0.7688 and accuracy 0.7350. Baseline F1 is
0.7525. The absolute F1 gain is 2.53 percentage points.
Confusion counts are TN=352, FP=163, FN=181, TP=602.

F1 is 0.7910 for customers seen in development and 0.6230 for those unseen.
Of the latter group's 145 purchases, 101 already have supplied purchase history.
Unseen in development therefore does not mean first purchase. The 44 first
purchases contain 12 positives but no true positives from the classifier (F1=0).
This is a concrete cold-start weakness, with a small subgroup sample. No model
changes follow this analysis. The business brief gives no costs or margins, so
we do not translate these results into profit.

## 6. Refit, reproducibility and limits
Deployment refits the chosen method on all 6,469 eligible labelled purchases,
including the former gap. Test starts 98 days after the final training purchase,
so all retained training targets are mature. Preparation is relearned from training
only, producing 302 inputs; L1 selects 58 before fitting the same forest settings.

ML_Group_06_Version01.csv contains exactly 986 original IDs and binary predictions:
645 positives and 341 negatives. It has only ID and repeat_purchase_90d, without
an extra index. These are prediction counts, not test-performance estimates.
Kaggle upload and the public score are pending. The six notebooks reproduce the CSV.

The earlier group exploration used the full training file, so the holdout is not
completely untouched. There is one outer period; customers repeat across rows;
cleaning assumes reliable totals/counts; customer attributes lack historical
snapshots. Calendar features and rare-category choices need further study.
The final project will extend the comparison to at least five taught algorithms,
optimisation and an additional investigation.

Sources: course homework/project briefs, feature-selection and model-assessment
materials, supplied CSVs, and Kaggle Overview/Data/Rules screenshots provided on
19 October 2026. No external research or public leaderboard result is used here.
