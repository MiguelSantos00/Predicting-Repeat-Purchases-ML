# Notebooks

Each step of the project has its own notebook, run in this order:

| step | notebook | status |
|---|---|---|
| 1. Exploration | `Homework_Group06_Exploration.ipynb` | in progress |
| 2. Clean and pre-process | `Homework_Group06_Clean_Pre-Process.ipynb` | done (date features left for later) |
| 3. Feature selection | `Homework_Group06_Feature_Selection.ipynb` | done |
| 4. Modelling and submission | `Homework_Group06_Modelling.ipynb` | done |

---

# 1. Exploration

`Homework_Group06_Exploration.ipynb`: look at the data and decide what has to be fixed. Nothing here changes the data for later steps, except dropping exact duplicates.

## To do
- continue exploration: multivariate relationships

## Dataset observations

### Loading
- there is a column named ID so we dont want it to be a feature, so on read_csv we use index_col = "ID"
- `clientes.csv` is joined on `customer_id` with `raw_train.join(raw_clientes, on="customer_id")`, which keeps the ID index (`pd.merge` resets it)
- checking for duplicates with df.duplicated().sum() we get 10 exact duplicates, dropped right after loading

### Missing values
- since customer type is null for all rows its safe to drop
- there are a lot of features with exactly 65 missing values, all in the same rows. (raw_data.isna().sum() == 65).sum() gives us 24, so 24 features are empty in those 65 rows. since these give us no info we believe they are safe to drop
- total number of rows is 6534, 20% of that is 1300, there are features with more than 1300 total values missing, those could be dropped
- street has around 60% missing values so probably safe to drop

### Invalid values and outliers
- some columns contain negative values that can't be real:
    - `days_since_previous_purchase` and `prior_mean_gap_days`: exactly -4 in 16 rows, looks like a code, not a real value
    - `prior_mean_purchase_value` (16 rows) and `spend_365d` (15 rows): different negative values
- `prior_mean_purchase_value` and `spend_365d` also have a maximum of exactly 1,000,000,000, which also looks like a code
- lots of very big outliers

### Types
- some variables are floats when they should be ints. they are float only because they have missing values (NaN is a float), so they can be cast after imputing
    - days_since_previous_purchase
    - customer_tenure_days
    - prior_purchase_count
    - invoice_count_current_purchase
    - purchase_count_30d
    - purchase_count_90d
    - purchase_count_180d
    - purchase_count_365d
    - N_ Detalhes
- values like 3506.3900000000003 in Total a Pagar are not a data problem: computers cannot store most decimals exactly, so 3506.39 is shown with a tiny rounding error

### Noise feature
- there is a col named "random_noise". its distribution is the same for both classes, so it has no relation to the target and is not used as a feature

### String features against the target (chi-square)
- every row of a customer has the same values for the `clientes.csv` columns, and customers repeat a lot (one has 277 purchases), so the test is also run on one random purchase per customer
- the location and company columns (`company_name`, `address`, `postal_code`, `city`, `postal_area`, `municipality`, `street`, `commercial_region`, `island`, `country_or_region`) only look linked to the target because they identify the customer. with each customer counted once the link is gone
- `payment_type` and `salesperson` are the most useful string features. careful: "Payment 9" has a 95% repeat rate, but it comes from only 10 customers
- `transport_method` might help if rare levels are pooled
- `commercial_zone`, `warehouse` and `document_series` show almost no link to the target

---

# 2. Clean and pre-process

`Homework_Group06_Clean_Pre-Process.ipynb`: turn the raw data into features a model can use. Every step has to be applied to `test.csv` the same way, so the professor's recommendation applies: `from sklearn.pipeline import Pipeline`.

## To do
- turn `purchase_date` into numbers (month, day of week, ...); for now it is dropped by the pipeline
- maybe: a gap between train and validation in `TimeSeriesSplit` (the target looks 90 days ahead)

## What the notebook does

Plain pandas, on the training data (and the same on `test.csv`, except dropping rows):
- drop exact duplicates, first, before dropping any column (10 rows)
- drop columns: `customer_type`, `random_noise`, `customer_id` and the 7 columns that identify the customer (`company_name`, `address`, `postal_code`, `street`, `city`, `postal_area`, `municipality`)
- negative values -> NaN (functions `negatives_to_nan`)
- the 1,000,000,000 placeholder -> NaN (`placeholders_to_nan`)
- drop the 55 rows that are empty in 24 columns and have no date (train only)

Split: rows sorted by date, then `TimeSeriesSplit(n_splits=5)`: each fold trains on the past and validates on the next block, because `test.csv` is entirely later than train.

Pipeline (`preprocess`), learned on the training rows of each fold only:

| group | steps |
|---|---|
| numbers | cap at the 99th percentile -> impute median -> `StandardScaler` |
| `customer_since_year` | impute median -> `StandardScaler` (a year, so not capped) |
| missing flags | 0/1 "was missing" (`MissingIndicator`), not scaled |
| text | impute "Missing" -> one-hot (categories with < 30 rows grouped) |

## Notes
- outliers are capped, not removed: the IQR rule flags 66% of the rows (they repeat 60% vs 47%), and rows with a top-1% value repeat more often (67% vs 54%), so they carry information
- the capping step `CapOutliers` is in `cap_outliers.py`, not in the notebook, so the saved pipeline can be loaded by another notebook
- the missing flags are kept because missing `prior_*` values are linked to the target
- 0/1 columns are not scaled: scaling stretches rare flags and gives them too much weight in distance-based models

## Output (`data/pre-processed/`)
- `train_clean.csv`: cleaned training data + target, sorted by date
- `test_clean.csv`: cleaned test data
- `preprocess.joblib`: the pipeline, not fitted

---

# 3. Feature selection

`Homework_Group06_Feature_Selection.ipynb`: decide which features the model keeps. Loads the files saved by step 2.

## Strategy
1. identifier-like variables checked first, critically (not kept or dropped automatically)
2. filter methods: low variance (> 95% same value) -> drop; redundant (Spearman > 0.9) -> drop the weaker; relation with target (Spearman / chi-square, p < 0.05) -> 1 vote
3. wrapper: RFE (`RFECV`, logistic regression) -> 1 vote
4. embedded: Lasso (L1 logistic regression) -> 1 vote; random forest importance > `random_noise` -> 1 vote
5. keep if not low-variance, not redundant, and at least 2 of 4 votes; then checked with cross-validation (F1, `TimeSeriesSplit`)

## Results
- 22 of 38 features kept, same F1 as all 38 (logistic 0.733 vs 0.730, random forest 0.750 vs 0.751)
- `customer_id` excluded: linked to the target only because it recognises customers; adds nothing as a number, hurts logistic regression as a category, and can't work for new customers
- customer-level `clientes.csv` features (`commercial_region`, `island`, `customer_since_year`) dropped: removing them doesn't lower the score
- the strongest features are purchase frequency (counts over 90-365 days) and the typical gap between purchases

## Output (`data/pre-processed/`)
- `selected_features.joblib`: the 22 feature names
- `preprocess_selected.joblib`: the pre-processing pipeline using only those features, not fitted

---

# 4. Modelling and submission

`Homework_Group06_Modelling.ipynb`: train the model, assess it, and predict `test.csv`. Loads the files saved by steps 2 and 3 (it must stay in this `notebooks` folder so it can find `cap_outliers.py`).

## Choices
- assessment: 5-fold `TimeSeriesSplit`, because `test.csv` is entirely later than train and customers buy many times
- main metric: **macro F1**, second: **ROC AUC**. Not F1 or accuracy: always predicting "repeat" already gets F1 0.72, but only 0.36 macro F1 and 0.5 ROC AUC
- models: logistic regression (reference) and random forest; the random forest is better on every metric
- tuning: small grid on `min_samples_leaf` and `class_weight`; best is 20 and `"balanced"`

## Results
| model | macro F1 | ROC AUC |
|---|---|---|
| always predict 1 | 0.360 | 0.500 |
| logistic regression | 0.696 | 0.775 |
| random forest | 0.708 | 0.786 |
| random forest, tuned (final) | 0.719 | 0.790 |

- stable across the 5 periods: macro F1 0.705-0.727
- most recent fold: catches 70% of repeat buyers and 77% of non-repeat ones

## Output
- `submissions/Homework_Group06_submission.csv`: 986 predictions, `ID` unchanged, same format as `sample_submission.csv`

## Ideas to improve
- tune the 0.5 threshold that turns probabilities into 0/1
- date features from `purchase_date` (month, day of week)
- compare more models (the final project asks for at least 5)
