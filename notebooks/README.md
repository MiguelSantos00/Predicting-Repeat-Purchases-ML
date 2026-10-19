# Notebooks

Each step of the project has its own notebook, run in this order:

| step | notebook | status |
|---|---|---|
| 1. Exploration | `Homework_Group06_Exploration.ipynb` | in progress |
| 2. Clean and pre-process | `Homework_Group06_Clean_Pre-Process.ipynb` | done (feature selection and date features left for later) |
| 3. Modelling and submission | not created yet | to do |

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
- feature selection (better done in modelling, by comparing scores)
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
- outliers are capped, not removed: the IQR rule flags 66% of the rows, and those rows repeat more often (67% vs 54%)
- the capping step `CapOutliers` is in `cap_outliers.py`, not in the notebook, so the saved pipeline can be loaded by another notebook
- the missing flags are kept because missing `prior_*` values are linked to the target
- 0/1 columns are not scaled: scaling stretches rare flags and gives them too much weight in distance-based models

## Output (`data/pre-processed/`)
- `train_clean.csv`: cleaned training data + target, sorted by date
- `test_clean.csv`: cleaned test data
- `preprocess.joblib`: the pipeline, not fitted

---

# 3. Modelling and submission

Not started yet. The notebook goes in this `notebooks` folder (so it can find `cap_outliers.py`) and starts with:

```python
train = pd.read_csv("../data/pre-processed/train_clean.csv", index_col="ID")
X, y = train.drop(columns=["repeat_purchase_90d"]), train["repeat_purchase_90d"]
X_test = pd.read_csv("../data/pre-processed/test_clean.csv", index_col="ID")
preprocess = joblib.load("../data/pre-processed/preprocess.joblib")
time_split = TimeSeriesSplit(n_splits=5)

model = Pipeline([("preprocess", preprocess), ("model", ...)])
cross_val_score(model, X, y, cv=time_split)
```
