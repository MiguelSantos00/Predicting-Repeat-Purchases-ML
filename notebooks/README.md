# Notebooks

Each step of the project has its own notebook, run in this order:

| step | notebook | status |
|---|---|---|
| 1. Exploration | `Homework_Group06_Exploration.ipynb` | in progress |
| 2. Clean and pre-process | `Homework_Group06_Clean_Pre-Process.ipynb` | in progress |
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
- drop duplicates before dropping any column (dropping columns first can make different rows look identical)
- handle missing values
- handle negative values (set to NaN, maybe keep a flag column)
- encoding the string features
- handle outliers
- turn `purchase_date` into numbers (month, day of week, ...) or drop it
- feature selection
- validation split: `test.csv` covers 2025-05 to 2026-10 and train ends in 2025-02, so validating on the most recent part of train is closer to the real test than a random split
...

## Notes

---

# 3. Modelling and submission

Not started yet.
