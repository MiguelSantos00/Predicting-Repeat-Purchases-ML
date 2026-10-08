# TO DO
- continue exploration : multivariate relationships
- handle missing values
- enconding the string features
- handle outliers
- feature selection
...

## dataset observations

- there is a column named ID so we dont want it to be a feature so on read_csv we use index_col = "ID"
- there is a col named "random noise" so we have to check if there is any relation to the target
- since customer type is null for all rows its safe to drop
- checking for duplicates with df.duplicates().sum() we get 10 duplicates
- while checking missing values there are a lot of features with "65", after looking at those rows we can affirm that they are all from the same rows. (raw_data.isna().sum() == 65).sum() gives us 24. this means that 24 features have 65 NA values, them being the same exact rows. since these give us no info i believe they are safe to drop.
- total number of rows is 6534, 20% of that is 1300, there are features with more than 1300 total values missing, those could be dropped
- some float columns contain negative values. those rows can be removed
- street has around 60% missing values so probably safe to drop
- lots of very big outliers
- some variables are floats when should be ints to save space
    - days_since_previous_purchase
    - customer_tenure_days
    - prior_purchase_count
    - invoice_count_current_purchase
    - purchase_count_30d
    - purchase_count_90d
    - purchase_count_180d
    - purchase_count_365d
    - n _detalhes
- Float values sometimes dont make sense because they have too many decimal cases, for example in Total_a_pagar there is the following value 3506.3900000000003, how is someone going to spend 00000000003 cents?

#### string features we think might not matter
- payment type
- transport method
- warehouse
...