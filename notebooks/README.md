## dataset observations TRAIN

- there is a column named ID so we dont want it to be a feature so on read_csv we use index_col = "ID"
- there is a col named "random noise" so we have to check if there is any relation to the target
- since customer type is null for all rows its safe to drop
- checking for duplicates with df.duplicates().sum() we get 10 duplicates
- while checking missing values there are a lot of features with "65", after looking at those rows we can affirm that they are all from the same rows. (raw_data.isna().sum() == 65).sum() gives us 24. this means that 24 features have 65 NA values, them being the same exact rows. since these give us no info i believe they are safe to drop.
- total number of rows is 6534, 20% of that is 1300, there are features with more than 1300 total values missing, those could be dropped
- some float columns contain negative values. those rows can be removed