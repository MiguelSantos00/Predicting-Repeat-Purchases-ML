## dataset observations TRAIN

- there is a column named ID so we dont want it to be a feature so on read_csv we use index_col = "ID"
- there is a col named "random noise" so we have to check if there is any relation to the target
- since customer type is null for all rows its safe to drop