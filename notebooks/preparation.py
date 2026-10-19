import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder


def fill_data(train, valid):
    train = train.copy()
    valid = valid.copy()
    num = train.select_dtypes(include="number").columns.tolist()
    cat = train.select_dtypes(exclude="number").columns.tolist()
    for col in num:
        if train[col].isna().any():
            train[col + "_missing"] = train[col].isna().astype(int)
            valid[col + "_missing"] = valid[col].isna().astype(int)
    med = train[num].median()
    train[num] = train[num].fillna(med)
    valid[num] = valid[num].fillna(med)
    for col in cat:
        train[col] = train[col].fillna("Missing")
        valid[col] = valid[col].fillna("Missing")
    return train, valid, num, cat


def scale_data(train, valid, num):
    money = ["Total a Pagar", "Total Bruto", "Total Liquido", "Total Imp_",
             "Total Portes", "Total Desc_ Global", "Total Desc_ Linha", "Custo",
             "prior_total_spend", "prior_mean_purchase_value",
             "prior_min_purchase_value", "prior_max_purchase_value",
             "prior_std_purchase_value", "spend_30d", "spend_90d",
             "spend_180d", "spend_365d"]
    train[money] = np.log1p(train[money])
    valid[money] = np.log1p(valid[money])
    scaler = StandardScaler()
    train[num] = scaler.fit_transform(train[num])
    valid[num] = scaler.transform(valid[num])
    return train, valid


def encode_data(train, valid, cat):
    enc = OneHotEncoder(min_frequency=20, handle_unknown="infrequent_if_exist",
                        sparse_output=False)
    cat_train = enc.fit_transform(train[cat])
    cat_valid = enc.transform(valid[cat])
    cols = enc.get_feature_names_out(cat)
    cat_train = pd.DataFrame(cat_train, columns=cols, index=train.index)
    cat_valid = pd.DataFrame(cat_valid, columns=cols, index=valid.index)
    train = train.drop(columns=cat).join(cat_train)
    valid = valid.drop(columns=cat).join(cat_valid)
    return train, valid
