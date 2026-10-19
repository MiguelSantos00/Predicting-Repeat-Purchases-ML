"""Outlier capping step, shared by the pre-processing and modelling notebooks.

It lives in its own file (not inside a notebook) so a pipeline saved with
joblib in one notebook can be loaded in another.
"""
from sklearn.base import BaseEstimator, TransformerMixin


class CapOutliers(TransformerMixin, BaseEstimator):
    """Caps each column at its 99th percentile, learned in fit."""

    def fit(self, X, y=None):
        self.limits_ = X.quantile(0.99)   # learned from the training rows only
        return self

    def transform(self, X):
        return X.clip(upper=self.limits_, axis=1)

    def get_feature_names_out(self, input_features=None):
        return self.limits_.index.to_numpy()   # same columns as it receives
