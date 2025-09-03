import cupy as cp
from cuml.metrics.regression import mean_squared_error
import re
import cudf as cd
import pandas as pd

def extract_number(text):
    match = re.search(r'\d+', text)
    return int(match.group()) if match else None

def adjusting_data(df, method, sigma):
    df['hist_value'] = pd.to_numeric(df['hist_value'], errors='coerce').fillna(0)
    df['flag_adj'] = df['flag_adj'].astype(str)

    if method == 'Lower/Upper Bound':
        mean = df['hist_value'].mean()
        std = df['hist_value'].std()
        treshold = float(sigma)

        upper = mean + treshold * std
        lower = mean - treshold * std

        median = (upper + lower) / 2

        if lower < 0:
            lower = 0.0

        df.loc[(df['flag_adj'] == 'Y') & (df['hist_value'] > median), 'hist_value'] = float(upper)
        df.loc[(df['flag_adj'] == 'Y') & (df['hist_value'] <= median), 'hist_value'] = float(lower)

    elif method == 'Average':
        mean = float(df['hist_value'].mean())

        if mean > 0:
            mean = 0.0

        df.loc[(df['flag_adj'] == 'Y'), 'hist_value'] = mean

    elif method == 'Remove from History':
        df.loc[(df['flag_adj'] == 'Y'), 'hist_value'] = 0

def cleansing_outliers(df, sigma):

    mean = df['hist_value'].mean()
    std = df['hist_value'].std()
    treshold = float(sigma)

    upper = mean + treshold * std
    lower = mean - treshold * std

    def replace_outliers(value):
        if value < lower:
            return float(lower)
        elif value > upper:
            return float(upper)
        else:
            return float(value)

    df['hist_value'] = df['hist_value'].clip(lower, upper)

def get_weeks_by_number(weeks):
    Z = [[0]]
    for a in range(weeks - 1):
      w = Z[a][0] + 7
      Z.append([w])

    Z = cp.array(Z)
    return Z

def get_rmse(y_true, y_pred):
    mse = mean_squared_error(y_true, y_pred)
    rmse = cp.sqrt(mse)
    return rmse

def mean_absolute_percentage_error(y_true, y_pred):
    y_true, y_pred = cp.array(y_true), cp.array(y_pred)
    return cp.mean(cp.abs((y_true - y_pred) / y_true))

def get_mad(y_pred):
    mad = cp.median(cp.absolute(y_pred - cp.median(y_pred)))
    return mad

def get_bias(y_true, y_pred):
    min_length = min(len(y_true), len(y_pred))

    y_true = y_true[:min_length]
    y_pred = y_pred[:min_length]

    bias = cp.sum(y_pred - y_true) / cp.sum(y_true) * 100
    return bias * 0.01

def print_process_percentage(current_process):
    print('\nCurrent Progress : ', current_process, '%\n')

class GPUMinMaxScaler:
    def __init__(self, feature_range=(0, 1)):
        self.feature_range = feature_range
        self.data_min_ = None
        self.data_max_ = None
        self.scale_ = None
        self.min_ = None
        self.n_features_in_ = None  # Tambahkan ini

    def fit(self, X):
        if isinstance(X, cd.Series):
            X = X.to_cupy().reshape(-1, 1)
        elif isinstance(X, cd.DataFrame):
            X = X.to_cupy()
        elif isinstance(X, cp.ndarray) and X.ndim == 1:
            X = X.reshape(-1, 1)
        elif not isinstance(X, cp.ndarray):
            raise ValueError("Input must be a cudf.Series, cudf.DataFrame or cupy.ndarray")

        self.data_min_ = cp.min(X, axis=0)
        self.data_max_ = cp.max(X, axis=0)
        self.scale_ = (self.feature_range[1] - self.feature_range[0]) / (self.data_max_ - self.data_min_ + 1e-9)
        self.min_ = self.feature_range[0] - self.data_min_ * self.scale_
        self.n_features_in_ = X.shape[1]  # Simpan jumlah fitur
        return self

    def transform(self, X):
        if isinstance(X, cd.Series):
            X = X.to_cupy().reshape(-1, 1)
        elif isinstance(X, cd.DataFrame):
            X = X.to_cupy()
        elif isinstance(X, cp.ndarray) and X.ndim == 1:
            X = X.reshape(-1, 1)
        elif not isinstance(X, cp.ndarray):
            raise ValueError("Input must be a cudf.Series, cudf.DataFrame or cupy.ndarray")

        return X * self.scale_ + self.min_

    def fit_transform(self, X):
        return self.fit(X).transform(X)

    def inverse_transform(self, X_scaled):
        if isinstance(X_scaled, cd.Series):
            X_scaled = X_scaled.to_cupy().reshape(-1, 1)
        elif isinstance(X_scaled, cd.DataFrame):
            X_scaled = X_scaled.to_cupy()
        elif isinstance(X_scaled, cp.ndarray) and X_scaled.ndim == 1:
            X_scaled = X_scaled.reshape(-1, 1)
        elif not isinstance(X_scaled, cp.ndarray):
            raise ValueError("Input must be a cudf.Series, cudf.DataFrame or cupy.ndarray")

        return (X_scaled - self.min_) / self.scale_


