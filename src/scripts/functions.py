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

def get_encoder_decoder_length(df, group_ids, freq: str):
    series_lengths = df.groupby(group_ids).size()
    min_series_len = series_lengths.min()

    # aturan default per frekuensi
    if freq.upper().startswith("D"):  # Daily
        prediction_length = max(2, min(7, min_series_len // 5))
    elif freq.upper().startswith("W"):  # Weekly
        prediction_length = max(2, min(4, min_series_len // 3))
    elif freq.upper().startswith("M"):  # Monthly
        prediction_length = max(2, min(3, min_series_len // 2))
    else:
        prediction_length = max(2, min(5, min_series_len // 5))

    encoder_length = max(2, min(min_series_len - prediction_length, prediction_length * 2))

    # fallback kalau terlalu pendek
    if encoder_length + prediction_length > min_series_len:
        encoder_length = max(2, min_series_len - 1)
        prediction_length = max(1, min_series_len - encoder_length)

    return {
        "encoder_length": int(encoder_length),
        "prediction_length": int(prediction_length),
        "min_series_len": int(min_series_len),
    }

def safe_training_cutoff(df, time_idx_col, prediction_length, buffer: int = 1):
    """
    Pastikan training_cutoff cukup mundur supaya decoder window muat.
    """
    max_idx = df[time_idx_col].max()
    cutoff = max_idx - prediction_length - buffer
    if cutoff <= df[time_idx_col].min():
        raise ValueError(
            f"Cutoff {cutoff} too kecil untuk series length {max_idx}. "
            f"Coba kurangi prediction_length."
        )
    return cutoff

def adjust_params(df, encoder_length, prediction_length):
    """
    Adjust encoder, prediction, cutoff, and min_series_len automatically
    based on dataset length.
    """
    max_time_idx = df["time_idx"].max()

    # minimal panjang series yg bisa dipakai
    min_series_len = encoder_length + prediction_length  

    # cutoff otomatis (pastikan masih ada ruang prediksi)
    cutoff = max_time_idx - prediction_length

    return {
        "encoder_length": encoder_length,
        "prediction_length": prediction_length,
        "min_series_len": min_series_len,
        "cutoff": cutoff,
    }