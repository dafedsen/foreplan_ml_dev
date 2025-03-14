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