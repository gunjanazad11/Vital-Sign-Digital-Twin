import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error


def build_error_reliability_dataframe(observed_test_signals, y_pred_inv):
    hr_error = np.abs(observed_test_signals[:, 0] - y_pred_inv[:, 0])
    rr_error = np.abs(observed_test_signals[:, 1] - y_pred_inv[:, 1])
    spo2_error = np.abs(observed_test_signals[:, 2] - y_pred_inv[:, 2])

    alpha_hr = 0.2
    alpha_rr = 0.3
    alpha_spo2 = 0.5

    hr_rel = np.exp(-alpha_hr * hr_error)
    rr_rel = np.exp(-alpha_rr * rr_error)
    spo2_rel = np.exp(-alpha_spo2 * spo2_error)

    error_rel_df = pd.DataFrame({
        'HR_observed': observed_test_signals[:, 0],
        'HR_pred_clean': y_pred_inv[:, 0],
        'HR_error': hr_error,
        'HR_rel': hr_rel,

        'RR_observed': observed_test_signals[:, 1],
        'RR_pred_clean': y_pred_inv[:, 1],
        'RR_error': rr_error,
        'RR_rel': rr_rel,

        'SpO2_observed': observed_test_signals[:, 2],
        'SpO2_pred_clean': y_pred_inv[:, 2],
        'SpO2_error': spo2_error,
        'SpO2_rel': spo2_rel,
    })

    return error_rel_df


def add_low_reliability_flags(error_rel_df: pd.DataFrame, threshold: float = 0.7) -> pd.DataFrame:
    df = error_rel_df.copy()

    df['HR_low_rel_flag'] = (df['HR_rel'] < threshold).astype(int)
    df['RR_low_rel_flag'] = (df['RR_rel'] < threshold).astype(int)
    df['SpO2_low_rel_flag'] = (df['SpO2_rel'] < threshold).astype(int)

    return df


def apply_self_correction(error_rel_df: pd.DataFrame, threshold: float = 0.7) -> pd.DataFrame:
    df = error_rel_df.copy()

    df['HR_corrected'] = np.where(
        df['HR_rel'] < threshold,
        df['HR_pred_clean'],
        df['HR_observed']
    )

    df['RR_corrected'] = np.where(
        df['RR_rel'] < threshold,
        df['RR_pred_clean'],
        df['RR_observed']
    )

    df['SpO2_corrected'] = np.where(
        df['SpO2_rel'] < threshold,
        df['SpO2_pred_clean'],
        df['SpO2_observed']
    )

    return df


def evaluate_self_correction(error_rel_df: pd.DataFrame, y_true_inv):
    metrics = {}

    observed_cols = ['HR_observed', 'RR_observed', 'SpO2_observed']
    corrected_cols = ['HR_corrected', 'RR_corrected', 'SpO2_corrected']
    signal_names = ['HR', 'RR', 'SpO2']

    for i, signal in enumerate(signal_names):
        observed_mae = mean_absolute_error(y_true_inv[:, i], error_rel_df[observed_cols[i]])
        observed_rmse = np.sqrt(mean_squared_error(y_true_inv[:, i], error_rel_df[observed_cols[i]]))

        corrected_mae = mean_absolute_error(y_true_inv[:, i], error_rel_df[corrected_cols[i]])
        corrected_rmse = np.sqrt(mean_squared_error(y_true_inv[:, i], error_rel_df[corrected_cols[i]]))

        metrics[signal] = {
            'observed_mae': observed_mae,
            'observed_rmse': observed_rmse,
            'corrected_mae': corrected_mae,
            'corrected_rmse': corrected_rmse,
        }

    return metrics
