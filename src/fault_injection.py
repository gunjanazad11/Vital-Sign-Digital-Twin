import numpy as np
import pandas as pd


SIGNAL_COLUMNS = ['HR', 'RR', 'SpO2']


def add_clean_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df['HR_clean'] = df['HR']
    df['RR_clean'] = df['RR']
    df['SpO2_clean'] = df['SpO2']

    return df


def initialize_fault_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df['HR_fault'] = 0
    df['RR_fault'] = 0
    df['SpO2_fault'] = 0

    return df


def inject_missing_values(df: pd.DataFrame, frac: float = 0.05, seed: int = 42) -> pd.DataFrame:
    df = df.copy()
    np.random.seed(seed)

    for col in SIGNAL_COLUMNS:
        idx = df.sample(frac=frac, random_state=seed).index
        df.loc[idx, col] = np.nan
        df.loc[idx, f'{col}_fault'] = 1

    return df


def inject_noise(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    noise_idx_hr = df.sample(frac=0.05, random_state=1).index
    noise_idx_rr = df.sample(frac=0.05, random_state=2).index
    noise_idx_spo2 = df.sample(frac=0.05, random_state=3).index

    df.loc[noise_idx_hr, 'HR'] += np.random.normal(0, 8, size=len(noise_idx_hr))
    df.loc[noise_idx_rr, 'RR'] += np.random.normal(0, 3, size=len(noise_idx_rr))
    df.loc[noise_idx_spo2, 'SpO2'] += np.random.normal(0, 2, size=len(noise_idx_spo2))

    df.loc[noise_idx_hr, 'HR_fault'] = 1
    df.loc[noise_idx_rr, 'RR_fault'] = 1
    df.loc[noise_idx_spo2, 'SpO2_fault'] = 1

    return df


def inject_spikes(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    spike_idx_hr = df.sample(frac=0.02, random_state=4).index
    spike_idx_rr = df.sample(frac=0.02, random_state=5).index
    spike_idx_spo2 = df.sample(frac=0.02, random_state=6).index

    df.loc[spike_idx_hr, 'HR'] += np.random.choice([25, -25], size=len(spike_idx_hr))
    df.loc[spike_idx_rr, 'RR'] += np.random.choice([10, -10], size=len(spike_idx_rr))
    df.loc[spike_idx_spo2, 'SpO2'] += np.random.choice([8, -8], size=len(spike_idx_spo2))

    df.loc[spike_idx_hr, 'HR_fault'] = 1
    df.loc[spike_idx_rr, 'RR_fault'] = 1
    df.loc[spike_idx_spo2, 'SpO2_fault'] = 1

    return df


def clean_corrupted_ranges(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df.loc[(df['HR'] < 20) | (df['HR'] > 220), 'HR'] = np.nan
    df.loc[(df['RR'] < 4) | (df['RR'] > 45), 'RR'] = np.nan
    df.loc[(df['SpO2'] < 50) | (df['SpO2'] > 100), 'SpO2'] = np.nan

    return df


def create_corrupted_dataset(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df = add_clean_columns(df)
    df = initialize_fault_columns(df)
    df = inject_missing_values(df)
    df = inject_noise(df)
    df = inject_spikes(df)
    df = clean_corrupted_ranges(df)

    return df
