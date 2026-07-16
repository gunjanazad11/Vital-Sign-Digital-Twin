import numpy as np
import pandas as pd


REQUIRED_TRACKS = ['HR', 'RR', 'SpO2']


def clean_physiological_ranges(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df.loc[(df['HR'] < 20) | (df['HR'] > 220), 'HR'] = np.nan
    df.loc[(df['RR'] < 4) | (df['RR'] > 45), 'RR'] = np.nan
    df.loc[(df['SpO2'] < 50) | (df['SpO2'] > 100), 'SpO2'] = np.nan

    return df


def create_masks(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df['HR_mask'] = df['HR'].notnull().astype(int)
    df['RR_mask'] = df['RR'].notnull().astype(int)
    df['SpO2_mask'] = df['SpO2'].notnull().astype(int)

    return df


def fill_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    for col in REQUIRED_TRACKS:
        df[col] = df[col].interpolate().bfill().ffill()

    return df


def add_time_index(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df['time'] = np.arange(len(df))
    return df


def reorder_columns(df: pd.DataFrame, caseid: int) -> pd.DataFrame:
    df = df.copy()
    df['caseid'] = caseid

    return df[['caseid', 'time', 'HR', 'RR', 'SpO2', 'HR_mask', 'RR_mask', 'SpO2_mask']]


def preprocess_case(df: pd.DataFrame, caseid: int) -> pd.DataFrame:
    df = df.copy()
    df.columns = ['HR', 'RR', 'SpO2']

    df = clean_physiological_ranges(df)
    df = create_masks(df)
    df = fill_missing_values(df)
    df = add_time_index(df)
    df = reorder_columns(df, caseid)

    return df
