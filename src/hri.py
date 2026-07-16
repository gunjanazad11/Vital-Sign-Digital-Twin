import pandas as pd


def hr_abnormality(x):
    if 60 <= x <= 100:
        return 0
    if x < 60:
        return (60 - x) / 60
    return (x - 100) / 100


def rr_abnormality(x):
    if 12 <= x <= 20:
        return 0
    if x < 12:
        return (12 - x) / 12
    return (x - 20) / 20


def spo2_abnormality(x):
    if 95 <= x <= 100:
        return 0
    if x < 95:
        return (95 - x) / 95
    return 0


def add_abnormality_scores(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df['HR_abn'] = df['HR_corrected'].apply(hr_abnormality)
    df['RR_abn'] = df['RR_corrected'].apply(rr_abnormality)
    df['SpO2_abn'] = df['SpO2_corrected'].apply(spo2_abnormality)

    return df


def compute_hri(df: pd.DataFrame, w_hr: float = 0.3, w_rr: float = 0.3, w_spo2: float = 0.4) -> pd.DataFrame:
    df = df.copy()

    df['HRI'] = (
        w_hr * df['HR_abn'] * df['HR_rel'] +
        w_rr * df['RR_abn'] * df['RR_rel'] +
        w_spo2 * df['SpO2_abn'] * df['SpO2_rel']
    )

    return df


def hri_category(hri):
    if hri < 0.1:
        return 'Normal'
    if hri < 0.3:
        return 'Warning'
    return 'Critical'


def add_hri_category(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df['HRI_category'] = df['HRI'].apply(hri_category)
    return df
