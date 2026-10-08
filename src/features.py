"""Feature engineering and preprocessing utilities for mining machinery telemetry."""

import numpy as np
import pandas as pd


TYPE_MAPPING = {'L': 0, 'M': 1, 'H': 2}
LEAKAGE_AND_ID_COLS = [
    'UDI', 'Product ID', 'Failure Type', 'TWF', 'HDF', 'PWF', 'OSF', 'RNF'
]

# Standard feature column order expected by trained models
FEATURE_COLUMNS = [
    'Type',
    'Air temperature (K)',
    'Process temperature (K)',
    'Rotational speed (rpm)',
    'Torque (Nm)',
    'Tool wear (min)',
    'Mechanical Power (W)',
    'Delta Temperature (K)',
    'Wear Strain Index'
]


def clean_raw_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Standardizes target name and drops non-predictive/leakage columns."""
    df_clean = df.copy()

    # Standardize target label
    if 'Target' in df_clean.columns and 'Machine failure' not in df_clean.columns:
        df_clean = df_clean.rename(columns={'Target': 'Machine failure'})

    # Drop IDs and specific failure mode indicators
    cols_to_drop = [c for c in LEAKAGE_AND_ID_COLS if c in df_clean.columns]
    if cols_to_drop:
        df_clean = df_clean.drop(columns=cols_to_drop)

    return df_clean


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Computes physics-based domain features and sanitizes column names."""
    df_feat = df.copy()

    # Ordinal encoding of variant type if non-numeric
    if 'Type' in df_feat.columns and not pd.api.types.is_numeric_dtype(df_feat['Type']):
        df_feat['Type'] = df_feat['Type'].astype(str).str.upper().map(TYPE_MAPPING)

    # Resolve column names for brackets or parentheses
    def get_col(possible_names):
        for name in possible_names:
            if name in df_feat.columns:
                return df_feat[name]
        raise KeyError(f"None of {possible_names} found in DataFrame columns: {df_feat.columns.tolist()}")

    torque = get_col(['Torque [Nm]', 'Torque (Nm)', 'Torque'])
    speed = get_col(['Rotational speed [rpm]', 'Rotational speed (rpm)', 'Rotational speed'])
    proc_temp = get_col(['Process temperature [K]', 'Process temperature (K)', 'Process temperature'])
    air_temp = get_col(['Air temperature [K]', 'Air temperature (K)', 'Air temperature'])
    tool_wear = get_col(['Tool wear [min]', 'Tool wear (min)', 'Tool wear'])

    # 1. Mechanical Power (W) = Torque [Nm] * Rotational speed [rpm] * (2 * pi / 60)
    df_feat['Mechanical Power (W)'] = torque * speed * (2.0 * np.pi / 60.0)

    # 2. Delta Temperature (K) = Process temperature [K] - Air temperature [K]
    df_feat['Delta Temperature (K)'] = proc_temp - air_temp

    # 3. Wear Strain Index = Torque [Nm] * Tool wear [min]
    df_feat['Wear Strain Index'] = torque * tool_wear

    # Replace brackets with parentheses for XGBoost compatibility
    df_feat.columns = [col.replace('[', '(').replace(']', ')') for col in df_feat.columns]

    return df_feat
