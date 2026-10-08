"""Standalone Inference Script / CLI for Mining Machinery Predictive Maintenance.

Allows real-time single-asset scoring or batch CSV predictions using serialized models.
"""

import argparse
import json
import os
import sys
import joblib
import numpy as np
import pandas as pd

# Add parent directory to path if running directly from src/
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from src.features import (
    TYPE_MAPPING,
    FEATURE_COLUMNS,
    clean_raw_dataframe,
    engineer_features
)


def load_artifacts(models_dir: str = None, model_type: str = 'rf'):
    """Loads trained model and fitted scaler from disk."""
    if models_dir is None:
        models_dir = os.path.join(parent_dir, 'models')

    scaler_path = os.path.join(models_dir, 'scaler.joblib')
    if model_type == 'xgb':
        model_path = os.path.join(models_dir, 'xgboost_model.joblib')
    else:
        model_path = os.path.join(models_dir, 'random_forest_model.joblib')

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found at: {model_path}. Please train and save the model first.")
    if not os.path.exists(scaler_path):
        raise FileNotFoundError(f"Scaler file not found at: {scaler_path}. Please train and save the scaler first.")

    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    return model, scaler


def assign_risk_tier(probability: float) -> dict:
    """Categorizes failure probability into operational mining risk tier and prescribed action."""
    if probability < 0.25:
        return {
            "tier": "Low Risk",
            "code": "GREEN",
            "action": "Nominal operational state. Asset cleared for full production load. Routine telemetry logging."
        }
    elif probability < 0.65:
        return {
            "tier": "Moderate Risk",
            "code": "YELLOW",
            "action": "Early degradation detected. Issue dispatch warning. Perform non-intrusive vibration audit and lubrication check at next shift handover."
        }
    else:
        return {
            "tier": "High Risk",
            "code": "RED",
            "action": "CRITICAL ALARM. High risk of catastrophic mechanical seizure. Initiate controlled load reduction and dispatch emergency mechanical maintenance crew."
        }


def predict_single(
    variant: str,
    air_temp: float,
    process_temp: float,
    rotational_speed: float,
    torque: float,
    tool_wear: float,
    model_type: str = 'rf'
):
    """Performs inference on a single machinery telemetry reading."""
    model, scaler = load_artifacts(model_type=model_type)

    raw_data = pd.DataFrame([{
        'Type': variant.upper(),
        'Air temperature [K]': float(air_temp),
        'Process temperature [K]': float(process_temp),
        'Rotational speed [rpm]': float(rotational_speed),
        'Torque [Nm]': float(torque),
        'Tool wear [min]': float(tool_wear)
    }])

    feat_df = engineer_features(raw_data)
    X_input = feat_df[FEATURE_COLUMNS]
    X_scaled = pd.DataFrame(scaler.transform(X_input), columns=FEATURE_COLUMNS)

    prob = float(model.predict_proba(X_scaled)[0, 1])
    pred = int(model.predict(X_scaled)[0])
    risk_info = assign_risk_tier(prob)

    result = {
        "model_used": "Random Forest (Balanced)" if model_type == 'rf' else "XGBoost (Cost-Sensitive)",
        "telemetry_input": {
            "variant": variant.upper(),
            "air_temperature_K": air_temp,
            "process_temperature_K": process_temp,
            "rotational_speed_rpm": rotational_speed,
            "torque_Nm": torque,
            "tool_wear_min": tool_wear
        },
        "engineered_metrics": {
            "mechanical_power_W": round(float(feat_df['Mechanical Power (W)'].iloc[0]), 2),
            "delta_temperature_K": round(float(feat_df['Delta Temperature (K)'].iloc[0]), 2),
            "wear_strain_index": round(float(feat_df['Wear Strain Index'].iloc[0]), 2)
        },
        "prediction": {
            "failure_predicted": bool(pred == 1),
            "failure_probability": round(prob, 4),
            "risk_tier": risk_info["tier"],
            "alert_level": risk_info["code"],
            "prescribed_action": risk_info["action"]
        }
    }
    return result


def predict_batch(
    input_csv_path: str,
    output_csv_path: str,
    model_type: str = 'rf'
):
    """Performs batch scoring on an external CSV sensor dataset."""
    model, scaler = load_artifacts(model_type=model_type)

    print(f"Loading input data from: {input_csv_path}")
    df_raw = pd.read_csv(input_csv_path)

    df_cleaned = clean_raw_dataframe(df_raw)
    df_feat = engineer_features(df_cleaned)

    # Separate target if present
    has_actual = 'Machine failure' in df_feat.columns
    if has_actual:
        actual_status = df_feat['Machine failure']
        X_data = df_feat.drop(columns=['Machine failure'])[FEATURE_COLUMNS]
    else:
        actual_status = None
        X_data = df_feat[FEATURE_COLUMNS]

    X_scaled = pd.DataFrame(scaler.transform(X_data), columns=FEATURE_COLUMNS)
    probabilities = model.predict_proba(X_scaled)[:, 1]
    predictions = model.predict(X_scaled)

    results_df = df_raw.copy()
    if has_actual:
        results_df['Actual_Status'] = actual_status
    results_df['Predicted_Status'] = predictions
    results_df['Failure_Probability'] = np.round(probabilities, 4)
    results_df['Risk_Tier'] = [assign_risk_tier(p)['tier'] for p in probabilities]

    os.makedirs(os.path.dirname(os.path.abspath(output_csv_path)), exist_ok=True)
    results_df.to_csv(output_csv_path, index=False)
    print(f"Successfully processed {len(results_df)} records.")
    print(f"Exported batch predictions to: {output_csv_path}")
    print("\nRisk Tier Summary:")
    print(results_df['Risk_Tier'].value_counts())
    return results_df


def main():
    parser = argparse.ArgumentParser(
        description="Industrial Predictive Maintenance Inference CLI for Mining Machinery"
    )
    parser.add_argument("--model", choices=['rf', 'xgb'], default='rf', help="Model to use (default: rf)")

    # Batch mode
    parser.add_argument("--input-file", type=str, help="Path to input CSV file for batch predictions")
    parser.add_argument("--output-file", type=str, help="Path to output CSV file for batch predictions")

    # Single prediction mode
    parser.add_argument("--type", choices=['L', 'M', 'H', 'l', 'm', 'h'], help="Machine variant quality (L, M, or H)")
    parser.add_argument("--air-temp", type=float, help="Air temperature [K] (e.g. 298.1)")
    parser.add_argument("--process-temp", type=float, help="Process temperature [K] (e.g. 308.6)")
    parser.add_argument("--speed", type=float, help="Rotational speed [rpm] (e.g. 1500)")
    parser.add_argument("--torque", type=float, help="Torque [Nm] (e.g. 40.0)")
    parser.add_argument("--tool-wear", type=float, help="Tool wear [min] (e.g. 120)")

    args = parser.parse_args()

    if args.input_file:
        out = args.output_file or os.path.join(parent_dir, "data", "processed", "batch_predictions.csv")
        predict_batch(args.input_file, out, model_type=args.model)
    elif all(v is not None for v in [args.type, args.air_temp, args.process_temp, args.speed, args.torque, args.tool_wear]):
        result = predict_single(
            variant=args.type,
            air_temp=args.air_temp,
            process_temp=args.process_temp,
            rotational_speed=args.speed,
            torque=args.torque,
            tool_wear=args.tool_wear,
            model_type=args.model
        )
        print("\n" + "=" * 70)
        print("MINING EQUIPMENT PREDICTIVE MAINTENANCE INFERENCE REPORT")
        print("=" * 70)
        print(json.dumps(result, indent=2))
        print("=" * 70)
    else:
        parser.print_help()
        print("\nExample single asset check:")
        print("python src/predict.py --type M --air-temp 298.2 --process-temp 308.7 --speed 1400 --torque 65.5 --tool-wear 210")
        print("\nExample batch CSV scoring:")
        print("python src/predict.py --input-file data/raw/predictive_maintenance.csv --output-file data/processed/scored.csv")


if __name__ == '__main__':
    main()
