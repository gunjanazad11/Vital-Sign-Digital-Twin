import os
import pandas as pd

from fault_injection import create_corrupted_dataset
from gru_model import (
    fill_model_inputs,
    create_reliability_targets,
    create_sequences,
    scale_sequences,
    prepare_dataloaders,
    GRURegressor,
    train_gru_model,
    predict_with_uncertainty,
    inverse_targets,
    evaluate_predictions,
    reconstruct_observed_test_signals,
)
from self_correction import (
    build_error_reliability_dataframe,
    add_low_reliability_flags,
    apply_self_correction,
    evaluate_self_correction,
)
from hri import add_abnormality_scores, compute_hri, add_hri_category


DATA_PATH = os.path.join("data", "processed", "vitaldb_10cases_clean.csv")
RESULTS_PATH = os.path.join("data", "results", "final_digital_twin_results.csv")


def uncertainty_level(x):
    if x < 0.5:
        return "High Confidence"
    if x < 1.5:
        return "Moderate Confidence"
    return "Low Confidence"


def main():
    print("Loading processed dataset...")
    df = pd.read_csv(DATA_PATH)
    print("Input shape:", df.shape)

    print("Creating corrupted dataset...")
    corrupted_df = create_corrupted_dataset(df)

    print("Preparing model inputs...")
    model_df = fill_model_inputs(corrupted_df)
    model_df = create_reliability_targets(model_df)

    print("Creating sequences...")
    sequences, targets, faults = create_sequences(model_df, window_size=20)
    print("Sequences shape:", sequences.shape)
    print("Targets shape:", targets.shape)

    print("Scaling sequences...")
    x_scaled, y_scaled, x_scaler, y_scaler = scale_sequences(sequences, targets)

    print("Preparing dataloaders...")
    (
        train_loader,
        test_loader,
        x_train,
        x_test,
        y_train,
        y_test,
        x_train_tensor,
        x_test_tensor,
    ) = prepare_dataloaders(x_scaled, y_scaled, batch_size=64)

    print("Building GRU model...")
    input_size = x_train.shape[2]
    hidden_size = 64
    output_size = y_train.shape[1]

    model = GRURegressor(input_size, hidden_size, output_size)

    print("Training GRU model...")
    train_gru_model(model, train_loader, epochs=10, lr=0.001)

    print("Generating predictions with uncertainty...")
    y_pred_inv, y_pred_std = predict_with_uncertainty(model, x_test_tensor, y_scaler, n_samples=20)
    y_test_inv = inverse_targets(y_test, y_scaler)

    print("Evaluating reconstruction...")
    overall_mae, overall_rmse, signal_metrics = evaluate_predictions(y_test_inv, y_pred_inv)
    print("Overall MAE:", overall_mae)
    print("Overall RMSE:", overall_rmse)
    print("Per-signal metrics:", signal_metrics)

    print("Computing error-based reliability...")
    observed_test_signals = reconstruct_observed_test_signals(x_test, x_scaler)
    error_rel_df = build_error_reliability_dataframe(observed_test_signals, y_pred_inv)
    error_rel_df = add_low_reliability_flags(error_rel_df, threshold=0.7)

    error_rel_df['HR_uncertainty'] = y_pred_std[:, 0]
    error_rel_df['RR_uncertainty'] = y_pred_std[:, 1]
    error_rel_df['SpO2_uncertainty'] = y_pred_std[:, 2]

    print("Applying self-correction...")
    error_rel_df = apply_self_correction(error_rel_df, threshold=0.7)
    correction_metrics = evaluate_self_correction(error_rel_df, y_test_inv)
    print("Self-correction metrics:", correction_metrics)

    print("Computing HRI...")
    error_rel_df = add_abnormality_scores(error_rel_df)
    error_rel_df = compute_hri(error_rel_df)
    error_rel_df = add_hri_category(error_rel_df)

    error_rel_df['HRI_uncertainty'] = (
        0.3 * error_rel_df['HR_uncertainty'] +
        0.3 * error_rel_df['RR_uncertainty'] +
        0.4 * error_rel_df['SpO2_uncertainty']
    )
    error_rel_df['HRI_confidence'] = error_rel_df['HRI_uncertainty'].apply(uncertainty_level)

    print("Saving results...")
    os.makedirs(os.path.dirname(RESULTS_PATH), exist_ok=True)
    error_rel_df.to_csv(RESULTS_PATH, index=False)

    print("Saved to:", RESULTS_PATH)
    print("HRI category counts:")
    print(error_rel_df["HRI_category"].value_counts())
    print("HRI confidence counts:")
    print(error_rel_df["HRI_confidence"].value_counts())


if __name__ == "__main__":
    main()
