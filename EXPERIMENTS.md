# Hyperparameter Tuning Experiments

This document summarizes the experiments conducted to optimize the hyperparameters of the LSTM model for EURUSD 5-minute price forecasting. Each experiment aims to find settings that improve the model's predictive performance, primarily measured by Mean Absolute Error (MAE) and Root Mean Squared Error (RMSE) on the test set.

## Table of Contents
1.  [Optimizing LSTM Lookback Window (`WINDOW` Size)](#experiment-1-optimizing-lstm-lookback-window-window-size)
    *   [Methodology](#methodology-window-size)
    *   [Results Summary](#results-summary-window-size)
    *   [Analysis](#analysis-window-size)
    *   [Conclusion & Recommended `WINDOW`](#conclusion--recommended-window)
2.  *(Future Experiment Summaries will be added here)*

---

## Experiment 1: Optimizing LSTM Lookback Window (`WINDOW` Size)
<a name="experiment-1-optimizing-lstm-lookback-window-window-size"></a>

The objective of this experiment was to determine an effective lookback window (`WINDOW` parameter) for the LSTM model. The `WINDOW` parameter defines how many previous 5-minute time steps the model uses as input to predict the next closing price.

For detailed logs, individual run outputs, or specific code configurations for each run in this experiment, please refer to the `Hiperparameter Tuning Experiments/Window_Size_Optimization/` directory in this repository. *(You'll create this directory and can place individual notebooks or log files there if needed).*

### Methodology
<a name="methodology-window-size"></a>

The core LSTM architecture and other training parameters were kept constant across all runs in this experiment series:

*   **Model Architecture:**
    *   LSTM Layer 1: 50 units, `return_sequences=True`
    *   Dropout: 0.2
    *   LSTM Layer 2: 50 units
    *   Dropout: 0.2
    *   Dense Output Layer: 1 unit
*   **Training Parameters:**
    *   `epochs`: 5 (with EarlyStopping, `patience=3`)
    *   `batch_size`: 32
    *   `SEED`: 42 (for reproducibility)
    *   `optimizer`: 'adam'
    *   `loss`: 'mse'

The `WINDOW` size was varied across the following values: 30, 50, 100, 150, 200, and 288.

### Results Summary
<a name="results-summary-window-size"></a>

Performance was evaluated using MAE and RMSE on the training, validation, and test sets (original EURUSD price scale).

| WINDOW Size | Test Set MAE (EURUSD) | Test Set RMSE (EURUSD) | Val Set MAE (EURUSD) | Val Set RMSE (EURUSD) | Train Set MAE (EURUSD) |
| :---------- | :-------------------- | :--------------------- | :------------------- | :-------------------- | :--------------------- |
| 30          | 0.000804              | 0.001015               | 0.000612             | 0.000883              | 0.000557               |
| 50          | 0.000949              | 0.001179               | 0.001041             | 0.001253              | 0.000695               |
| **100**     | **0.000580**          | **0.000846**           | **0.000511**         | **0.000811**          | **0.000358**           |
| 150         | 0.001081              | 0.001348               | 0.001111             | 0.001338              | 0.000657               |
| 200         | 0.000947              | 0.001234               | 0.000811             | 0.001101              | 0.000997               |
| 288         | 0.000656              | 0.000923               | 0.000561             | 0.000865              | 0.000513               |

*(Note: The results for WINDOW=100 are based on a consistently reproduced run with the same architecture and parameters, yielding the best observed metrics in this context.)*

### Analysis
<a name="analysis-window-size"></a>

*   **Optimal Range:** `WINDOW = 100` yielded the best overall performance, achieving the lowest MAE and RMSE on both validation and test sets. This suggests it captures an optimal balance of historical context for this specific prediction task.
*   **Performance of Other Windows:**
    *   `WINDOW = 30` performed well on the validation set but did not generalize as effectively to the test set compared to `WINDOW = 100`.
    *   `WINDOW = 288` (representing a full trading day) also demonstrated strong performance, with results close to those of `WINDOW = 100`. This indicates that daily patterns might hold some predictive value.
    *   Windows of 50, 150, and 200 generally resulted in higher prediction errors compared to the 100 or 288 settings.
*   **Computational Trade-off:** Larger window sizes significantly increase training time and computational resource requirements.

### Conclusion & Recommended `WINDOW`
<a name="conclusion--recommended-window"></a>

Based on this experimental sweep, **`WINDOW = 100`** is selected as the recommended lookback period for the current LSTM model configuration and dataset. It achieved the best predictive accuracy on unseen test data.

While `WINDOW = 288` also showed promising results, `WINDOW = 100` provides a slight edge in performance with a more favorable computational cost. This `WINDOW` size will be used as the default in the main forecasting notebook. Further fine-tuning around this value may be explored in subsequent experiments.

---

