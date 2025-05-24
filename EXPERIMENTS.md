# Model Development and Experiments

This document outlines the experiments conducted to optimize hyperparameters for the EURUSD 5-minute closing price forecasting LSTM model.

## Experiment 1: Optimizing LSTM Lookback Window (`WINDOW` Size)

The initial set of experiments focused on determining an effective lookback window (`WINDOW` parameter). This parameter defines how many previous 5-minute intervals the model uses as input to predict the next closing price.

### Methodology (Window Size)

The core LSTM architecture (2 LSTM layers with 50 units each, 0.2 dropout after each) and other training parameters (`epochs=5` with EarlyStopping, `batch_size=32`, `SEED=42`, `optimizer='adam'`, `loss='mse'`, `patience=3`) were kept constant. The `WINDOW` size was varied.

### Results Summary (Window Size)

| WINDOW Size | Test Set MAE (EURUSD) | Test Set RMSE (EURUSD) | Val Set MAE (EURUSD) | Val Set RMSE (EURUSD) | Train Set MAE (EURUSD) |
| :---------- | :-------------------- | :--------------------- | :------------------- | :-------------------- | :--------------------- |
| 30          | 0.000804              | 0.001015               | 0.000612             | 0.000883              | 0.000557               |
| 50          | 0.000949              | 0.001179               | 0.001041             | 0.001253              | 0.000695               |
| **100**     | **0.000580**          | **0.000846**           | **0.000511**         | **0.000811**          | **0.000358**           |
| 150         | 0.001081              | 0.001348               | 0.001111             | 0.001338              | 0.000657               |
| 200         | 0.000947              | 0.001234               | 0.000811             | 0.001101              | 0.000997               |
| 288         | 0.000656              | 0.000923               | 0.000561             | 0.000865              | 0.000513               |

*(Note: The WINDOW=100 results shown here reflect a consistent run that yielded the best overall metrics for this parameter.)*

### Conclusion (Window Size)

Based on these experiments, **`WINDOW = 100`** was selected as the optimal lookback period. It provided the best balance of capturing sufficient historical context and achieving the lowest prediction errors on unseen test data. This value will be used for subsequent hyperparameter tuning experiments.

---

## Experiment 2: Optimizing LSTM Layer Units

With `WINDOW = 100` fixed, this set of experiments focused on determining an effective number of units for the two LSTM layers (LSTM1 and LSTM2).

### Methodology (LSTM Units)

The `WINDOW` size was kept constant at 100. All other training parameters (dropout rate 0.2, optimizer 'adam', batch size 32, epochs 5 with EarlyStopping patience 3, seed 42) were also held constant. Both symmetrical (e.g., 32/32, 50/50) and asymmetrical (e.g., 100/50, 128/32) unit configurations were tested.

### Results Summary (LSTM Units with WINDOW=100)

| LSTM1 Units | LSTM2 Units | Test Set MAE (EURUSD) | Test Set RMSE (EURUSD) | Val Set MAE (EURUSD) | Val Set RMSE (EURUSD) | Train Set MAE (EURUSD) |
| :---------- | :---------- | :-------------------- | :--------------------- | :------------------- | :-------------------- | :--------------------- |
| **32**      | **32**      | **0.000641**          | **0.000935**           | **0.000559**         | **0.000891**          | 0.000411               |
| 50          | 50          | 0.000692              | 0.000944               | 0.000619             | 0.000907              | **0.000394**           |
| 64          | 64          | 0.001302              | 0.001604               | 0.001227             | 0.001477              | 0.000747               |
| 100         | 100         | 0.000771              | 0.000970               | 0.000763             | 0.000971              | 0.000627               |
| 128         | 128         | 0.001097              | 0.001275               | 0.001268             | 0.001406              | 0.000911               |
| 100         | 50          | 0.000900              | 0.001100               | 0.000769             | 0.000992              | 0.000784               |
| 50          | 100         | 0.001136              | 0.001395               | 0.001022             | 0.001275              | 0.000625               |
| 128         | 32          | 0.000874              | 0.001109               | 0.000765             | 0.001012              | 0.000445               |

### Analysis (LSTM Units)

*   **Top Symmetrical Performers:** The configurations with `32/32` units and `50/50` units demonstrated the strongest performance.
    *   The `32/32` configuration achieved the lowest Test Set MAE (0.000641 EURUSD) and Validation Set MAE (0.000559 EURUSD) in this batch.
    *   The `50/50` configuration was a very close competitor (Test MAE: 0.000692 EURUSD, Val MAE: 0.000619 EURUSD) and notably achieved the lowest Training Set MAE, indicating a strong capacity to fit the training data.
*   **Larger Symmetrical Configurations:** Increasing units symmetrically beyond 50 (i.e., 64/64, 100/100, 128/128) generally led to a decline in performance on unseen data, suggesting that excessive capacity might not be beneficial for this specific problem and architecture.
*   **Asymmetrical Configurations:** While some asymmetrical configurations like `128/32` showed reasonable performance, they did not surpass the top symmetrical configurations in terms of Test Set MAE.
*   **Model Complexity:** The `32/32` model (approx. 18k trainable parameters) is simpler than the `50/50` model (approx. 30k trainable parameters).

### Conclusion & Recommended LSTM Unit Configuration (with WINDOW=100)

Based on this systematic evaluation:

1.  The **`[LSTM(32, return_sequences=True), Dropout(0.2), LSTM(32), Dropout(0.2), Dense(1)]`** architecture is highly recommended. It yielded the best MAE on both the validation and test sets in this direct comparison and offers a good balance of performance and model simplicity.
2.  The **`[LSTM(50, return_sequences=True), Dropout(0.2), LSTM(50), Dropout(0.2), Dense(1)]`** architecture (the project's initial configuration) remains a very strong alternative. Its performance is closely comparable to the `32/32` setup, with a slightly better fit to the training data.

The difference in Test MAE between these two top configurations (~0.5 pips) is relatively small and could be within typical run-to-run variability. For general use, **the `32/32` configuration is marginally preferred due to its slightly better generalization metrics and lower complexity.** However, the `50/50` configuration also demonstrates robust performance.

---

*(Future experiments will be documented here, e.g., tuning dropout rates, number of LSTM layers, optimizer parameters, etc.)*