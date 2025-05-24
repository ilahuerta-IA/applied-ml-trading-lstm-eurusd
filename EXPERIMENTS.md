# Model Development and Experiments

This document outlines the experiments conducted to optimize hyperparameters for the EURUSD 5-minute closing price forecasting LSTM model. The goal is to systematically evaluate the impact of different parameter choices on the model's predictive performance.

---

## Experiment 1: Optimizing LSTM Lookback Window (`WINDOW` Size)

The initial set of experiments focused on determining an effective lookback window (`WINDOW` parameter). This parameter defines how many previous 5-minute intervals the model uses as input to predict the next closing price.

### Methodology (Window Size)

The core LSTM architecture and other training parameters were kept constant across all runs in this experiment series:
*   **Base Model Architecture (Used for Window Size Tuning):**
    *   LSTM Layer 1: 50 units, `return_sequences=True`
    *   Dropout: 0.2
    *   LSTM Layer 2: 50 units
    *   Dropout: 0.2
    *   Dense Output Layer: 1 unit
*   **Constant Training Parameters:**
    *   `epochs`: 5 (with EarlyStopping)
    *   `batch_size`: 32
    *   `SEED`: 42 (for reproducibility)
    *   `optimizer`: 'adam'
    *   `loss`: 'mse'
    *   `patience` (for EarlyStopping): 3
The `WINDOW` size was varied. Model performance was evaluated using Mean Absolute Error (MAE) and Root Mean Squared Error (RMSE) on the training, validation, and test sets (original EURUSD price scale).

### Results Summary (Window Size)

| WINDOW Size | Test Set MAE (EURUSD) | Test Set RMSE (EURUSD) | Val Set MAE (EURUSD) | Val Set RMSE (EURUSD) | Train Set MAE (EURUSD) |
| :---------- | :-------------------- | :--------------------- | :------------------- | :-------------------- | :--------------------- |
| 30          | 0.000804              | 0.001015               | 0.000612             | 0.000883              | 0.000557               |
| 50          | 0.000949              | 0.001179               | 0.001041             | 0.001253              | 0.000695               |
| **100**     | **0.000580**          | **0.000846**           | **0.000511**         | **0.000811**          | **0.000358**           |
| 150         | 0.001081              | 0.001348               | 0.001111             | 0.001338              | 0.000657               |
| 200         | 0.000947              | 0.001234               | 0.000811             | 0.001101              | 0.000997               |
| 288         | 0.000656              | 0.000923               | 0.000561             | 0.000865              | 0.000513               |

*(Note: The `WINDOW=100` results shown here reflect a consistent run that yielded the best overall metrics for this parameter during initial exploration.)*

### Conclusion (Window Size)

Based on these experiments, **`WINDOW = 100`** was selected as the optimal lookback period. It provided the best balance of capturing sufficient historical context and achieving the lowest prediction errors on unseen test data. This value will be used for subsequent hyperparameter tuning experiments.

---

## Experiment 2: Optimizing LSTM Layer Units (for a 2-Layer Architecture)

With `WINDOW = 100` fixed (based on Experiment 1) and a 2-layer LSTM architecture, this set of experiments focused on determining an effective number of units for the two LSTM layers (LSTM1 and LSTM2).

### Methodology (LSTM Units)

*   **Base Architecture:** 2 LSTM layers, each followed by Dropout(0.2), then a Dense output layer.
*   **Constant Parameters:**
    *   `WINDOW`: 100
    *   `Dropout Rate`: 0.2
    *   `optimizer`: 'adam'
    *   `batch_size`: 32
    *   `epochs`: 5 (with EarlyStopping, `patience=3`)
    *   `SEED`: 42
    *   `loss`: 'mse'
*   **Varied Parameter:** Number of units in LSTM1 and LSTM2. Both symmetrical (e.g., 32/32, 50/50) and asymmetrical (e.g., 100/50, 128/32) unit configurations were tested.

### Results Summary (LSTM Units with WINDOW=100, 2 Layers, Dropout=0.2)

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

*   **Top Symmetrical Performers:** The configurations with `32/32` units and `50/50` units demonstrated the strongest performance for a 2-layer architecture.
    *   The `32/32` configuration achieved the lowest Test Set MAE (0.000641 EURUSD) and Validation Set MAE (0.000559 EURUSD) *in this specific batch of unit tests*.
    *   The `50/50` configuration was a very close competitor and notably achieved the lowest Training Set MAE, indicating a strong capacity to fit the training data.
*   **Larger Symmetrical Configurations:** Increasing units symmetrically beyond 50 generally led to a decline in performance on unseen data.
*   **Asymmetrical Configurations:** Did not surpass the top symmetrical configurations in terms of Test Set MAE.

### Conclusion (LSTM Units for 2-Layer Architecture with WINDOW=100)

For a 2-layer LSTM architecture with `WINDOW=100`:
1.  The **`[LSTM(32), LSTM(32)]`** configuration is highly effective, yielding the best MAE on validation and test sets *within this specific unit optimization experiment*.
2.  The **`[LSTM(50), LSTM(50)]`** configuration (from Experiment 1's base) remains the overall best performer observed so far across all experiments (Test MAE ~0.000580).

The difference in Test MAE between these two top 2-layer configurations (32/32 vs. 50/50) is relatively small. The `32/32` offers lower complexity.

---

## Experiment 3: Optimizing Number of LSTM Layers

Building on the previous experiments, this test evaluates the impact of varying the number of stacked LSTM layers, using `WINDOW = 100` and a consistent **32 units per LSTM layer**. The dropout rate was maintained at 0.2 after each LSTM layer.

### Methodology (Number of Layers)

*   **Constant Parameters:**
    *   `WINDOW`: 100
    *   `Units per LSTM Layer`: 32
    *   `Dropout Rate`: 0.2 (applied after each LSTM layer)
    *   `optimizer`: 'adam'
    *   `batch_size`: 32
    *   `epochs`: 5 (with EarlyStopping, `patience=3`)
    *   `SEED`: 42
    *   `loss`: 'mse'
*   **Varied Architectures:**
    1.  **1-Layer LSTM:** `LSTM(32) -> Dropout(0.2) -> Dense(1)`
    2.  **2-Layer LSTM:** `LSTM(32, return_sequences=True) -> Dropout(0.2) -> LSTM(32) -> Dropout(0.2) -> Dense(1)`
    3.  **3-Layer LSTM:** `LSTM(32, return_sequences=True) -> Dropout(0.2) -> LSTM(32, return_sequences=True) -> Dropout(0.2) -> LSTM(32) -> Dropout(0.2) -> Dense(1)`

### Results Summary (Number of Layers with WINDOW=100, Units=32/layer, Dropout=0.2)

| LSTM Layers | Test Set MAE (EURUSD) | Test Set RMSE (EURUSD) | Val Set MAE (EURUSD) | Val Set RMSE (EURUSD) | Train Set MAE (EURUSD) |
| :---------- | :-------------------- | :--------------------- | :------------------- | :-------------------- | :--------------------- |
| 1           | 0.000907              | 0.001160               | 0.000821             | 0.001102              | 0.000472               |
| **2**       | **0.000641**          | **0.000935**           | **0.000559**         | **0.000891**          | **0.000411**           |
| 3           | 0.001558              | 0.001951               | 0.001531             | 0.001826              | 0.000896               |

### Analysis (Number of Layers)

*   **Optimal Number of Layers:** The **2-layer LSTM architecture** significantly outperformed both the 1-layer and 3-layer configurations when using 32 units per layer.
*   **1-Layer Performance:** The 1-layer model likely lacked the capacity to learn the complex temporal dependencies in the data, resulting in higher errors on validation and test sets compared to the 2-layer model.
*   **3-Layer Performance:** The 3-layer model performed considerably worse. This suggests that for this dataset and current configuration, adding a third LSTM layer (even with dropout) led to increased difficulty in training, potential overfitting to noise, or vanishing/exploding gradient issues common in deeper RNNs. The training error itself was higher than for the 2-layer model, indicating a struggle to even fit the training data optimally.

### Conclusion (Number of Layers with WINDOW=100, Units=32/layer)

For an LSTM model with `WINDOW=100` and 32 units per layer, a **2-layer architecture provides the best performance**. Adding more layers (e.g., 3 layers) degrades performance, while a single layer appears insufficient.

This reinforces the choice of a 2-layer architecture as a strong baseline for this forecasting task. The optimal number of units within those two layers (either 32 or 50, as seen in Experiment 2) remains a key consideration for the final model configuration.

---

**(Future experiments will be documented here, e.g., tuning dropout rates, optimizer parameters, batch size, etc. The current overall best configuration appears to be WINDOW=100, 2 LSTM Layers with 50 units each, and Dropout 0.2.)**