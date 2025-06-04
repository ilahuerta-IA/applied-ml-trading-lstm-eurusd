# Applied Machine Learning for Trading: EURUSD Forecasting with LSTM

## Short Description
This project develops and optimizes a **Long Short-Term Memory (LSTM) neural network** to forecast EURUSD 5-minute closing prices. The goal is to create a robust predictive "tool" that can generate signals to confirm or filter entries for an algorithmic trading strategy. This repository documents the systematic experimentation and final configuration of this LSTM model.

## Project Objective
The primary objective is to build and evaluate a **simple yet robust LSTM model** for short-term currency exchange rate prediction. This model, referred to as "LSTM Tool V1.0," is designed to produce a reliable next-bar price prediction, which can then be integrated as a **confirmation or filtering signal** into a separate, rule-based trading strategy. This project serves as a practical exercise in applying deep learning techniques to financial time series data and understanding the nuances involved in developing a machine learning component for trading.

## Reference
This project draws inspiration and context from:
*   **Title:** MACHINE LEARNING APLICADO AL TRADING
*   **Author:** Macarena Salvador Maceira
*   **Institution:** Facultad Ciencias Económicas y Empresariales, Universidad Pontificia Comillas
*   **Date:** Junio 2019
    *Note: This project is an independent implementation and exploration, not an official reproduction or extension of the cited work unless explicitly stated.*

## Dataset
*   **Asset:** EURUSD (Euro / US Dollar)
*   **Frequency:** 5-minute intervals
*   **Period:** 10 years (for optimal model training)
*   **Source:** `EURUSD_5m_10Yea.csv` (Included in the repository)
*   **Columns used (for LSTM Tool V1.0):** `Date`, `Time`, `Close` (The model uses only 'Close' price as its input feature.)

## Features
*   Data loading and preprocessing (Pandas).
*   Time series windowing for LSTM input.
*   MinMax scaling of data (Scikit-learn).
*   LSTM model construction using TensorFlow/Keras (Sequential API with explicit Input layer, LSTM, Dropout, Dense layers).
*   Model training with EarlyStopping callback.
*   Model evaluation using Mean Squared Error (MSE), Root Mean Squared Error (RMSE), and Mean Absolute Error (MAE).
*   Visualization of actual vs. predicted prices on original and scaled data (Matplotlib).
*   **Export of Trained Model:** Saving the final LSTM model and its scaler for external use in trading strategies.

## Technologies Used
*   Python 3.x
*   TensorFlow & Keras
*   Pandas
*   NumPy
*   Scikit-learn
*   Matplotlib
*   Joblib (for saving scaler)

## Setup and Installation
1.  Clone the repository:
    ```bash
    git clone https://github.com/ilahuerta-IA/applied-ml-trading-lstm-eurusd.git
    cd applied-ml-trading-lstm-eurusd
    ```
2.  (Recommended) Create a virtual environment:
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    ```
3.  Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```
    *(You'll need a `requirements.txt` file. You can generate one using `pip freeze > requirements.txt` in your activated environment after installing all necessary packages locally, or list them manually based on your Colab imports).*
    **Minimum `requirements.txt` content (for LSTM Tool V1.0):**
    ```
    tensorflow
    pandas
    numpy
    scikit-learn
    matplotlib
    joblib
    ```

## Usage
The primary code for developing and training the LSTM tool is within the Jupyter Notebook (`Machine_Learning_Applied_to_Trading.ipynb`).
1.  Ensure the dataset (`EURUSD_5m_10Yea.csv`) is in the same directory as the notebook.
2.  Open and run all cells in the Jupyter Notebook sequentially.
    *   You can run this in Google Colab by uploading the notebook and the CSV file.
3.  Upon successful execution, the trained LSTM model (`lstm_eurusd_5m_close_only_v1.h5`) and its corresponding `MinMaxScaler` (`scaler_eurusd_5m_close_only_v1.pkl`) will be saved in the `Models/` directory, ready for integration into a trading strategy.

## Model Architecture (LSTM Tool V1.0)
The final **LSTM Tool V1.0** model configuration is the result of extensive hyperparameter optimization (documented in `EXPERIMENTS.md`). It aims for a balance of simplicity, accuracy, and robustness.

*   **Input Features:** `['Close']` (Only 'Close' price as input to the LSTM)
*   **Lookback Window (WINDOW):** 30 (The LSTM uses the last 30 5-minute bars to make a prediction)
*   **Layers:** 2 LSTM layers
    *   Explicit Input Layer with shape `(30, 1)`
    *   LSTM Layer 1: 32 units, `return_sequences=True`
    *   Dropout Layer 1: 0.1
    *   LSTM Layer 2: 32 units
    *   Dropout Layer 2: 0.1
    *   Dense Output Layer: 1 unit (predicting the next 5-minute 'Close' price)
*   **Optimizer:** Adam (default learning rate ~0.001)
*   **Loss Function:** Mean Squared Error (MSE)
*   **Training Schedule:** Max epochs=20, EarlyStopping patience=10

## Example Results (LSTM Tool V1.0 Performance)
This section showcases the **fairly evaluated performance** of the **LSTM Tool V1.0** model ('Close' only input) on the EURUSD 5-minute data (10-year historical dataset). This model achieves strong generalization performance.

| Model Configuration                   | Test Set MAE (EURUSD) | Test Set RMSE (EURUSD) | Val Set MAE (EURUSD) | Val Set RMSE (EURUSD) | Train Set MAE (EURUSD) | Min `val_loss` (Scaled) <br> (at Epoch) | Early Stopping Epoch |
| :------------------------------------ | :-------------------- | :--------------------- | :------------------- | :-------------------- | :--------------------- | :------------------------------------ | :------------------- |
| **LSTM Tool V1.0 (Close only)**       | **0.000237**          | **0.000387**           | **0.000213**         | **0.000299**          | **0.000196**           | **7.7975e-06 (Ep 16)**                | 20                   |

*Note: The Test Set MAE of 0.000237 EURUSD is approximately 2.37 pips on a 5-minute timeframe. This indicates a high level of predictive accuracy for a financial time series model.*

## Exported Model Files (LSTM Tool V1.0)
The trained model and its scaler, ready for external integration, are located in the `Models/` directory:
*   `Models/lstm_eurusd_5m_close_only_v1.h5`
*   `Models/scaler_eurusd_5m_close_only_v1.pkl`

## Future Work: Strategy Integration & LSTM Filtering Concepts

The LSTM Tool V1.0 developed in this repository provides a robust predictive signal. The next critical phase involves **integrating this signal into an algorithmic trading strategy**. This process is documented in a separate repository:

**[Trading Strategy Optimizer Repository](https://github.com/ilahuerta-IA/trading-strategy-optimizer.git)**

Directly using the raw LSTM prediction can be risky due to inherent noise and variability. Therefore, various **filtering concepts** will be employed to enhance the robustness and reliability of trading signals derived from the LSTM's output. These filtering strategies will be implemented and tested within the aforementioned Backtrader repository.

Detailed concepts for filtering the LSTM's output can be found in the [LSTM_Filtering_Concepts.md](Docs/LSTM_Filtering_Concepts.md) document within this repository. This includes:
*   Magnitude-Based Filters
*   Price Level & Range Filters
*   Consensus & Smoothing Filters
*   Uncertainty/Confidence-Based Filters

This iterative approach of building the LSTM tool, integrating it into a strategy with filters, and then backtesting rigorously is key to developing a performant and resilient trading system.

## Contributing
Contributions, issues, and feature requests are welcome. Please feel free to fork the repository, make changes, and open a pull request.

## License
This project is licensed under the MIT License - see the `LICENSE` file for details.