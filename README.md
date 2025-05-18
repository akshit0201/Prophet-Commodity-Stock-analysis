---
title: Prophet Commodity & Stock Forecaster
emoji: 📈📉
colorFrom: blue
colorTo: green
sdk: gradio
sdk_version: "5.29.1"
app_file: app.py
pinned: false
---

# Prophet Commodity & Stock Price Forecaster

This application fetches the latest market data using the Alpha Vantage API, 
re-fits a Prophet time series model on-the-fly using pre-defined hyperparameters, 
and generates a future price forecast.

**Features:**
- Select from various commodity and stock tickers.
- Specify the number of days to forecast.
- View process logs and forecast data.

**Note:** Forecasts are for informational purposes only and not financial advice. 
Data fetching may be slow on the first request for a ticker each day due to Alpha Vantage API calls. Alpha Vantage free tier has API call limits (5 calls/min, 100 calls/day).

---

## Methodology & Model Details

This section outlines the approach used for generating forecasts.

### 1. Training Process (On-Demand Re-fitting)

For each forecast request:

1.  **Data Sourcing:** Historical daily price data (unadjusted close) is fetched from Alpha Vantage using the `TIME_SERIES_DAILY` endpoint (`outputsize='full'`). This provides an extensive history for the selected ticker.
2.  **Data Caching:** To optimize performance and respect API limits, fetched data is cached daily within the Hugging Face Space's temporary storage. Subsequent requests for the same ticker on the same day will use this cache.
3.  **Preprocessing:** The target variable ('y', representing the closing price) is **log-transformed** (`np.log(y)`) to stabilize variance and help the model better capture multiplicative effects. Non-positive price values are filtered out before this transformation.
4.  **Model Fitting:** A new Prophet model instance is created using a default set of robust hyperparameters (see Model Architecture below). This model is then fit on the *entire available historical dataset* for the selected ticker.
5.  **Forecasting:** The newly fitted model generates future dates for the user-specified forecast horizon.
6.  **Output Postprocessing:** The point forecast (`yhat`) and uncertainty intervals (`yhat_lower`, `yhat_upper`) are **exponentiated** (`np.exp()`) to bring them back to the original price scale for display.

### 2. Model Architecture (Prophet Configuration)

The core forecasting engine is [Prophet](https://facebook.github.io/prophet/), a procedure for forecasting time series data developed by Facebook (now Meta). The following default hyperparameters are used for model instantiation during each on-demand fit:

*   `growth`: 'linear'
*   `yearly_seasonality`: True
*   `weekly_seasonality`: False
*   `daily_seasonality`: False
*   `changepoint_prior_scale`: 0.05 (controls the flexibility of the trend by adjusting the influence of changepoints)
*   `seasonality_prior_scale`: 10.0 (controls the strength of the seasonality components)

No additional regressors or custom holidays are used in the current version of the deployed application.

### 3. Backtesting Results

The modeling approach was evaluated using a backtesting strategy on historical data. The models were trained on an initial portion of the data, and then forecasts were made for a subsequent hold-out period. Performance was measured using standard regression metrics.

**Summary Metrics:**

The following table summarizes the performance metrics observed for each ticker on a representative test set (e.g., forecasting 1 year ahead after training on approximately 5-6 years of data, with the test set being the year following the training period). *Note: These specific metrics are from the provided `prophet_evaluation_summary.csv` and pertain to a specific historical backtest run.*

| Ticker | MAE        | RMSE       | MAPE (%)  | R2 Score    |
|--------|------------|------------|-----------|-------------|
| USO    | 8.2238909  | 9.71955902 | 10.92745477 | -4.067935694|
| UNG    | 13.02581342| 14.11032288| 80.21660263 | -8.997556366|
| GLD    | 23.95825101| 30.21382976| 9.611583257 | 0.1389319607|
| SLV    | 2.08769802 | 2.640192904| 8.04601763  | 0.2844266462|
| FCX    | 9.501484141| 12.5243823 | 24.47343386 | -4.361008267|
| DBA    | 1.776854996| 2.136406692| 6.88682417  | -0.09878885025|
| XOM    | 12.38567233| 15.71312   | 11.55524469 | -4.466316231|
| NEM    | 8.447357676| 10.52025772| 18.47559543 | -1.681251746|

**Interpreting Metrics:**
*   **MAE (Mean Absolute Error):** Average absolute difference between predicted and actual values, in the same units as the price.
*   **RMSE (Root Mean Squared Error):** Square root of the average of squared differences. Also in price units, penalizes larger errors more.
*   **MAPE (Mean Absolute Percentage Error):** Average absolute percentage difference. Useful for comparing across series with different price scales. High MAPE (like for UNG) can indicate challenging forecastability or periods of high volatility.
*   **R2 Score:** Coefficient of determination. Values closer to 1 indicate a better fit. Negative R2 scores indicate the model performs worse than a simple horizontal line (mean of the data). Many of these negative R2 scores suggest that for these specific long-term hold-out sets, Prophet (with these hyperparameters) struggled to outperform a naive baseline, which is common for volatile financial time series over long horizons without incorporating external factors or more complex modeling.

**Example Forecast vs. Actuals Plots from Backtesting:**

*(These plots illustrate how the model's forecast (blue line) and confidence interval (shaded area) compared against the actual test data (black dots) during a historical backtest run.)*

**DBA - Test Forecast vs. Actuals**
![DBA Test Forecast](images/DBA_test_forecast_vs_actual.png)

**FCX - Test Forecast vs. Actuals**
![FCX Test Forecast](images/FCX_test_forecast_vs_actual.png)

**GLD - Test Forecast vs. Actuals**
![GLD Test Forecast](images/GLD_test_forecast_vs_actual.png)

**NEM - Test Forecast vs. Actuals**
![NEM Test Forecast](images/NEM_test_forecast_vs_actual.png)

**SLV - Test Forecast vs. Actuals**
![SLV Test Forecast](images/SLV_test_forecast_vs_actual.png)

**UNG - Test Forecast vs. Actuals**
![UNG Test Forecast](images/UNG_test_forecast_vs_actual.png)

**USO - Test Forecast vs. Actuals**
![USO Test Forecast](images/USO_test_forecast_vs_actual.png)

![XOM Test Forecast](https://drive.google.com/uc?export=view&id=112aylZnv3NKTUQWz5HrU2hsRwaNSymMg)

**XOM - Test Forecast vs. Actuals**
![XOM Test Forecast](images/XOM_test_forecast_vs_actual.png)

*(Image for XOM provided shows a wide confidence interval, indicating high uncertainty in the long-term forecast for this particular backtest period).*


### 4. Deployment Architecture

The application is deployed as follows:

*   **User Interface:** A [Gradio](https://gradio.app) web application provides the interactive components.
*   **Hosting:** The Gradio app is hosted on [Hugging Face Spaces](https://huggingface.co/spaces).
*   **Backend Logic:** A Python script (`app.py`) running on the Space handles:
    *   User input processing.
    *   API calls to Alpha Vantage for data.
    *   Data caching.
    *   Prophet model instantiation, fitting, and prediction.
    *   Formatting results for display.
*   **Continuous Deployment:** Code changes pushed to the `main` branch of the [GitHub repository](https://github.com/akshit0201/Prophet-Commodity-Stock-analysis) are automatically synced and deployed to the Hugging Face Space via GitHub Actions.
*   **API Key Management:** The Alpha Vantage API key is stored as a secure Secret within the Hugging Face Space settings and accessed as an environment variable by the application.

---

## How to Use the Application

1.  Navigate to the application on [Hugging Face Spaces](https://huggingface.co/spaces/GuitarGeorge/Prophet-commodity-stock-analysis).
2.  Select a ticker symbol from the "Select Ticker Symbol" dropdown menu (e.g., "SPDR Gold Shares (GLD)").
3.  Enter the desired number of future days to forecast in the "Forecast Periods (Days)" field (e.g., 30, 90, 365).
4.  Click the "Generate Forecast" button.
5.  Observe the "Process Status & Logs" textbox for updates on data fetching and model fitting.
6.  The "Forecast Results" table will populate with the predicted dates, prices, and uncertainty bounds.

---

## Limitations and Considerations

*   **Model Simplicity:** Forecasts are based solely on historical price data patterns (trend and yearly seasonality) as identified by Prophet. They do not account for external news, geopolitical events, fundamental company analysis, market sentiment, or sudden shocks.
*   **API Limits:** The application uses the free tier of the Alpha Vantage API, which has usage limits (typically 5 calls per minute and 100 calls per day). If these limits are exceeded (e.g., by many users accessing the app), data fetching may temporarily fail.
*   **Forecast Accuracy:** While Prophet is a robust forecasting tool, the accuracy of its predictions can vary significantly depending on the ticker, market volatility, and the length of the forecast horizon. Forecasts further into the future inherently have higher uncertainty. The provided backtesting metrics indicate areas where the model (with current default hyperparameters) may struggle.
*   **Not Financial Advice:** This tool and its outputs are provided for educational, informational, and illustrative purposes only. **It does NOT constitute financial advice, investment recommendations, or an endorsement of any particular trading strategy.** Always conduct your own thorough research and consult with a qualified financial advisor before making any investment decisions.