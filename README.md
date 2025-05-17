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
Data fetching may be slow on the first request for a ticker each day due to Alpha Vantage API calls.