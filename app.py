import gradio as gr
from prophet import Prophet
import pandas as pd
import numpy as np
import os
import json
from datetime import datetime, timedelta
from alpha_vantage.timeseries import TimeSeries # Alpha Vantage library

# --- Configuration ---
# Directory where your .json model files are (for hyperparameters)
MODEL_PARAMS_DIR = "./trained_models"
MODEL_PARAMS_PREFIX = "prophet_model_"
DATA_CACHE_FILE = "data_cache.json" # File to cache Alpha Vantage data

# Fetch Alpha Vantage API Key from Hugging Face Space Secrets
ALPHAVANTAGE_API_KEY = os.environ.get("ALPHAVANTAGE_API_KEY")

if not ALPHAVANTAGE_API_KEY:
    print("CRITICAL WARNING: ALPHAVANTAGE_API_KEY secret not found in Space settings!")
    # The app might still run but data fetching will fail.
    # Gradio UI can show an error message if this happens during data fetch.

# Default Prophet parameters (can be overridden by those in JSON files)
# Based on your training script
DEFAULT_PROPHET_PARAMS = {
    'yearly_seasonality': True,
    'weekly_seasonality': False, # You had this as False
    'daily_seasonality': False,  # You had this as False
    'changepoint_prior_scale': 0.05,
    'seasonality_prior_scale': 10.0,
    'growth': 'linear' # Common default
}

# --- Load Model Hyperparameters ---
# These JSONs now primarily serve to list available tickers and potentially
# override default hyperparameters if specific ones were saved.
model_hyperparams_catalogue = {}
print("Loading model hyperparameter configurations...")
if os.path.exists(MODEL_PARAMS_DIR):
    for filename in os.listdir(MODEL_PARAMS_DIR):
        if filename.startswith(MODEL_PARAMS_PREFIX) and filename.endswith(".json"):
            model_name_key = filename.replace(MODEL_PARAMS_PREFIX, "").replace(".json", "")
            file_path = os.path.join(MODEL_PARAMS_DIR, filename)
            try:
                with open(file_path, 'r') as f:
                    # The JSONs from model_to_json are full models.
                    # We primarily need the ticker name. Hyperparameters can be
                    # extracted if they are top-level, or we use defaults.
                    # For simplicity, we'll use DEFAULT_PROPHET_PARAMS for now
                    # but confirm they exist.
                    # json_data = json.load(f)
                    # specific_params = {
                    #     'yearly_seasonality': json_data.get('yearly_seasonality', DEFAULT_PROPHET_PARAMS['yearly_seasonality']),
                    #     # ... extract other relevant params ...
                    # }
                    model_hyperparams_catalogue[model_name_key] = DEFAULT_PROPHET_PARAMS.copy() # Use defaults
                print(f"Registered model config for: {model_name_key}")
            except Exception as e:
                print(f"Error reading or parsing model JSON {filename}: {e}")
else:
    print(f"WARNING: Model parameters directory '{MODEL_PARAMS_DIR}' not found.")

available_model_names = sorted(list(model_hyperparams_catalogue.keys()))
if not available_model_names:
    print("WARNING: No model configurations loaded. The application might not function correctly.")

# --- Data Fetching and Caching Logic ---
def load_data_cache():
    if os.path.exists(DATA_CACHE_FILE):
        try:
            with open(DATA_CACHE_FILE, 'r') as f:
                return json.load(f)
        except json.JSONDecodeError:
            print(f"Cache file {DATA_CACHE_FILE} is corrupted. Starting with an empty cache.")
            return {}
        except Exception as e:
            print(f"Error loading cache file {DATA_CACHE_FILE}: {e}. Starting with an empty cache.")
            return {}
    return {}

def save_data_cache(cache):
    try:
        with open(DATA_CACHE_FILE, 'w') as f:
            json.dump(cache, f, indent=4)
    except Exception as e:
        print(f"Error saving cache file {DATA_CACHE_FILE}: {e}")

def get_timeseries_data_from_alphavantage(ticker_symbol):
    """
    Fetches time series data from Alpha Vantage for a given ticker symbol.
    Returns a pandas DataFrame with 'ds' and 'y' columns, or None on error.
    """
    if not ALPHAVANTAGE_API_KEY:
        raise ValueError("Alpha Vantage API key is not configured.")

    print(f"Attempting to fetch new data for {ticker_symbol} from Alpha Vantage...")
    ts = TimeSeries(key=ALPHAVANTAGE_API_KEY, output_format='pandas')
    try:
        # Get daily adjusted prices for robustness (handles splits/dividends)
        # 'full' gives up to 20+ years of data. 'compact' gives last 100 days.
        # Prophet generally benefits from at least 1-2 years of data.
        data_av, meta_data = ts.get_daily_adjusted(symbol=ticker_symbol, outputsize='full')
        
        # Process Alpha Vantage DataFrame:
        # 1. Sort by date (Alpha Vantage usually returns newest first)
        data_av = data_av.sort_index(ascending=True)
        # 2. Rename date index to 'ds' and the chosen price column to 'y'
        #    '5. adjusted close' is generally preferred.
        df_prophet = data_av[['5. adjusted close']].reset_index()
        df_prophet.rename(columns={'date': 'ds', '5. adjusted close': 'y'}, inplace=True)
        # 3. Ensure 'ds' is datetime
        df_prophet['ds'] = pd.to_datetime(df_prophet['ds'])
        # 4. Ensure 'y' is numeric
        df_prophet['y'] = pd.to_numeric(df_prophet['y'], errors='coerce')
        df_prophet.dropna(subset=['y'], inplace=True) # Remove rows where y could not be coerced

        if df_prophet.empty:
            print(f"No valid data returned from Alpha Vantage for {ticker_symbol} after processing.")
            return None
            
        print(f"Successfully fetched and processed {len(df_prophet)} data points for {ticker_symbol}.")
        return df_prophet[['ds', 'y']]

    except Exception as e:
        print(f"Alpha Vantage API Error for {ticker_symbol}: {type(e).__name__} - {e}")
        # Common issues: Invalid API key, rate limit exceeded, ticker not found.
        if "Invalid API call" in str(e) or "does not exist" in str(e):
             print(f"Ticker {ticker_symbol} might not be valid or API call issue.")
        elif "rate limit" in str(e).lower():
             print("Alpha Vantage rate limit likely exceeded.")
        return None

def get_and_cache_data(ticker_symbol, min_history_days=730): # Need enough history for Prophet
    cache = load_data_cache()
    today_str = datetime.now().strftime("%Y-%m-%d")

    # Check cache first
    if ticker_symbol in cache and cache[ticker_symbol].get("date_fetched") == today_str:
        print(f"Using cached data for {ticker_symbol} from {today_str}")
        try:
            # Data stored as list of dicts; convert back to DataFrame
            df_data = pd.DataFrame(cache[ticker_symbol]["data"])
            df_data['ds'] = pd.to_datetime(df_data['ds'])
            return df_data
        except Exception as e:
            print(f"Error loading data from cache for {ticker_symbol}: {e}. Will try fetching.")
    
    # If not in cache or stale, fetch from Alpha Vantage
    df_new_data = get_timeseries_data_from_alphavantage(ticker_symbol)

    if df_new_data is not None and not df_new_data.empty:
        if len(df_new_data) < min_history_days / 2: # Arbitrary check for too little history
             print(f"WARNING: Fetched data for {ticker_symbol} is very short ({len(df_new_data)} days). Forecast quality may be poor.")

        # Update cache
        # Convert 'ds' to string for JSON serialization
        data_to_cache = df_new_data.copy()
        data_to_cache['ds'] = data_to_cache['ds'].dt.strftime('%Y-%m-%d')
        cache[ticker_symbol] = {
            "date_fetched": today_str,
            "data": data_to_cache.to_dict(orient='records')
        }
        save_data_cache(cache)
        return df_new_data # Return with 'ds' as datetime
    else:
        # If fetching failed, check if older cache data exists and return that with a warning
        if ticker_symbol in cache and "data" in cache[ticker_symbol]:
            print(f"Fetching new data failed for {ticker_symbol}. Using older cached data.")
            df_data = pd.DataFrame(cache[ticker_symbol]["data"])
            df_data['ds'] = pd.to_datetime(df_data['ds'])
            return df_data # This data might be stale
        print(f"Failed to fetch or find any cached data for {ticker_symbol}.")
        return None


# --- Main Prediction Function (Re-fitting Prophet on demand) ---
def predict_dynamic_forecast(ticker_selection, forecast_periods_str):
    status_message = ""
    if not ALPHAVANTAGE_API_KEY:
        return "ERROR: Alpha Vantage API Key not configured in Space Secrets.", None
    if not ticker_selection:
        return "Please select a ticker.", None
    
    try:
        forecast_periods = int(forecast_periods_str)
        if forecast_periods <= 0:
            return "Forecast periods must be a positive number.", None
    except ValueError:
        return "Invalid number for forecast periods.", None

    hyperparams = model_hyperparams_catalogue.get(ticker_selection)
    if not hyperparams: # Should not happen if dropdown is populated correctly
        return f"Configuration for '{ticker_selection}' not found.", None

    try:
        status_message += f"Fetching/loading data for {ticker_selection}...\n"
        # Prophet generally needs at least a year or two of data.
        # Alpha Vantage 'full' outputsize should provide this.
        historical_df = get_and_cache_data(ticker_selection, min_history_days=365 * 2)
        
        if historical_df is None or historical_df.empty:
            status_message += f"Failed to retrieve sufficient historical data for {ticker_selection}."
            return status_message, None
        
        if len(historical_df) < 30: # Prophet needs some minimal data
            status_message += f"Historical data for {ticker_selection} is too short ({len(historical_df)} points) to make a forecast."
            return status_message, None

        status_message += f"Data loaded. Preprocessing for Prophet (log transform 'y')...\n"
        fit_df = historical_df.copy()
        # IMPORTANT: Log transform 'y' as done during original training
        fit_df['y'] = np.log(fit_df['y'])
        # Handle potential -inf/inf/NaN from log(0) or log(negative_value)
        fit_df.replace([np.inf, -np.inf], np.nan, inplace=True)
        fit_df['y'] = fit_df['y'].ffill().bfill() # Forward fill then backward fill

        if fit_df['y'].isnull().any():
            status_message += f"NaNs remain in log-transformed 'y' for {ticker_selection} after fill. Cannot fit model."
            return status_message, None

        status_message += f"Fitting Prophet model for {ticker_selection} with latest data...\n"
        model = Prophet(**hyperparams)
        model.fit(fit_df[['ds', 'y']]) # Fit on 'ds' and log-transformed 'y'

        status_message += f"Generating forecast for {forecast_periods} periods...\n"
        future_df = model.make_future_dataframe(periods=forecast_periods, freq='D') # Daily frequency
        forecast_log_scale = model.predict(future_df)
        
        # IMPORTANT: Inverse transform (exponentiate) predictions
        output_df = forecast_log_scale[['ds']].copy()
        output_df['Predicted Price (yhat)'] = np.exp(forecast_log_scale['yhat'])
        output_df['Lower Bound (yhat_lower)'] = np.exp(forecast_log_scale['yhat_lower'])
        output_df['Upper Bound (yhat_upper)'] = np.exp(forecast_log_scale['yhat_upper'])
        
        # Return only the future forecast part
        final_forecast_df = output_df.tail(forecast_periods).reset_index(drop=True)
        
        status_message += "Forecast generated successfully."
        return status_message, final_forecast_df
    
    except Exception as e:
        error_full_message = f"ERROR during prediction for {ticker_selection}: {type(e).__name__} - {str(e)}"
        print(error_full_message) # Log to Hugging Face Space console for debugging
        # Also include parts of it in status_message for the user
        status_message += f"\nAn error occurred: {type(e).__name__}. Check Space logs for details."
        return status_message, None

# --- Gradio Interface Definition ---
with gr.Blocks(css="footer {visibility: hidden}", title="Stock/Commodity Forecaster") as iface:
    gr.Markdown("# Stock & Commodity Price Forecaster")
    gr.Markdown(
        "This tool fetches the latest market data using the Alpha Vantage API, "
        "re-fits a Prophet time series model on-the-fly using pre-defined hyperparameters, "
        "and generates a future price forecast."
        "\n\n**Note:** Forecasts are for informational purposes only and not financial advice. "
        "Data fetching may be slow on the first request for a ticker each day."
    )
    if not ALPHAVANTAGE_API_KEY:
        gr.Markdown("<h3 style='color:red;'>WARNING: Alpha Vantage API Key is not configured in Space Secrets. Data fetching will fail.</h3>")

    with gr.Row():
        with gr.Column(scale=1):
            ticker_dropdown = gr.Dropdown(
                choices=available_model_names,
                label="Select Ticker Symbol",
                info="Choose the stock/commodity to forecast."
            )
            periods_input = gr.Number(
                value=30, 
                label="Forecast Periods (Days)", 
                minimum=1, 
                maximum=365 * 2, # Max 2 years forecast
                step=1,
                info="Number of future days to predict."
            )
            predict_button = gr.Button("Generate Forecast", variant="primary")
        
        with gr.Column(scale=3):
            status_textbox = gr.Textbox(
                label="Process Status & Logs", 
                lines=6, 
                interactive=False,
                placeholder="Status messages will appear here..."
            )
            
    gr.Markdown("## Forecast Results")
    forecast_output_table = gr.DataFrame(
        label="Price Forecast Data",
        headers=['Date (ds)', 'Predicted Price (yhat)', 'Lower Bound', 'Upper Bound'] # Match output_df columns
    )

    predict_button.click(
        fn=predict_dynamic_forecast,
        inputs=[ticker_dropdown, periods_input],
        outputs=[status_textbox, forecast_output_table]
    )

    gr.Markdown("---")
    gr.Markdown(
        "**How it works:** Models are based on Facebook's Prophet. Hyperparameters are pre-set. "
        "Historical data is log-transformed before fitting. Predictions are exponentiated back. "
        "Data is cached daily to minimize Alpha Vantage API calls."
    )

# --- Launch the Gradio App ---
if __name__ == "__main__":
    iface.launch()