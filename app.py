import gradio as gr
from prophet import Prophet
import pandas as pd
import numpy as np
import os
import json
from datetime import datetime, timedelta
from alpha_vantage.timeseries import TimeSeries # Alpha Vantage library
import traceback # For detailed error tracebacks

# --- Configuration ---
# Directory where your .json model files are (for hyperparameters)
MODEL_PARAMS_DIR = "./trained_models" # Assuming this is at the root of your Space
MODEL_PARAMS_PREFIX = "prophet_model_"
DATA_CACHE_FILE = "data_cache.json" # File to cache Alpha Vantage data

# Fetch Alpha Vantage API Key from Hugging Face Space Secrets
ALPHAVANTAGE_API_KEY = os.environ.get("ALPHAVANTAGE_API_KEY")

# This will be printed to the Space's container logs when the app starts
if not ALPHAVANTAGE_API_KEY:
    print("CRITICAL STARTUP WARNING: ALPHAVANTAGE_API_KEY secret not found in Space settings!")
else:
    print("STARTUP INFO: ALPHAVANTAGE_API_KEY found.")


# Default Prophet parameters (can be overridden by those in JSON files)
DEFAULT_PROPHET_PARAMS = {
    'yearly_seasonality': True,
    'weekly_seasonality': False,
    'daily_seasonality': False,
    'changepoint_prior_scale': 0.05,
    'seasonality_prior_scale': 10.0,
    'growth': 'linear'
}

# In app.py, near the top with other configurations

TICKER_TO_FULL_NAME = {
    "DBA": "Invesco DB Agriculture Fund (DBA)",
    "FCX": "Freeport-McMoRan Inc. (FCX)",
    "GLD": "SPDR Gold Shares (GLD)",
    "NEM": "Newmont Corporation (NEM)",
    "SLV": "iShares Silver Trust (SLV)",
    "UNG": "United States Natural Gas Fund (UNG)",
    "USO": "United States Oil Fund (USO)",
    "XOM": "Exxon Mobil Corporation (XOM)"
}

# --- Load Model Hyperparameters ---
model_hyperparams_catalogue = {}
# This will now store (display_name, ticker_symbol) for the dropdown
dropdown_choices = []

print("STARTUP INFO: Loading model hyperparameter configurations...")
if os.path.exists(MODEL_PARAMS_DIR):
    for filename in os.listdir(MODEL_PARAMS_DIR):
        if filename.startswith(MODEL_PARAMS_PREFIX) and filename.endswith(".json"):
            ticker_symbol = filename.replace(MODEL_PARAMS_PREFIX, "").replace(".json", "")
            
            # Store the actual hyperparameters with the ticker_symbol as the key
            model_hyperparams_catalogue[ticker_symbol] = DEFAULT_PROPHET_PARAMS.copy() # Or load from JSON if needed

            # Get the full name for display, default to ticker if not found
            display_name = TICKER_TO_FULL_NAME.get(ticker_symbol, ticker_symbol)
            dropdown_choices.append((display_name, ticker_symbol)) # (label, value) tuple
            
            print(f"STARTUP INFO: Registered model config for: {ticker_symbol} (Display: {display_name})")
else:
    print(f"STARTUP WARNING: Model parameters directory '{MODEL_PARAMS_DIR}' not found.")

# Sort dropdown choices by the display name (the first element of the tuple)
dropdown_choices = sorted(dropdown_choices, key=lambda x: x[0])

if not dropdown_choices:
    print("STARTUP WARNING: No model configurations loaded. Ticker dropdown will be empty.")
else:
    print(f"STARTUP INFO: Available models for dropdown: {dropdown_choices}")

# --- Data Fetching and Caching Logic ---
def load_data_cache():
    if os.path.exists(DATA_CACHE_FILE):
        try:
            with open(DATA_CACHE_FILE, 'r') as f:
                return json.load(f)
        except json.JSONDecodeError:
            print(f"CACHE WARNING: Cache file {DATA_CACHE_FILE} is corrupted. Starting with an empty cache.")
            return {}
        except Exception as e:
            print(f"CACHE ERROR: Error loading cache file {DATA_CACHE_FILE}: {e}. Starting with an empty cache.")
            return {}
    return {}

def save_data_cache(cache):
    try:
        with open(DATA_CACHE_FILE, 'w') as f:
            json.dump(cache, f, indent=4)
    except Exception as e:
        print(f"CACHE ERROR: Error saving cache file {DATA_CACHE_FILE}: {e}")

def get_timeseries_data_from_alphavantage(ticker_symbol):
    if not ALPHAVANTAGE_API_KEY:
        print("AV_FETCH ERROR: Alpha Vantage API key is not configured (checked within function).")
        raise ValueError("Alpha Vantage API key is not configured.")

    print(f"AV_FETCH INFO: Attempting to fetch UNADJUSTED daily data for {ticker_symbol} from Alpha Vantage using TIME_SERIES_DAILY...")
    ts = TimeSeries(key=ALPHAVANTAGE_API_KEY, output_format='pandas')
    try:
        # Use get_daily() for the TIME_SERIES_DAILY (unadjusted) endpoint
        data_av, meta_data = ts.get_daily(symbol=ticker_symbol, outputsize='full')
        
        # --- Process Alpha Vantage DataFrame (for TIME_SERIES_DAILY) ---
        # 1. Sort by date (Alpha Vantage usually returns newest first for this endpoint too)
        data_av = data_av.sort_index(ascending=True)
        
        # 2. Rename date index to 'ds' and the chosen price column to 'y'
        #    For TIME_SERIES_DAILY, the close column is typically '4. close'.
        close_column_name = '4. close' 
        if close_column_name not in data_av.columns:
            print(f"AV_FETCH WARNING: Column '{close_column_name}' not found in TIME_SERIES_DAILY data for {ticker_symbol}. Available columns: {data_av.columns.tolist()}")
            # You might want to try '5. adjusted close' if that's what your key somehow provides, or other fallbacks.
            # For now, let's assume this is critical.
            raise KeyError(f"Expected column '{close_column_name}' not found in TIME_SERIES_DAILY response for {ticker_symbol}.")

        df_prophet = data_av[[close_column_name]].reset_index()
        df_prophet.rename(columns={'date': 'ds', close_column_name: 'y'}, inplace=True)
        
        # 3. Ensure 'ds' is datetime
        df_prophet['ds'] = pd.to_datetime(df_prophet['ds'])
        
        # 4. Ensure 'y' is numeric
        df_prophet['y'] = pd.to_numeric(df_prophet['y'], errors='coerce')
        df_prophet.dropna(subset=['y'], inplace=True) # Remove rows where y could not be coerced

        if df_prophet.empty:
            print(f"AV_FETCH WARNING: No valid data returned from Alpha Vantage for {ticker_symbol} after processing TIME_SERIES_DAILY.")
            raise ValueError(f"Processed Alpha Vantage TIME_SERIES_DAILY data for {ticker_symbol} is empty.")
            
        print(f"AV_FETCH INFO: Successfully fetched and processed {len(df_prophet)} unadjusted data points for {ticker_symbol}.")
        return df_prophet[['ds', 'y']]

    except KeyError as ke:
        error_detail = f"KeyError accessing TIME_SERIES_DAILY Alpha Vantage data for {ticker_symbol}: {str(ke)}. Check column names."
        print(f"AV_FETCH ERROR: {error_detail}")
        raise Exception(error_detail) # Re-raise with more context
    except ValueError as ve: # Catch specific errors like empty processed data
        error_detail = f"ValueError processing TIME_SERIES_DAILY Alpha Vantage data for {ticker_symbol}: {str(ve)}."
        print(f"AV_FETCH ERROR: {error_detail}")
        raise Exception(error_detail) # Re-raise with more context
    except Exception as e: # General catch-all for other AV errors
        error_detail = f"Alpha Vantage API Error (TIME_SERIES_DAILY) for {ticker_symbol}: {type(e).__name__} - {str(e)}."
        print(f"AV_FETCH ERROR: {error_detail}")
        if "invalid api call" in str(e).lower() or "does not exist" in str(e).lower():
             print(f"AV_FETCH DETAIL: Ticker {ticker_symbol} might not be valid on Alpha Vantage or API call syntax issue.")
        elif "premium membership" in str(e).lower() or "rate limit" in str(e).lower():
             # This condition should be less likely now with TIME_SERIES_DAILY, but keep for safety
             print("AV_FETCH DETAIL: Alpha Vantage rate limit likely exceeded OR an unexpected premium issue with TIME_SERIES_DAILY.")
        raise Exception(error_detail) # Re-raise with more context


def get_and_cache_data(ticker_symbol, min_history_days=730):
    cache = load_data_cache()
    today_str = datetime.now().strftime("%Y-%m-%d")
    status_updates = [] # To collect messages for UI

    if ticker_symbol in cache and cache[ticker_symbol].get("date_fetched") == today_str:
        status_updates.append(f"Using cached data for {ticker_symbol} from {today_str}.")
        print(f"CACHE INFO: Using cached data for {ticker_symbol} from {today_str}")
        try:
            df_data = pd.DataFrame(cache[ticker_symbol]["data"])
            df_data['ds'] = pd.to_datetime(df_data['ds'])
            return df_data, "\n".join(status_updates)
        except Exception as e:
            status_updates.append(f"Error loading data from cache for {ticker_symbol}: {e}. Will try fetching.")
            print(f"CACHE ERROR: Error loading data from cache for {ticker_symbol}: {e}. Will try fetching.")
    
    status_updates.append(f"No fresh cache for {ticker_symbol}. Attempting to fetch from Alpha Vantage...")
    try:
        df_new_data = get_timeseries_data_from_alphavantage(ticker_symbol)
    except ValueError as ve: # Catch missing API key error specifically
        status_updates.append(f"Data Fetch ERROR: {str(ve)}")
        return None, "\n".join(status_updates)


    if df_new_data is not None and not df_new_data.empty:
        status_updates.append(f"Successfully fetched {len(df_new_data)} new data points for {ticker_symbol}.")
        if len(df_new_data) < min_history_days / 4: # Stricter check for very short history
             warning_msg = f"WARNING: Fetched data for {ticker_symbol} is very short ({len(df_new_data)} days). Forecast quality may be poor."
             status_updates.append(warning_msg)
             print(f"DATA_QUALITY WARNING: {warning_msg}")

        data_to_cache = df_new_data.copy()
        data_to_cache['ds'] = data_to_cache['ds'].dt.strftime('%Y-%m-%d')
        cache[ticker_symbol] = {
            "date_fetched": today_str,
            "data": data_to_cache.to_dict(orient='records')
        }
        save_data_cache(cache)
        return df_new_data, "\n".join(status_updates)
    else:
        status_updates.append(f"Failed to fetch new data for {ticker_symbol} from Alpha Vantage.")
        if ticker_symbol in cache and "data" in cache[ticker_symbol]:
            status_updates.append(f"Using older cached data for {ticker_symbol} as a fallback.")
            print(f"CACHE INFO: Using older cached data for {ticker_symbol} as fallback.")
            df_data = pd.DataFrame(cache[ticker_symbol]["data"])
            df_data['ds'] = pd.to_datetime(df_data['ds'])
            return df_data, "\n".join(status_updates)
        status_updates.append(f"No data available (neither new nor cached) for {ticker_symbol}.")
        print(f"DATA_ERROR: No data available for {ticker_symbol}.")
        return None, "\n".join(status_updates)

# --- Main Prediction Function ---
def predict_dynamic_forecast(ticker_selection, forecast_periods_str):
    status_message = ""
    empty_forecast_df = pd.DataFrame(columns=['Date (ds)', 'Predicted Price (yhat)', 'Lower Bound', 'Upper Bound'])

    if not ALPHAVANTAGE_API_KEY: # Check at the very beginning of the request
        print("PREDICT_ERROR: Alpha Vantage API Key not configured.")
        return "ERROR: Alpha Vantage API Key not configured in Space Secrets. Please check Space settings.", empty_forecast_df

    if not ticker_selection:
        return "Please select a ticker.", empty_forecast_df
    
    try:
        forecast_periods = int(forecast_periods_str)
        if forecast_periods <= 0:
            return "Forecast periods must be a positive number.", empty_forecast_df
    except ValueError:
        return "Invalid number for forecast periods (must be an integer).", empty_forecast_df

    hyperparams = model_hyperparams_catalogue.get(ticker_selection)
    if not hyperparams: 
        return f"Internal Error: Configuration for '{ticker_selection}' not found (though it should be in dropdown).", empty_forecast_df

    try:
        status_message += f"Initiating forecast for {ticker_selection} for {forecast_periods} days...\n"
        
        historical_df, data_fetch_status = get_and_cache_data(ticker_selection, min_history_days=365 * 1) # Min 1 year
        status_message += data_fetch_status + "\n"
        
        if historical_df is None or historical_df.empty:
            status_message += f"Cannot proceed: Failed to retrieve sufficient historical data for {ticker_selection}."
            print(f"PREDICT_ERROR: historical_df is None or empty for {ticker_selection}")
            return status_message, empty_forecast_df
        
        status_message += f"Data loaded ({len(historical_df)} rows). Preprocessing for Prophet (log transform 'y')...\n"
        print(f"PREDICT_INFO: Data loaded for {ticker_selection}, rows: {len(historical_df)}")

        # Prophet needs at least 2 data points, practically more.
        if len(historical_df) < 10: 
            status_message += f"Historical data for {ticker_selection} is too short ({len(historical_df)} points) to make a reliable forecast."
            print(f"PREDICT_ERROR: Data too short for {ticker_selection} ({len(historical_df)} points)")
            return status_message, empty_forecast_df

        fit_df = historical_df.copy()
        # Ensure 'y' is positive before log transform
        if (fit_df['y'] <= 0).any():
            status_message += "WARNING: Historical data contains zero or negative prices. These will be removed before log transformation. This might affect data quantity.\n"
            print(f"PREDICT_WARNING: Zero/negative prices found for {ticker_selection}. Filtering them out.")
            fit_df = fit_df[fit_df['y'] > 0]
            if len(fit_df) < 10:
                 status_message += f"After removing non-positive prices, data for {ticker_selection} is too short ({len(fit_df)} points)."
                 print(f"PREDICT_ERROR: Data too short after filtering non-positive prices for {ticker_selection}")
                 return status_message, empty_forecast_df
        
        fit_df['y'] = np.log(fit_df['y'])
        # No need to replace inf/nan if we filter y > 0 before log.
        # fit_df.replace([np.inf, -np.inf], np.nan, inplace=True) # Should not be needed if y > 0
        # fit_df['y'] = fit_df['y'].ffill().bfill() # Should not be needed if y > 0 and no other NaNs

        if fit_df['y'].isnull().any(): # Check if any NaNs created for other reasons
            status_message += f"NaNs present in log-transformed 'y' for {ticker_selection}. Cannot fit model."
            print(f"PREDICT_ERROR: NaNs in log-transformed y for {ticker_selection}")
            return status_message, empty_forecast_df

        status_message += f"Fitting Prophet model for {ticker_selection}...\n"
        print(f"PREDICT_INFO: Fitting Prophet for {ticker_selection}")
        model = Prophet(**hyperparams)
        model.fit(fit_df[['ds', 'y']]) 

        status_message += f"Generating forecast for {forecast_periods} periods...\n"
        print(f"PREDICT_INFO: Generating forecast for {ticker_selection}, periods: {forecast_periods}")
        future_df = model.make_future_dataframe(periods=forecast_periods, freq='D') 
        forecast_log_scale = model.predict(future_df)
        
        output_df = forecast_log_scale[['ds']].copy()
        output_df['Predicted Price (yhat)'] = np.exp(forecast_log_scale['yhat'])
        output_df['Lower Bound (yhat_lower)'] = np.exp(forecast_log_scale['yhat_lower'])
        output_df['Upper Bound (yhat_upper)'] = np.exp(forecast_log_scale['yhat_upper'])
        
        final_forecast_df = output_df.tail(forecast_periods).reset_index(drop=True)
        final_forecast_df['Date (ds)'] = final_forecast_df['ds'].dt.strftime('%Y-%m-%d') # Format date string
        final_forecast_df = final_forecast_df[['Date (ds)', 'Predicted Price (yhat)', 'Lower Bound (yhat_lower)', 'Upper Bound (yhat_upper)']]


        status_message += "Forecast generated successfully."
        print(f"PREDICT_INFO: Forecast successful for {ticker_selection}")
        return status_message, final_forecast_df
    
    except Exception as e:
        tb_str = traceback.format_exc()
        print(f"PREDICT_ERROR Unhandled exception for {ticker_selection}: {type(e).__name__} - {str(e)}\nTRACEBACK:\n{tb_str}")
        
        error_ui_message = (
            f"CRITICAL ERROR processing {ticker_selection}:\n"
            f"Type: {type(e).__name__}\n"
            f"Message: {str(e)}\n\n"
            f"--- Traceback (last few lines) ---\n"
        )
        # Add last few lines of traceback, ensuring not to overflow status box too much
        traceback_lines = tb_str.strip().splitlines()
        for line in traceback_lines[-7:]: # Show last 7 lines
            error_ui_message += line + "\n"
        
        status_message += f"\n{error_ui_message}"
        return status_message, empty_forecast_df

# --- Gradio Interface Definition ---
with gr.Blocks(css="footer {visibility: hidden}", title="Stock/Commodity Forecaster") as iface:
    gr.Markdown("# Stock & Commodity Price Forecaster")
    gr.Markdown(
        "This tool fetches the latest market data using the Alpha Vantage API, "
        "re-fits a Prophet time series model on-the-fly using pre-defined hyperparameters, "
        "and generates a future price forecast. [More details in the readme](https://huggingface.co/spaces/GuitarGeorge/Prophet-commodity-stock-analysis/blob/main/README.md)"
        "\n\n**Note:** Forecasts are for informational purposes only and not financial advice. "
        "Data fetching may be slow on the first request for a ticker each day. "
        "Alpha Vantage free tier has API call limits (5 calls/min, 100 calls/day)."
    )
    if not ALPHAVANTAGE_API_KEY: # Display warning in UI if key is missing at startup
        gr.Markdown("<h3 style='color:red;'>WARNING: Alpha Vantage API Key is not configured in Space Secrets. Data fetching will fail.</h3>")
    if not dropdown_choices:
        gr.Markdown("<h3 style='color:red;'>WARNING: No model configurations loaded. Ticker selection will be empty. Check 'trained_models' folder and filenames.</h3>")


    with gr.Row():
        with gr.Column(scale=1):
            ticker_dropdown = gr.Dropdown(
                choices=dropdown_choices, # Use the list of (label, value) tuples
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
                lines=15, # Increased lines for more detailed logs/errors
                interactive=False,
                placeholder="Status messages will appear here..."
            )
            
    gr.Markdown("## Forecast Results")
    forecast_output_table = gr.DataFrame(
        label="Price Forecast Data"
        # Headers will be inferred from the DataFrame column names now
    )

    predict_button.click(
        fn=predict_dynamic_forecast,
        inputs=[ticker_dropdown, periods_input],
        outputs=[status_textbox, forecast_output_table]
    )

    gr.Markdown("---")
    gr.Markdown(
        "**How it works:** Models are based on Facebook's Prophet. Hyperparameters are pre-set. "
        "Historical data for the selected ticker is fetched from Alpha Vantage, log-transformed, and used to fit the model. "
        "Predictions are then exponentiated back to the original price scale. "
        "Fetched data is cached daily in the Space's temporary storage to minimize Alpha Vantage API calls and speed up subsequent requests for the same ticker on the same day."
    )

# --- Launch the Gradio App ---
if __name__ == "__main__":
    print("STARTUP INFO: Launching Gradio interface...")
    iface.launch() # In Hugging Face Spaces, this is all you need.