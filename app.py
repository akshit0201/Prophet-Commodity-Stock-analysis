import gradio as gr
from prophet import Prophet
import pandas as pd
import numpy as np
import os
import json
from datetime import datetime, timedelta
import yfinance as yf
import traceback
import threading
from concurrent.futures import ThreadPoolExecutor

# --- Configuration & Threading Locks ---
MODEL_PARAMS_DIR = "./trained_models" 
MODEL_PARAMS_PREFIX = "prophet_model_"
DATA_CACHE_FILE = "data_cache.json" 

# Reentrant lock to safely read/write daily cache during concurrent thread executions
CACHE_LOCK = threading.RLock()

print("STARTUP INFO: Running with yfinance integration. No API key needed.")

# Default Prophet parameters (can be overridden by those in JSON files)
DEFAULT_PROPHET_PARAMS = {
    'yearly_seasonality': True,
    'weekly_seasonality': False,
    'daily_seasonality': False,
    'changepoint_prior_scale': 0.05,
    'seasonality_prior_scale': 10.0,
    'growth': 'linear'
}

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
dropdown_choices = []

print("STARTUP INFO: Loading model hyperparameter configurations...")
if os.path.exists(MODEL_PARAMS_DIR):
    for filename in os.listdir(MODEL_PARAMS_DIR):
        if filename.startswith(MODEL_PARAMS_PREFIX) and filename.endswith(".json"):
            ticker_symbol = filename.replace(MODEL_PARAMS_PREFIX, "").replace(".json", "")
            
            # Store default or configuration parameters
            model_hyperparams_catalogue[ticker_symbol] = DEFAULT_PROPHET_PARAMS.copy() 

            display_name = TICKER_TO_FULL_NAME.get(ticker_symbol, ticker_symbol)
            dropdown_choices.append((display_name, ticker_symbol)) 
            
            print(f"STARTUP INFO: Registered model config for: {ticker_symbol} (Display: {display_name})")
else:
    print(f"STARTUP WARNING: Model parameters directory '{MODEL_PARAMS_DIR}' not found.")

# Sort individual dropdown choices alphabetically
dropdown_choices = sorted(dropdown_choices, key=lambda x: x[0])

# Add the 'Select All' option at the very top of the list
dropdown_choices.insert(0, ("All Tickers (Parallel Inference)", "ALL"))

if len(dropdown_choices) <= 1:
    print("STARTUP WARNING: No model configurations loaded. Ticker dropdown will only contain default option.")
else:
    print(f"STARTUP INFO: Available models for dropdown: {dropdown_choices}")

# --- Thread-Safe Data Fetching and Caching Logic ---
def load_data_cache():
    if os.path.exists(DATA_CACHE_FILE):
        try:
            with open(DATA_CACHE_FILE, 'r') as f:
                return json.load(f)
        except json.JSONDecodeError:
            print(f"CACHE WARNING: Cache file {DATA_CACHE_FILE} is corrupted. Starting empty.")
            return {}
        except Exception as e:
            print(f"CACHE ERROR: Error loading cache file {DATA_CACHE_FILE}: {e}")
            return {}
    return {}

def save_data_cache(cache):
    try:
        with open(DATA_CACHE_FILE, 'w') as f:
            json.dump(cache, f, indent=4)
    except Exception as e:
        print(f"CACHE ERROR: Error saving cache file {DATA_CACHE_FILE}: {e}")

def get_timeseries_data_from_yfinance(ticker_symbol):
    print(f"YF_FETCH INFO: Fetching daily history for {ticker_symbol} from Yahoo Finance...")
    try:
        ticker = yf.Ticker(ticker_symbol)
        data_yf = ticker.history(period="max")
        
        if data_yf.empty:
            print(f"YF_FETCH WARNING: No data returned from yfinance for {ticker_symbol}.")
            raise ValueError(f"No data found on Yahoo Finance for ticker {ticker_symbol}.")
            
        data_yf = data_yf.sort_index(ascending=True)
        df_prophet = data_yf[['Close']].reset_index()
        
        # Safe column rename handling differences in default index naming ('Date' vs 'Datetime')
        df_prophet.rename(columns={df_prophet.columns[0]: 'ds', 'Close': 'y'}, inplace=True)
        
        # Remove timezones to prevent serialization and model compatibility issues
        df_prophet['ds'] = pd.to_datetime(df_prophet['ds']).dt.tz_localize(None)
        df_prophet['y'] = pd.to_numeric(df_prophet['y'], errors='coerce')
        df_prophet.dropna(subset=['y'], inplace=True) 

        if df_prophet.empty:
            print(f"YF_FETCH WARNING: No valid data returned from yfinance for {ticker_symbol} after processing.")
            raise ValueError(f"Processed Yahoo Finance data for {ticker_symbol} is empty.")
            
        print(f"YF_FETCH INFO: Successfully fetched {len(df_prophet)} data points for {ticker_symbol}.")
        return df_prophet[['ds', 'y']]

    except Exception as e:
        error_detail = f"Yahoo Finance API Error for {ticker_symbol}: {type(e).__name__} - {str(e)}."
        print(f"YF_FETCH ERROR: {error_detail}")
        raise Exception(error_detail)

def get_and_cache_data(ticker_symbol, min_history_days=730):
    today_str = datetime.now().strftime("%Y-%m-%d")
    status_updates = [] 

    # 1. Thread-safe read check
    with CACHE_LOCK:
        cache = load_data_cache()
        if ticker_symbol in cache and cache[ticker_symbol].get("date_fetched") == today_str:
            status_updates.append(f"Using cached data for {ticker_symbol} from {today_str}.")
            print(f"CACHE INFO: Using cached data for {ticker_symbol} from {today_str}")
            try:
                df_data = pd.DataFrame(cache[ticker_symbol]["data"])
                df_data['ds'] = pd.to_datetime(df_data['ds'])
                return df_data, "\n".join(status_updates)
            except Exception as e:
                status_updates.append(f"Error loading from cache: {e}. Attempting download.")
                print(f"CACHE ERROR: Error loading cached data for {ticker_symbol}: {e}")
    
    # 2. Network IO executed outside the lock to allow full parallel throughput
    status_updates.append(f"No fresh cache for {ticker_symbol}. Fetching from Yahoo Finance...")
    try:
        df_new_data = get_timeseries_data_from_yfinance(ticker_symbol)
    except Exception as e: 
        status_updates.append(f"Data Fetch ERROR: {str(e)}")
        # Thread-safe fallback to stale cache data if download fails
        with CACHE_LOCK:
            cache = load_data_cache()
            if ticker_symbol in cache and "data" in cache[ticker_symbol]:
                status_updates.append(f"Using older cached data for {ticker_symbol} as a fallback.")
                df_data = pd.DataFrame(cache[ticker_symbol]["data"])
                df_data['ds'] = pd.to_datetime(df_data['ds'])
                return df_data, "\n".join(status_updates)
        return None, "\n".join(status_updates)

    # 3. Thread-safe cache write back
    if df_new_data is not None and not df_new_data.empty:
        status_updates.append(f"Successfully fetched {len(df_new_data)} data points.")
        if len(df_new_data) < min_history_days / 4: 
             warning_msg = f"WARNING: Fetched data for {ticker_symbol} is short. Forecast quality may be poor."
             status_updates.append(warning_msg)

        data_to_cache = df_new_data.copy()
        data_to_cache['ds'] = data_to_cache['ds'].dt.strftime('%Y-%m-%d')
        
        with CACHE_LOCK:
            cache = load_data_cache()
            cache[ticker_symbol] = {
                "date_fetched": today_str,
                "data": data_to_cache.to_dict(orient='records')
            }
            save_data_cache(cache)
            
        return df_new_data, "\n".join(status_updates)
    else:
        status_updates.append(f"Failed to fetch new data for {ticker_symbol}.")
        with CACHE_LOCK:
            cache = load_data_cache()
            if ticker_symbol in cache and "data" in cache[ticker_symbol]:
                status_updates.append(f"Using older cached data as a fallback.")
                df_data = pd.DataFrame(cache[ticker_symbol]["data"])
                df_data['ds'] = pd.to_datetime(df_data['ds'])
                return df_data, "\n".join(status_updates)
        return None, "\n".join(status_updates)

# --- Thread Task Execution for Single Tickers ---
def run_single_inference(ticker_symbol, forecast_periods, hyperparams):
    status_logs = []
    try:
        status_logs.append(f"[{ticker_symbol}] Starting processing pipeline...")
        historical_df, data_fetch_status = get_and_cache_data(ticker_symbol, min_history_days=365)
        status_logs.append(data_fetch_status)
        
        if historical_df is None or historical_df.empty:
            return {
                "ticker": ticker_symbol,
                "success": False,
                "logs": "\n".join(status_logs) + f"\n[{ticker_symbol}] Error: Historical data not available.",
                "df": None, "current_price": None, "predicted_price": None, "expected_return": None
            }
        
        # Retrieve the latest actual close price
        current_price = float(historical_df['y'].iloc[-1])
        
        if len(historical_df) < 10:
            return {
                "ticker": ticker_symbol,
                "success": False,
                "logs": "\n".join(status_logs) + f"\n[{ticker_symbol}] Error: Dataset too short to fit.",
                "df": None, "current_price": current_price, "predicted_price": None, "expected_return": None
            }
            
        fit_df = historical_df.copy()
        if (fit_df['y'] <= 0).any():
            status_logs.append(f"[{ticker_symbol}] Removing non-positive values prior to log transform.")
            fit_df = fit_df[fit_df['y'] > 0]
            if len(fit_df) < 10:
                return {
                    "ticker": ticker_symbol,
                    "success": False,
                    "logs": "\n".join(status_logs) + f"\n[{ticker_symbol}] Error: Dataset too short after filtering.",
                    "df": None, "current_price": current_price, "predicted_price": None, "expected_return": None
                }
        
        fit_df['y'] = np.log(fit_df['y'])
        if fit_df['y'].isnull().any():
            return {
                "ticker": ticker_symbol,
                "success": False,
                "logs": "\n".join(status_logs) + f"\n[{ticker_symbol}] Error: Log transform generated null values.",
                "df": None, "current_price": current_price, "predicted_price": None, "expected_return": None
            }
            
        # Fit Prophet model
        model = Prophet(**hyperparams)
        model.fit(fit_df[['ds', 'y']])
        
        future_df = model.make_future_dataframe(periods=forecast_periods, freq='D')
        forecast_log_scale = model.predict(future_df)
        
        output_df = forecast_log_scale[['ds']].copy()
        output_df['Predicted Price (yhat)'] = np.exp(forecast_log_scale['yhat'])
        output_df['Lower Bound (yhat_lower)'] = np.exp(forecast_log_scale['yhat_lower'])
        output_df['Upper Bound (yhat_upper)'] = np.exp(forecast_log_scale['yhat_upper'])
        
        final_forecast_df = output_df.tail(forecast_periods).reset_index(drop=True)
        final_forecast_df['Date (ds)'] = final_forecast_df['ds'].dt.strftime('%Y-%m-%d')
        final_forecast_df = final_forecast_df[['Date (ds)', 'Predicted Price (yhat)', 'Lower Bound (yhat_lower)', 'Upper Bound (yhat_upper)']]
        
        # Calculate final forecasted value and percentage change
        predicted_price = float(final_forecast_df['Predicted Price (yhat)'].iloc[-1])
        expected_return = ((predicted_price - current_price) / current_price) * 100
        
        status_logs.append(f"[{ticker_symbol}] Forecast generated. Current Close: {current_price:.2f}, Final Predicted: {predicted_price:.2f}, Expected Return: {expected_return:+.2f}%")
        
        return {
            "ticker": ticker_symbol,
            "success": True,
            "logs": "\n".join(status_logs),
            "df": final_forecast_df,
            "current_price": current_price,
            "predicted_price": predicted_price,
            "expected_return": expected_return,
            "lower_bound": float(final_forecast_df['Lower Bound (yhat_lower)'].iloc[-1]),
            "upper_bound": float(final_forecast_df['Upper Bound (yhat_upper)'].iloc[-1])
        }
        
    except Exception as e:
        tb_str = traceback.format_exc()
        status_logs.append(f"[{ticker_symbol}] Critical Pipeline Exception: {type(e).__name__} - {str(e)}\n{tb_str}")
        return {
            "ticker": ticker_symbol,
            "success": False,
            "logs": "\n".join(status_logs),
            "df": None, "current_price": None, "predicted_price": None, "expected_return": None
        }

# --- Main Prediction Function ---
def predict_dynamic_forecast(ticker_selection, forecast_periods_str):
    status_message = ""
    empty_forecast_df = pd.DataFrame(columns=['Date (ds)', 'Predicted Price (yhat)', 'Lower Bound', 'Upper Bound'])

    if not ticker_selection:
        return "Please select a ticker.", empty_forecast_df
    
    try:
        forecast_periods = int(forecast_periods_str)
        if forecast_periods <= 0:
            return "Forecast periods must be a positive number.", empty_forecast_df
    except ValueError:
        return "Invalid number for forecast periods (must be an integer).", empty_forecast_df

    # --- MODE: ALL TICKERS (PARALLEL EXECUTION) ---
    if ticker_selection == "ALL":
        status_message += f"Initiating parallel forecast execution for all models for {forecast_periods} days...\n\n"
        tickers_to_run = list(model_hyperparams_catalogue.keys())
        
        results = []
        # Multi-threaded runtime allocation for IO bound downloads and predictions
        with ThreadPoolExecutor(max_workers=min(len(tickers_to_run), 8)) as executor:
            futures = {
                executor.submit(
                    run_single_inference, tk, forecast_periods, model_hyperparams_catalogue[tk]
                ): tk for tk in tickers_to_run
            }
            for fut in futures:
                results.append(fut.result())
        
        summary_rows = []
        detailed_logs = []
        
        for res in results:
            ticker_name = TICKER_TO_FULL_NAME.get(res["ticker"], res["ticker"])
            detailed_logs.append(f"==================== {ticker_name} Logs ====================")
            detailed_logs.append(res["logs"])
            detailed_logs.append("-" * 60 + "\n")
            
            if res["success"]:
                summary_rows.append({
                    "Ticker": res["ticker"],
                    "Full Name": ticker_name,
                    "Current Price": round(res["current_price"], 2),
                    "Predicted Price (Final Day)": round(res["predicted_price"], 2),
                    "Expected Return (%)": f"{res['expected_return']:+.2f}%",
                    "Lower Bound (Final)": round(res["lower_bound"], 2),
                    "Upper Bound (Final)": round(res["upper_bound"], 2)
                })
            else:
                summary_rows.append({
                    "Ticker": res["ticker"],
                    "Full Name": ticker_name,
                    "Current Price": round(res["current_price"], 2) if res["current_price"] is not None else "N/A",
                    "Predicted Price (Final Day)": "FAILED",
                    "Expected Return (%)": "N/A",
                    "Lower Bound (Final)": "N/A",
                    "Upper Bound (Final)": "N/A"
                })
        
        # Build comprehensive returns summary inside log box
        status_message += "================== EXPECTED RETURNS SUMMARY ==================\n"
        for row in summary_rows:
            if row["Expected Return (%)"] != "N/A":
                status_message += f"• {row['Ticker']}: Current: {row['Current Price']} -> Predicted: {row['Predicted Price (Final Day)']} | Exp. Return: {row['Expected Return (%)']}\n"
            else:
                status_message += f"• {row['Ticker']}: Run Failed (Check detailed execution trace below)\n"
        
        status_message += "==============================================================\n\n"
        status_message += "\n".join(detailed_logs)
        
        summary_df = pd.DataFrame(summary_rows)
        return status_message, summary_df

    # --- MODE: SINGLE TICKER EXECUTION ---
    else:
        hyperparams = model_hyperparams_catalogue.get(ticker_selection)
        if not hyperparams: 
            return f"Internal Error: Config for '{ticker_selection}' not found.", empty_forecast_df
            
        res = run_single_inference(ticker_selection, forecast_periods, hyperparams)
        
        if res["success"]:
            ticker_name = TICKER_TO_FULL_NAME.get(ticker_selection, ticker_selection)
            # Display computed expected return percentages in the log box header
            summary_header = (
                f"================== FORECAST SUMMARY: {ticker_selection} ==================\n"
                f"Full Asset Name: {ticker_name}\n"
                f"Current Price (Latest Close): {res['current_price']:.2f}\n"
                f"Final Predicted Price ({forecast_periods} Days): {res['predicted_price']:.2f}\n"
                f"Expected Return Percentage: {res['expected_return']:+.2f}%\n"
                f"==================================================================\n\n"
            )
            status_message = summary_header + res["logs"]
            return status_message, res["df"]
        else:
            return res["logs"], empty_forecast_df

# --- Gradio Interface Definition ---
with gr.Blocks(css="footer {visibility: hidden}", title="Stock/Commodity Forecaster") as iface:
    gr.Markdown("# Stock & Commodity Price Forecaster")
    gr.Markdown(
        "This tool fetches the latest market data using Yahoo Finance via yfinance, "
        "re-fits a Prophet time series model on-the-fly, "
        "and generates future forecasts. Use the option **All Tickers (Parallel Inference)** "
        "to run simulations for every available configuration concurrently. "
        "[Project repository for details](https://github.com/akshit0201/Prophet-Commodity-Stock-analysis/blob/main/README.md)"
        "\n\n**Note:** Forecasts are for informational purposes only and not financial advice."
    )
    if not dropdown_choices:
        gr.Markdown("<h3 style='color:red;'>WARNING: No model configurations loaded. Check 'trained_models' folder structure.</h3>")

    with gr.Row():
        with gr.Column(scale=1):
            ticker_dropdown = gr.Dropdown(
                choices=dropdown_choices, 
                label="Select Ticker Symbol",
                info="Choose the stock/commodity or run parallel inference on all.",
                value="ALL" if len(dropdown_choices) > 1 else None
            )
            periods_input = gr.Number(
                value=30, 
                label="Forecast Periods (Days)", 
                minimum=1, 
                maximum=365 * 2, 
                step=1,
                info="Number of future days to predict."
            )
            predict_button = gr.Button("Generate Forecast", variant="primary")
        
        with gr.Column(scale=3):
            status_textbox = gr.Textbox(
                label="Process Status & Logs", 
                lines=15, 
                interactive=False,
                placeholder="Execution steps, metrics, and details will show here..."
            )
            
    gr.Markdown("## Forecast Results")
    forecast_output_table = gr.DataFrame(
        label="Forecast Summary & Metrics Table"
    )

    predict_button.click(
        fn=predict_dynamic_forecast,
        inputs=[ticker_dropdown, periods_input],
        outputs=[status_textbox, forecast_output_table]
    )

    gr.Markdown("---")
    gr.Markdown(
        "**How it works:** Core models utilize Prophet (log-transformed and exponentiated to base scales). "
        "When single assets are chosen, a daily forecast table is output. When running on **All Tickers**, "
        "the app utilizes multithreading to scale historical downloads and inference tasks simultaneously, "
        "outputting a standardized performance evaluation table measuring relative Expected Returns."
    )

# --- Launch the Gradio App ---
if __name__ == "__main__":
    print("STARTUP INFO: Launching Gradio interface...")
    iface.launch()