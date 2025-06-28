import os
import pandas as pd
from alpha_vantage.timeseries import TimeSeries
from googlesearch import search

# Load Alpha Vantage key from environment
ALPHA_VANTAGE_API_KEY = os.getenv("ALPHA_VANTAGE_API_KEY")

# Fetch daily stock data using Alpha Vantage
def fetch_stock_data(symbol: str, outputsize: str = 'compact'):
    ts = TimeSeries(key=ALPHA_VANTAGE_API_KEY, output_format='pandas')
    data, _ = ts.get_daily(symbol=symbol, outputsize=outputsize)

    data = data.rename(columns={
        "1. open": "open", "2. high": "high", "3. low": "low",
        "4. close": "close", "5. volume": "volume"
    })
    data.index = pd.to_datetime(data.index)
    return data.sort_index()

# Simple moving average forecast
def moving_average_forecast(data: pd.DataFrame, window: int = 5):
    close_prices = data['close']
    ma = close_prices.rolling(window=window).mean()

    # Slope determines trend
    slope = (ma.iloc[-1] - ma.iloc[-window]) / window
    trend = "up" if slope > 0 else "down" if slope < 0 else "stable"

    return {
        "trend": trend,
        "confidence": round(abs(slope) * 100, 2),
        "latest_price": round(close_prices.iloc[-1], 2),
        "predicted_price": round(close_prices.iloc[-1] + slope, 2)
    }

# Forecast pipeline
def forecast_pipeline(symbol: str):
    data = fetch_stock_data(symbol)
    forecast = moving_average_forecast(data)
    return forecast

def get_symbol(company_name):
    query = f"{company_name} stock symbol site:finance.yahoo.com"
    for result in search(query, num_results=5):
        if "quote/" in result:
            symbol = result.split("quote/")[1].split("?")[0].split("/")[0]
            return symbol
    return None