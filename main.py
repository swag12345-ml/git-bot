import yfinance as yf
import pandas as pd

# Define tickers: Stocks, Bonds, Gold, Crypto
tickers = {
    "S&P 500": "^GSPC",   # US stock index
    "Bonds (IEF)": "IEF",  # Treasury ETF
    "Gold (GLD)": "GLD",   # Gold ETF
    "Bitcoin": "BTC-USD",  # Bitcoin in USD
    "Ethereum": "ETH-USD"  # Ethereum in USD
}

print("🔄 Fetching latest 5-day market data...\n")

for name, symbol in tickers.items():
    try:
        data = yf.Ticker(symbol).history(period="5d")["Close"]
        change = (data.iloc[-1] - data.iloc[0]) / data.iloc[0] * 100
        print(f"{name}: {change:+.2f}% over last 5 days")
    except Exception as e:
        print(f"{name}: ⚠️ Could not fetch data ({e})")

print("\n✅ Live market data fetched successfully!")
