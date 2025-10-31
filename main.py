import streamlit as st
import yfinance as yf
import pandas as pd
import time
import plotly.graph_objects as go

# ---------------- PAGE CONFIG ----------------
st.set_page_config(page_title="Live Market Dashboard", page_icon="📊", layout="wide")

# ---------------- HEADER ----------------
st.markdown("""
    <div style="text-align:center; padding: 10px 0;">
        <h1>📊 Live Market Dashboard</h1>
        <p>Track real-time performance of global assets — Stocks, Bonds, Gold, and Crypto — powered by <b>Yahoo Finance</b>.</p>
    </div>
""", unsafe_allow_html=True)

# ---------------- FUNCTION: FETCH DATA ----------------
def get_market_sentiment():
    tickers = {
        "S&P 500": "^GSPC",
        "Bonds (IEF)": "IEF",
        "Gold (GLD)": "GLD",
        "Bitcoin (BTC)": "BTC-USD",
        "Ethereum (ETH)": "ETH-USD"
    }

    results = {}
    for name, symbol in tickers.items():
        try:
            data = yf.Ticker(symbol).history(period="5d")["Close"]
            change = (data.iloc[-1] - data.iloc[0]) / data.iloc[0] * 100
            results[name] = round(change, 2)
        except Exception:
            results[name] = None
    return results

# ---------------- FUNCTION: PRICE HISTORY ----------------
def get_price_history(symbol):
    try:
        data = yf.Ticker(symbol).history(period="1mo")
        return data
    except Exception:
        return None

# ---------------- DASHBOARD UPDATE ----------------
placeholder = st.empty()

while True:
    sentiment = get_market_sentiment()

    with placeholder.container():
        st.markdown("### 🌍 Market Performance (Last 5 Days)")
        col1, col2, col3, col4, col5 = st.columns(5)

        # Dynamic metrics
        assets = list(sentiment.keys())
        cols = [col1, col2, col3, col4, col5]

        for i, asset in enumerate(assets):
            value = sentiment[asset]
            if value is None:
                cols[i].metric(asset, "N/A", "⚠️")
            else:
                cols[i].metric(
                    asset,
                    f"{value:+.2f}%",
                    "⬆️" if value > 0 else "⬇️"
                )

        st.markdown("---")
        st.markdown("### 📈 Trend Visualization (Past Month)")

        # Interactive chart
        chart_tickers = {
            "S&P 500": "^GSPC",
            "Bitcoin": "BTC-USD",
            "Gold": "GLD"
        }

        chart = go.Figure()
        for name, symbol in chart_tickers.items():
            data = get_price_history(symbol)
            if data is not None:
                chart.add_trace(go.Scatter(
                    x=data.index,
                    y=data["Close"],
                    mode="lines",
                    name=name
                ))

        chart.update_layout(
            title="Market Price Trends (Past Month)",
            xaxis_title="Date",
            yaxis_title="Price (USD)",
            template="plotly_white",
            height=500
        )
        st.plotly_chart(chart, use_container_width=True)

        # Info footer
        st.info("💡 Data refreshes automatically every 60 seconds without reloading the page.")
        st.caption("Source: Yahoo Finance | Updated in real time using yfinance")

    # Wait 60 seconds before updating
    time.sleep(60)
