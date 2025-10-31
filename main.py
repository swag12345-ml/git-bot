import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.graph_objects as go
import time

st.set_page_config(page_title="AI Investment Advisor", page_icon="🤖", layout="wide")

st.markdown("""
<div style="text-align:center; padding: 10px 0;">
  <h1>🤖 AI Investment Advisor</h1>
  <p>Live, data-driven asset allocation suggestions based on current market trends.</p>
</div>
""", unsafe_allow_html=True)


# ---------- Fetch 5-Day Market Data ----------
def get_market_sentiment():
    tickers = {
        "S&P 500": "^GSPC",
        "Bonds (IEF)": "IEF",
        "Gold (GLD)": "GLD",
        "Bitcoin (BTC)": "BTC-USD",
        "Ethereum (ETH)": "ETH-USD"
    }
    sentiment = {}
    for name, symbol in tickers.items():
        try:
            data = yf.Ticker(symbol).history(period="5d")["Close"]
            change = (data.iloc[-1] - data.iloc[0]) / data.iloc[0] * 100
            sentiment[name] = round(change, 2)
        except Exception:
            sentiment[name] = 0.0
    return sentiment


# ---------- AI-like Allocation Logic ----------
def ai_investment_recommendation(sentiment):
    """
    Allocate 100% among the 5 assets.
    - Positive change → higher weight
    - Negative change → lower weight
    - Normalized so total = 100%
    """
    # Convert sentiment to weights (shift so negatives still positive)
    min_val = min(sentiment.values())
    adjusted = {k: v - min_val + 1 for k, v in sentiment.items()}  # +1 to avoid zero
    total = sum(adjusted.values())
    allocation = {k: round(v / total * 100, 2) for k, v in adjusted.items()}
    return allocation


# ---------- Price History for Plot ----------
def get_price_history(symbol):
    try:
        data = yf.Ticker(symbol).history(period="1mo")
        return data
    except Exception:
        return None


# ---------- Live Dashboard ----------
placeholder = st.empty()

while True:
    sentiment = get_market_sentiment()
    allocation = ai_investment_recommendation(sentiment)

    with placeholder.container():
        st.markdown("### 📊 Live Market Sentiment (Last 5 Days)")
        cols = st.columns(5)
        for i, (asset, change) in enumerate(sentiment.items()):
            cols[i].metric(asset, f"{change:+.2f}%", "⬆️" if change > 0 else "⬇️")

        st.markdown("---")
        st.markdown("### 💼 AI-Suggested Allocation (100% Total)")
        alloc_cols = st.columns(5)
        for i, (asset, pct) in enumerate(allocation.items()):
            alloc_cols[i].metric(asset, f"{pct}%")

        # Explain logic
        st.info("""
        💡 **AI Logic Summary**
        - Assets rising faster receive higher allocation.  
        - Falling assets are down-weighted but still included for diversification.  
        - Total always equals 100%.  
        - Data refreshes every 60 seconds without reloading the page.
        """)

        # Plot 1-Month Trends
        st.markdown("### 📈 1-Month Price Trends")
        chart_tickers = {
            "S&P 500": "^GSPC",
            "Bitcoin": "BTC-USD",
            "Gold": "GLD",
            "Ethereum": "ETH-USD",
            "Bonds": "IEF"
        }
        fig = go.Figure()
        for name, symbol in chart_tickers.items():
            data = get_price_history(symbol)
            if data is not None:
                fig.add_trace(go.Scatter(
                    x=data.index, y=data["Close"], mode="lines", name=name))
        fig.update_layout(
            title="Asset Price Trends (Past Month)",
            xaxis_title="Date",
            yaxis_title="Price (USD)",
            template="plotly_white", height=500)
        st.plotly_chart(fig, use_container_width=True)

        st.caption("Source: Yahoo Finance | Auto-updates every 60 seconds")

    time.sleep(60)
