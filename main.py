"""
AI-Powered Investment Portfolio Dashboard
==========================================
A modern glassmorphism-styled portfolio dashboard with blue translucent theme,
live market data from Yahoo Finance, and AI-powered insights using LLaMA via LangChain.

Author: AI Investment Systems
License: MIT
"""

import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

try:
    from langchain_community.llms import Ollama
    from langchain.prompts import PromptTemplate
    from langchain.chains import LLMChain
    LANGCHAIN_AVAILABLE = True
except ImportError:
    LANGCHAIN_AVAILABLE = False
    st.warning("LangChain not available. AI features will be simulated.")


st.set_page_config(
    page_title="AI Portfolio Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)


GLASSMORPHISM_CSS = """
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap');

    * {
        font-family: 'Inter', sans-serif;
    }

    .stApp {
        background: linear-gradient(135deg,
            rgba(0, 40, 80, 0.95) 0%,
            rgba(0, 80, 140, 0.9) 25%,
            rgba(0, 100, 180, 0.85) 50%,
            rgba(0, 60, 120, 0.9) 75%,
            rgba(0, 30, 70, 0.95) 100%
        );
        background-attachment: fixed;
    }

    .glass-card {
        background: rgba(255, 255, 255, 0.08);
        backdrop-filter: blur(20px);
        -webkit-backdrop-filter: blur(20px);
        border-radius: 20px;
        border: 1px solid rgba(255, 255, 255, 0.18);
        padding: 25px;
        margin: 15px 0;
        box-shadow: 0 8px 32px 0 rgba(0, 150, 255, 0.2);
        transition: all 0.3s ease;
    }

    .glass-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 12px 40px 0 rgba(0, 200, 255, 0.4);
        border: 1px solid rgba(0, 200, 255, 0.4);
    }

    .metric-card {
        background: rgba(0, 150, 255, 0.12);
        backdrop-filter: blur(15px);
        border-radius: 15px;
        border: 1px solid rgba(0, 200, 255, 0.3);
        padding: 20px;
        margin: 10px 0;
        box-shadow: 0 4px 20px rgba(0, 180, 255, 0.25);
        transition: all 0.3s ease;
    }

    .metric-card:hover {
        background: rgba(0, 180, 255, 0.18);
        border: 1px solid rgba(0, 230, 255, 0.5);
        box-shadow: 0 8px 30px rgba(0, 200, 255, 0.4);
    }

    .neon-text {
        color: #00d4ff;
        text-shadow: 0 0 10px rgba(0, 212, 255, 0.8),
                     0 0 20px rgba(0, 212, 255, 0.6),
                     0 0 30px rgba(0, 212, 255, 0.4);
        font-weight: 700;
    }

    .header-text {
        color: #ffffff;
        text-shadow: 0 2px 10px rgba(0, 150, 255, 0.5);
        font-weight: 600;
    }

    .positive-change {
        color: #00ff88;
        font-weight: 600;
    }

    .negative-change {
        color: #ff4466;
        font-weight: 600;
    }

    .regime-bullish {
        background: linear-gradient(135deg, rgba(0, 255, 136, 0.2), rgba(0, 200, 100, 0.3));
        border: 2px solid rgba(0, 255, 136, 0.6);
        border-radius: 15px;
        padding: 15px;
        color: #00ff88;
        font-weight: 700;
        text-align: center;
        box-shadow: 0 0 20px rgba(0, 255, 136, 0.4);
    }

    .regime-bearish {
        background: linear-gradient(135deg, rgba(255, 68, 102, 0.2), rgba(200, 50, 80, 0.3));
        border: 2px solid rgba(255, 68, 102, 0.6);
        border-radius: 15px;
        padding: 15px;
        color: #ff4466;
        font-weight: 700;
        text-align: center;
        box-shadow: 0 0 20px rgba(255, 68, 102, 0.4);
    }

    .regime-sideways {
        background: linear-gradient(135deg, rgba(255, 200, 0, 0.2), rgba(200, 160, 0, 0.3));
        border: 2px solid rgba(255, 200, 0, 0.6);
        border-radius: 15px;
        padding: 15px;
        color: #ffc800;
        font-weight: 700;
        text-align: center;
        box-shadow: 0 0 20px rgba(255, 200, 0, 0.4);
    }

    .ai-insight-box {
        background: linear-gradient(135deg, rgba(138, 43, 226, 0.15), rgba(75, 0, 130, 0.2));
        backdrop-filter: blur(15px);
        border-radius: 15px;
        border: 2px solid rgba(138, 43, 226, 0.4);
        padding: 20px;
        margin: 15px 0;
        box-shadow: 0 0 25px rgba(138, 43, 226, 0.3);
        color: #e0d4ff;
    }

    .stButton>button {
        background: linear-gradient(135deg, rgba(0, 150, 255, 0.8), rgba(0, 100, 200, 0.9));
        color: white;
        border: 2px solid rgba(0, 200, 255, 0.5);
        border-radius: 12px;
        padding: 12px 30px;
        font-weight: 600;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(0, 150, 255, 0.4);
    }

    .stButton>button:hover {
        background: linear-gradient(135deg, rgba(0, 180, 255, 1), rgba(0, 120, 220, 1));
        border: 2px solid rgba(0, 230, 255, 0.8);
        box-shadow: 0 6px 25px rgba(0, 180, 255, 0.6);
        transform: translateY(-2px);
    }

    .stTabs [data-baseweb="tab-list"] {
        gap: 10px;
        background: rgba(0, 50, 100, 0.3);
        border-radius: 15px;
        padding: 10px;
    }

    .stTabs [data-baseweb="tab"] {
        background: rgba(255, 255, 255, 0.05);
        border-radius: 10px;
        padding: 10px 20px;
        color: #ffffff;
        border: 1px solid rgba(255, 255, 255, 0.1);
        transition: all 0.3s ease;
    }

    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, rgba(0, 150, 255, 0.4), rgba(0, 100, 200, 0.5));
        border: 1px solid rgba(0, 200, 255, 0.6);
        box-shadow: 0 4px 15px rgba(0, 150, 255, 0.4);
    }

    h1, h2, h3 {
        color: #ffffff;
        text-shadow: 0 2px 15px rgba(0, 150, 255, 0.6);
    }

    .stMetric {
        background: rgba(0, 150, 255, 0.1);
        padding: 15px;
        border-radius: 10px;
        border: 1px solid rgba(0, 200, 255, 0.2);
    }

    .stMetric label {
        color: #b0d4ff !important;
    }

    .stMetric [data-testid="stMetricValue"] {
        color: #00d4ff !important;
        font-weight: 700 !important;
    }
</style>
"""

st.markdown(GLASSMORPHISM_CSS, unsafe_allow_html=True)


MARKET_TICKERS = {
    'stocks': ['AAPL', 'TSLA', 'MSFT', 'META'],
    'bonds': ['^TNX'],
    'commodities': ['GC=F'],
    'currency': ['DX-Y.NYB']
}


@st.cache_data(ttl=300)
def get_market_data(ticker: str, period: str = '1mo', interval: str = '1d') -> Optional[pd.DataFrame]:
    """
    Fetch market data from Yahoo Finance with caching.

    Args:
        ticker: Stock ticker symbol
        period: Data period (1d, 5d, 1mo, 3mo, 6mo, 1y, 2y, 5y, 10y, ytd, max)
        interval: Data interval (1m, 2m, 5m, 15m, 30m, 60m, 90m, 1h, 1d, 5d, 1wk, 1mo, 3mo)

    Returns:
        DataFrame with OHLCV data or None if fetch fails
    """
    try:
        stock = yf.Ticker(ticker)
        df = stock.history(period=period, interval=interval)
        if df.empty:
            return None
        return df
    except Exception as e:
        st.error(f"Error fetching {ticker}: {str(e)}")
        return None


def calculate_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    """
    Calculate Relative Strength Index (RSI).

    Args:
        series: Price series
        period: RSI period (default 14)

    Returns:
        Series with RSI values
    """
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()

    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi


def calculate_volatility(series: pd.Series, window: int = 20) -> float:
    """
    Calculate annualized volatility.

    Args:
        series: Price series
        window: Rolling window period

    Returns:
        Annualized volatility percentage
    """
    returns = series.pct_change().dropna()
    volatility = returns.rolling(window=window).std().iloc[-1] * np.sqrt(252) * 100
    return volatility


def get_ticker_info(ticker: str) -> Dict:
    """
    Get comprehensive ticker information including price, changes, RSI, and volatility.

    Args:
        ticker: Stock ticker symbol

    Returns:
        Dictionary with ticker metrics
    """
    df = get_market_data(ticker, period='1mo', interval='1d')

    if df is None or df.empty:
        return {
            'ticker': ticker,
            'current_price': 0,
            'daily_change': 0,
            'five_day_change': 0,
            'rsi': 0,
            'volatility': 0,
            'data': None
        }

    current_price = df['Close'].iloc[-1]
    daily_change = ((current_price - df['Close'].iloc[-2]) / df['Close'].iloc[-2] * 100) if len(df) > 1 else 0
    five_day_change = ((current_price - df['Close'].iloc[-6]) / df['Close'].iloc[-6] * 100) if len(df) > 5 else 0

    rsi = calculate_rsi(df['Close']).iloc[-1]
    volatility = calculate_volatility(df['Close'])

    return {
        'ticker': ticker,
        'current_price': current_price,
        'daily_change': daily_change,
        'five_day_change': five_day_change,
        'rsi': rsi,
        'volatility': volatility,
        'data': df
    }


def create_candlestick_chart(df: pd.DataFrame, ticker: str) -> go.Figure:
    """
    Create an interactive candlestick chart with volume.

    Args:
        df: OHLCV DataFrame
        ticker: Stock ticker symbol

    Returns:
        Plotly figure object
    """
    fig = go.Figure()

    fig.add_trace(go.Candlestick(
        x=df.index,
        open=df['Open'],
        high=df['High'],
        low=df['Low'],
        close=df['Close'],
        name='OHLC',
        increasing_line_color='#00ff88',
        decreasing_line_color='#ff4466'
    ))

    fig.update_layout(
        title=f'{ticker} - Candlestick Chart',
        xaxis_title='Date',
        yaxis_title='Price',
        template='plotly_dark',
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,50,100,0.2)',
        font=dict(color='#ffffff'),
        xaxis=dict(gridcolor='rgba(255,255,255,0.1)'),
        yaxis=dict(gridcolor='rgba(255,255,255,0.1)'),
        hovermode='x unified',
        height=400
    )

    return fig


def create_line_chart(df: pd.DataFrame, ticker: str, days: int = 5) -> go.Figure:
    """
    Create an interactive line chart for short-term trends.

    Args:
        df: Price DataFrame
        ticker: Stock ticker symbol
        days: Number of days to display

    Returns:
        Plotly figure object
    """
    recent_df = df.tail(days)

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=recent_df.index,
        y=recent_df['Close'],
        mode='lines+markers',
        name=ticker,
        line=dict(color='#00d4ff', width=3),
        marker=dict(size=8, color='#00d4ff', line=dict(color='#ffffff', width=2)),
        fill='tozeroy',
        fillcolor='rgba(0, 212, 255, 0.2)'
    ))

    fig.update_layout(
        title=f'{ticker} - {days} Day Trend',
        xaxis_title='Date',
        yaxis_title='Price',
        template='plotly_dark',
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,50,100,0.2)',
        font=dict(color='#ffffff'),
        xaxis=dict(gridcolor='rgba(255,255,255,0.1)'),
        yaxis=dict(gridcolor='rgba(255,255,255,0.1)'),
        hovermode='x unified',
        height=300
    )

    return fig


def display_metric_card(ticker: str, info: Dict):
    """
    Display a glassmorphism metric card for a ticker.

    Args:
        ticker: Stock ticker symbol
        info: Dictionary with ticker metrics
    """
    change_color = 'positive-change' if info['daily_change'] >= 0 else 'negative-change'
    change_arrow = '▲' if info['daily_change'] >= 0 else '▼'

    card_html = f"""
    <div class="metric-card">
        <h3 class="neon-text">{ticker}</h3>
        <div style="font-size: 32px; color: #ffffff; margin: 10px 0;">
            ${info['current_price']:.2f}
        </div>
        <div class="{change_color}" style="font-size: 18px; margin: 5px 0;">
            {change_arrow} {abs(info['daily_change']):.2f}% (1D)
        </div>
        <div style="color: #b0d4ff; margin: 5px 0;">
            5D Change: {info['five_day_change']:.2f}%
        </div>
        <hr style="border-color: rgba(255,255,255,0.2); margin: 10px 0;">
        <div style="display: flex; justify-content: space-between; color: #b0d4ff;">
            <div>
                <div style="font-size: 12px;">RSI(14)</div>
                <div style="font-size: 20px; color: #00d4ff;">{info['rsi']:.1f}</div>
            </div>
            <div>
                <div style="font-size: 12px;">Volatility</div>
                <div style="font-size: 20px; color: #00d4ff;">{info['volatility']:.1f}%</div>
            </div>
        </div>
    </div>
    """

    st.markdown(card_html, unsafe_allow_html=True)


def monte_carlo_simulation(
    initial_investment: float,
    expected_return: float,
    volatility: float,
    time_horizon_years: int = 5,
    num_simulations: int = 1000
) -> np.ndarray:
    """
    Run Monte Carlo simulation for portfolio growth.

    Args:
        initial_investment: Starting investment amount
        expected_return: Annual expected return (as decimal)
        volatility: Annual volatility (as decimal)
        time_horizon_years: Investment time horizon in years
        num_simulations: Number of simulation runs

    Returns:
        Array of final portfolio values
    """
    days = time_horizon_years * 252
    results = np.zeros(num_simulations)

    for i in range(num_simulations):
        daily_returns = np.random.normal(
            expected_return / 252,
            volatility / np.sqrt(252),
            days
        )
        price_path = initial_investment * np.exp(np.cumsum(daily_returns))
        results[i] = price_path[-1]

    return results


def calculate_sharpe_ratio(returns: pd.Series, risk_free_rate: float = 0.02) -> float:
    """
    Calculate Sharpe Ratio.

    Args:
        returns: Series of returns
        risk_free_rate: Annual risk-free rate

    Returns:
        Sharpe ratio
    """
    excess_returns = returns - risk_free_rate / 252
    return np.sqrt(252) * excess_returns.mean() / excess_returns.std()


def get_ai_allocation(
    risk_profile: str,
    market_data: Dict,
    investment_amount: float
) -> Dict:
    """
    Get AI-powered portfolio allocation using LLaMA via LangChain.

    Args:
        risk_profile: User's risk profile (Conservative/Moderate/Aggressive)
        market_data: Dictionary with current market metrics
        investment_amount: Total investment amount

    Returns:
        Dictionary with allocation and reasoning
    """
    if not LANGCHAIN_AVAILABLE:
        if risk_profile == "Conservative":
            allocation = {"stocks": 30, "bonds": 50, "gold": 15, "cash": 5}
            reason = "Conservative portfolio focuses on capital preservation with higher bond allocation."
        elif risk_profile == "Moderate":
            allocation = {"stocks": 50, "bonds": 30, "gold": 15, "cash": 5}
            reason = "Balanced approach with equal weight to growth and stability."
        else:
            allocation = {"stocks": 70, "bonds": 15, "gold": 10, "cash": 5}
            reason = "Aggressive growth strategy with higher equity exposure."

        return {
            "allocation": allocation,
            "reason": reason
        }

    try:
        llm = Ollama(model="llama2", temperature=0.7)

        prompt_template = """You are an expert financial advisor analyzing market conditions.

Current Market Data:
- Stock Market Volatility: {stock_volatility:.2f}%
- Stock RSI Average: {stock_rsi:.1f}
- Bond Yield: {bond_yield:.2f}%
- Gold Price Change: {gold_change:.2f}%

User Profile:
- Risk Tolerance: {risk_profile}
- Investment Amount: ${investment_amount:,.2f}

Provide a diversified portfolio allocation across Stocks, Bonds, Gold, and Cash.
Explain your rationale considering current market conditions and user risk tolerance.

Format your response as JSON:
{{
  "allocation": {{"stocks": XX, "bonds": XX, "gold": XX, "cash": XX}},
  "reason": "Your explanation here"
}}
"""

        prompt = PromptTemplate(
            input_variables=["stock_volatility", "stock_rsi", "bond_yield", "gold_change", "risk_profile", "investment_amount"],
            template=prompt_template
        )

        chain = LLMChain(llm=llm, prompt=prompt)

        response = chain.run(
            stock_volatility=market_data.get('stock_volatility', 20),
            stock_rsi=market_data.get('stock_rsi', 50),
            bond_yield=market_data.get('bond_yield', 4),
            gold_change=market_data.get('gold_change', 0),
            risk_profile=risk_profile,
            investment_amount=investment_amount
        )

        import json
        result = json.loads(response)
        return result

    except Exception as e:
        st.warning(f"AI model unavailable, using rule-based allocation: {str(e)}")

        if risk_profile == "Conservative":
            allocation = {"stocks": 30, "bonds": 50, "gold": 15, "cash": 5}
            reason = "Conservative portfolio with focus on capital preservation and steady income."
        elif risk_profile == "Moderate":
            allocation = {"stocks": 50, "bonds": 30, "gold": 15, "cash": 5}
            reason = "Balanced growth and stability with diversified risk exposure."
        else:
            allocation = {"stocks": 70, "bonds": 15, "gold": 10, "cash": 5}
            reason = "Growth-oriented portfolio maximizing equity exposure for higher returns."

        return {
            "allocation": allocation,
            "reason": reason
        }


def detect_market_regime(market_data: Dict) -> Tuple[str, str]:
    """
    Detect current market regime based on momentum, RSI, and volatility.

    Args:
        market_data: Dictionary with market metrics

    Returns:
        Tuple of (regime, emoji)
    """
    avg_rsi = market_data.get('stock_rsi', 50)
    avg_volatility = market_data.get('stock_volatility', 20)
    momentum = market_data.get('momentum', 0)

    if avg_rsi > 60 and momentum > 2 and avg_volatility < 25:
        return "Bullish", "🟢"
    elif avg_rsi < 40 and momentum < -2:
        return "Bearish", "🔴"
    else:
        return "Sideways", "🟡"


def get_regime_advice(regime: str) -> str:
    """
    Get AI-powered advice for current market regime.

    Args:
        regime: Current market regime

    Returns:
        Advice string
    """
    if not LANGCHAIN_AVAILABLE:
        advice_map = {
            "Bullish": "Strong upward momentum detected. Consider maintaining equity positions with trailing stops. Look for breakout opportunities in growth sectors.",
            "Bearish": "Market showing weakness. Consider defensive positions, increase bond allocation, and hold higher cash reserves. Focus on quality stocks with strong fundamentals.",
            "Sideways": "Range-bound market conditions. Implement range trading strategies, focus on dividend stocks, and maintain balanced allocation. Avoid chasing trends."
        }
        return advice_map.get(regime, "Monitor market conditions closely.")

    try:
        llm = Ollama(model="llama2", temperature=0.7)

        prompt = f"""The current market regime is {regime}.
As a financial advisor, provide a brief 2-3 sentence strategy recommendation for investors in this regime.
Focus on risk management and tactical opportunities."""

        response = llm(prompt)
        return response

    except Exception as e:
        advice_map = {
            "Bullish": "Strong upward momentum detected. Consider maintaining equity positions with trailing stops.",
            "Bearish": "Market showing weakness. Consider defensive positions and higher cash reserves.",
            "Sideways": "Range-bound conditions. Focus on dividend stocks and maintain balanced allocation."
        }
        return advice_map.get(regime, "Monitor market conditions.")


def render_market_overview_tab():
    """
    Render Tab 1: Market Overview with live data and interactive charts.
    """
    st.markdown('<h1 class="header-text">📊 Market Overview</h1>', unsafe_allow_html=True)
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.markdown('<p style="color: #b0d4ff;">Real-time market data powered by Yahoo Finance</p>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

    all_tickers = MARKET_TICKERS['stocks'] + MARKET_TICKERS['bonds'] + MARKET_TICKERS['commodities']

    col1, col2, col3, col4 = st.columns(4)
    columns = [col1, col2, col3, col4]

    for idx, ticker in enumerate(all_tickers):
        with columns[idx % 4]:
            info = get_ticker_info(ticker)
            display_metric_card(ticker, info)

    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.markdown('<h2 class="neon-text">Detailed Charts</h2>', unsafe_allow_html=True)

    selected_ticker = st.selectbox(
        'Select ticker for detailed analysis:',
        all_tickers,
        key='market_overview_ticker'
    )

    info = get_ticker_info(selected_ticker)

    if info['data'] is not None:
        col1, col2 = st.columns(2)

        with col1:
            st.plotly_chart(
                create_line_chart(info['data'], selected_ticker),
                use_container_width=True
            )

        with col2:
            st.plotly_chart(
                create_candlestick_chart(info['data'], selected_ticker),
                use_container_width=True
            )

    st.markdown('</div>', unsafe_allow_html=True)


def render_portfolio_builder_tab():
    """
    Render Tab 2: Portfolio Builder with risk profiling and asset selection.
    """
    st.markdown('<h1 class="header-text">💹 Portfolio Builder</h1>', unsafe_allow_html=True)

    st.markdown('<div class="glass-card">', unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:
        investment_amount = st.number_input(
            'Investment Amount ($)',
            min_value=1000,
            max_value=10000000,
            value=10000,
            step=1000,
            key='investment_amount'
        )

        risk_profile = st.selectbox(
            'Risk Profile',
            ['Conservative', 'Moderate', 'Aggressive'],
            key='risk_profile'
        )

    with col2:
        st.markdown('<p class="header-text" style="font-size: 14px;">Select Asset Classes:</p>', unsafe_allow_html=True)
        include_stocks = st.checkbox('Stocks', value=True, key='include_stocks')
        include_bonds = st.checkbox('Bonds', value=True, key='include_bonds')
        include_gold = st.checkbox('Gold', value=True, key='include_gold')
        include_cash = st.checkbox('Cash', value=True, key='include_cash')

    st.markdown('</div>', unsafe_allow_html=True)

    if st.button('🚀 Generate Portfolio', key='generate_portfolio'):
        st.session_state['portfolio_generated'] = True
        st.session_state['investment_amount'] = investment_amount
        st.session_state['risk_profile'] = risk_profile
        st.session_state['asset_selection'] = {
            'stocks': include_stocks,
            'bonds': include_bonds,
            'gold': include_gold,
            'cash': include_cash
        }

    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.markdown('<h2 class="neon-text">Asset Analysis</h2>', unsafe_allow_html=True)

    stocks = ['AAPL', 'TSLA', 'MSFT', 'META']
    volatilities = []
    returns = []
    labels = []

    for ticker in stocks:
        info = get_ticker_info(ticker)
        if info['data'] is not None and len(info['data']) > 20:
            volatilities.append(info['volatility'])
            returns.append(info['five_day_change'])
            labels.append(ticker)

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=volatilities,
        y=returns,
        mode='markers+text',
        marker=dict(
            size=15,
            color=returns,
            colorscale='RdYlGn',
            showscale=True,
            line=dict(color='white', width=2)
        ),
        text=labels,
        textposition='top center',
        textfont=dict(size=12, color='white'),
        name='Assets'
    ))

    fig.update_layout(
        title='Risk vs Return Analysis',
        xaxis_title='Volatility (%)',
        yaxis_title='5-Day Return (%)',
        template='plotly_dark',
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,50,100,0.2)',
        font=dict(color='#ffffff'),
        xaxis=dict(gridcolor='rgba(255,255,255,0.1)'),
        yaxis=dict(gridcolor='rgba(255,255,255,0.1)'),
        height=400
    )

    st.plotly_chart(fig, use_container_width=True)

    st.markdown('</div>', unsafe_allow_html=True)


def render_ai_insights_tab():
    """
    Render Tab 3: AI Insights & Allocation with LLaMA integration.
    """
    st.markdown('<h1 class="header-text">🤖 AI Insights & Allocation</h1>', unsafe_allow_html=True)

    if 'portfolio_generated' not in st.session_state:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.info('Please generate a portfolio in the Portfolio Builder tab first.')
        st.markdown('</div>', unsafe_allow_html=True)
        return

    investment_amount = st.session_state.get('investment_amount', 10000)
    risk_profile = st.session_state.get('risk_profile', 'Moderate')

    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.markdown(f'<p class="header-text">Investment: ${investment_amount:,.2f} | Risk Profile: {risk_profile}</p>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

    with st.spinner('🧠 AI analyzing market conditions...'):
        stocks = ['AAPL', 'TSLA', 'MSFT', 'META']
        volatilities = []
        rsis = []

        for ticker in stocks:
            info = get_ticker_info(ticker)
            volatilities.append(info['volatility'])
            rsis.append(info['rsi'])

        market_data = {
            'stock_volatility': np.mean(volatilities),
            'stock_rsi': np.mean(rsis),
            'bond_yield': 4.5,
            'gold_change': 1.2,
            'momentum': 1.5
        }

        ai_result = get_ai_allocation(risk_profile, market_data, investment_amount)
        allocation = ai_result['allocation']
        reason = ai_result['reason']

    col1, col2 = st.columns([1, 1])

    with col1:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.markdown('<h3 class="neon-text">Recommended Allocation</h3>', unsafe_allow_html=True)

        for asset, percentage in allocation.items():
            amount = investment_amount * percentage / 100
            st.markdown(f"""
            <div style="margin: 15px 0;">
                <div style="display: flex; justify-content: space-between; margin-bottom: 5px;">
                    <span style="color: #00d4ff; font-weight: 600; text-transform: capitalize;">{asset}</span>
                    <span style="color: #ffffff;">{percentage}% (${amount:,.2f})</span>
                </div>
                <div style="background: rgba(255,255,255,0.1); border-radius: 10px; height: 10px; overflow: hidden;">
                    <div style="background: linear-gradient(90deg, #00d4ff, #0080ff); width: {percentage}%; height: 100%; border-radius: 10px; transition: width 1s ease;"></div>
                </div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown('</div>', unsafe_allow_html=True)

    with col2:
        fig = go.Figure(data=[go.Pie(
            labels=list(allocation.keys()),
            values=list(allocation.values()),
            hole=0.4,
            marker=dict(
                colors=['#00d4ff', '#0080ff', '#ffc800', '#00ff88'],
                line=dict(color='#ffffff', width=2)
            ),
            textfont=dict(size=14, color='white'),
            hovertemplate='<b>%{label}</b><br>%{value}%<br>$%{customdata:,.2f}<extra></extra>',
            customdata=[investment_amount * v / 100 for v in allocation.values()]
        )])

        fig.update_layout(
            title='Portfolio Distribution',
            template='plotly_dark',
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font=dict(color='#ffffff'),
            showlegend=True,
            height=400
        )

        st.plotly_chart(fig, use_container_width=True)

    st.markdown(f"""
    <div class="ai-insight-box">
        <h3 style="color: #e0d4ff; margin-bottom: 15px;">💡 AI Reasoning</h3>
        <p style="color: #ffffff; line-height: 1.8;">{reason}</p>
    </div>
    """, unsafe_allow_html=True)

    st.session_state['allocation'] = allocation


def render_simulator_tab():
    """
    Render Tab 4: Simulator & Risk Analysis with Monte Carlo simulation.
    """
    st.markdown('<h1 class="header-text">📉 Simulator & Risk Analysis</h1>', unsafe_allow_html=True)

    if 'allocation' not in st.session_state:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.info('Please generate AI allocation in the AI Insights tab first.')
        st.markdown('</div>', unsafe_allow_html=True)
        return

    investment_amount = st.session_state.get('investment_amount', 10000)
    allocation = st.session_state.get('allocation', {})

    st.markdown('<div class="glass-card">', unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:
        time_horizon = st.slider('Investment Horizon (Years)', 1, 10, 5, key='time_horizon')

    with col2:
        num_simulations = st.slider('Number of Simulations', 100, 2000, 1000, step=100, key='num_sims')

    st.markdown('</div>', unsafe_allow_html=True)

    with st.spinner('Running Monte Carlo simulation...'):
        expected_return = 0.08
        volatility = 0.15

        stock_weight = allocation.get('stocks', 50) / 100
        bond_weight = allocation.get('bonds', 30) / 100

        expected_return = stock_weight * 0.10 + bond_weight * 0.04 + 0.03
        volatility = stock_weight * 0.20 + bond_weight * 0.05 + 0.02

        results = monte_carlo_simulation(
            investment_amount,
            expected_return,
            volatility,
            time_horizon,
            num_simulations
        )

        min_value = np.min(results)
        median_value = np.median(results)
        max_value = np.max(results)
        percentile_10 = np.percentile(results, 10)
        percentile_90 = np.percentile(results, 90)

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            'Minimum',
            f'${min_value:,.0f}',
            f'{(min_value/investment_amount - 1)*100:.1f}%'
        )

    with col2:
        st.metric(
            '10th Percentile',
            f'${percentile_10:,.0f}',
            f'{(percentile_10/investment_amount - 1)*100:.1f}%'
        )

    with col3:
        st.metric(
            'Median',
            f'${median_value:,.0f}',
            f'{(median_value/investment_amount - 1)*100:.1f}%'
        )

    with col4:
        st.metric(
            '90th Percentile',
            f'${percentile_90:,.0f}',
            f'{(percentile_90/investment_amount - 1)*100:.1f}%'
        )

    st.markdown('<div class="glass-card">', unsafe_allow_html=True)

    fig = go.Figure()

    fig.add_trace(go.Histogram(
        x=results,
        nbinsx=50,
        marker=dict(
            color=results,
            colorscale='RdYlGn',
            line=dict(color='white', width=1)
        ),
        name='Outcomes'
    ))

    fig.add_vline(
        x=median_value,
        line_dash="dash",
        line_color="#00d4ff",
        annotation_text=f"Median: ${median_value:,.0f}",
        annotation_position="top"
    )

    fig.update_layout(
        title=f'Distribution of Portfolio Value After {time_horizon} Years',
        xaxis_title='Portfolio Value ($)',
        yaxis_title='Frequency',
        template='plotly_dark',
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,50,100,0.2)',
        font=dict(color='#ffffff'),
        xaxis=dict(gridcolor='rgba(255,255,255,0.1)'),
        yaxis=dict(gridcolor='rgba(255,255,255,0.1)'),
        height=400
    )

    st.plotly_chart(fig, use_container_width=True)

    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="glass-card">', unsafe_allow_html=True)

    days = time_horizon * 252
    sample_paths = []

    for i in range(min(50, num_simulations)):
        daily_returns = np.random.normal(
            expected_return / 252,
            volatility / np.sqrt(252),
            days
        )
        price_path = investment_amount * np.exp(np.cumsum(daily_returns))
        sample_paths.append(price_path)

    fig = go.Figure()

    time_axis = np.linspace(0, time_horizon, days)

    for path in sample_paths:
        fig.add_trace(go.Scatter(
            x=time_axis,
            y=path,
            mode='lines',
            line=dict(width=1, color='rgba(0, 212, 255, 0.3)'),
            showlegend=False,
            hovertemplate='Year %{x:.1f}<br>Value: $%{y:,.0f}<extra></extra>'
        ))

    median_path = np.median(sample_paths, axis=0)
    fig.add_trace(go.Scatter(
        x=time_axis,
        y=median_path,
        mode='lines',
        line=dict(width=3, color='#00ff88'),
        name='Median Path'
    ))

    fig.update_layout(
        title='Sample Portfolio Growth Trajectories',
        xaxis_title='Years',
        yaxis_title='Portfolio Value ($)',
        template='plotly_dark',
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,50,100,0.2)',
        font=dict(color='#ffffff'),
        xaxis=dict(gridcolor='rgba(255,255,255,0.1)'),
        yaxis=dict(gridcolor='rgba(255,255,255,0.1)'),
        height=400
    )

    st.plotly_chart(fig, use_container_width=True)

    st.markdown('</div>', unsafe_allow_html=True)

    returns_series = pd.Series(np.random.normal(expected_return/252, volatility/np.sqrt(252), 1000))
    sharpe = calculate_sharpe_ratio(returns_series)

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric('Expected Annual Return', f'{expected_return*100:.2f}%')

    with col2:
        st.metric('Annual Volatility', f'{volatility*100:.2f}%')

    with col3:
        st.metric('Sharpe Ratio', f'{sharpe:.2f}')


def render_market_regime_tab():
    """
    Render Tab 5: Market Regime Trends with AI-powered regime detection.
    """
    st.markdown('<h1 class="header-text">🧠 Market Regime Trends</h1>', unsafe_allow_html=True)

    with st.spinner('Analyzing market regime...'):
        stocks = ['AAPL', 'TSLA', 'MSFT', 'META']
        volatilities = []
        rsis = []
        changes = []

        for ticker in stocks:
            info = get_ticker_info(ticker)
            volatilities.append(info['volatility'])
            rsis.append(info['rsi'])
            changes.append(info['five_day_change'])

        market_data = {
            'stock_volatility': np.mean(volatilities),
            'stock_rsi': np.mean(rsis),
            'momentum': np.mean(changes)
        }

        regime, emoji = detect_market_regime(market_data)
        advice = get_regime_advice(regime)

    st.markdown('<div class="glass-card">', unsafe_allow_html=True)

    regime_class = f"regime-{regime.lower()}"
    st.markdown(f"""
    <div class="{regime_class}">
        <h2>{emoji} Current Market Regime: {regime.upper()}</h2>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            'Average RSI',
            f'{market_data["stock_rsi"]:.1f}',
            delta='Overbought' if market_data["stock_rsi"] > 70 else 'Oversold' if market_data["stock_rsi"] < 30 else 'Neutral'
        )

    with col2:
        st.metric(
            'Average Volatility',
            f'{market_data["stock_volatility"]:.1f}%',
            delta='High' if market_data["stock_volatility"] > 25 else 'Normal'
        )

    with col3:
        st.metric(
            '5-Day Momentum',
            f'{market_data["momentum"]:.2f}%',
            delta=market_data["momentum"]
        )

    st.markdown(f"""
    <div class="ai-insight-box">
        <h3 style="color: #e0d4ff; margin-bottom: 15px;">💡 AI Strategy Recommendation</h3>
        <p style="color: #ffffff; line-height: 1.8; font-size: 16px;">{advice}</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.markdown('<h2 class="neon-text">Historical Regime Analysis</h2>', unsafe_allow_html=True)

    aapl_data = get_market_data('AAPL', period='3mo', interval='1d')

    if aapl_data is not None and len(aapl_data) > 30:
        aapl_data['RSI'] = calculate_rsi(aapl_data['Close'])
        aapl_data['Momentum'] = aapl_data['Close'].pct_change(10) * 100

        conditions = [
            (aapl_data['RSI'] > 60) & (aapl_data['Momentum'] > 2),
            (aapl_data['RSI'] < 40) & (aapl_data['Momentum'] < -2)
        ]
        choices = ['Bullish', 'Bearish']
        aapl_data['Regime'] = np.select(conditions, choices, default='Sideways')

        color_map = {'Bullish': '#00ff88', 'Bearish': '#ff4466', 'Sideways': '#ffc800'}

        fig = go.Figure()

        for regime_type in ['Bullish', 'Bearish', 'Sideways']:
            regime_data = aapl_data[aapl_data['Regime'] == regime_type]
            fig.add_trace(go.Scatter(
                x=regime_data.index,
                y=regime_data['Close'],
                mode='markers',
                name=regime_type,
                marker=dict(
                    size=8,
                    color=color_map[regime_type],
                    line=dict(color='white', width=1)
                )
            ))

        fig.add_trace(go.Scatter(
            x=aapl_data.index,
            y=aapl_data['Close'],
            mode='lines',
            name='Price',
            line=dict(color='rgba(255,255,255,0.3)', width=1)
        ))

        fig.update_layout(
            title='AAPL Price with Regime Classification (3 Months)',
            xaxis_title='Date',
            yaxis_title='Price ($)',
            template='plotly_dark',
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,50,100,0.2)',
            font=dict(color='#ffffff'),
            xaxis=dict(gridcolor='rgba(255,255,255,0.1)'),
            yaxis=dict(gridcolor='rgba(255,255,255,0.1)'),
            height=400,
            hovermode='x unified'
        )

        st.plotly_chart(fig, use_container_width=True)

    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.markdown('<h2 class="neon-text">Multi-Asset Regime Heatmap</h2>', unsafe_allow_html=True)

    tickers = ['AAPL', 'TSLA', 'MSFT', 'META']
    regime_scores = []

    for ticker in tickers:
        info = get_ticker_info(ticker)
        score = (info['rsi'] - 50) / 10 + info['five_day_change'] / 5
        regime_scores.append(score)

    fig = go.Figure(data=go.Bar(
        x=tickers,
        y=regime_scores,
        marker=dict(
            color=regime_scores,
            colorscale='RdYlGn',
            line=dict(color='white', width=2)
        ),
        text=[f'{s:.2f}' for s in regime_scores],
        textposition='auto',
        hovertemplate='<b>%{x}</b><br>Regime Score: %{y:.2f}<extra></extra>'
    ))

    fig.update_layout(
        title='Regime Strength Score (Higher = More Bullish)',
        xaxis_title='Asset',
        yaxis_title='Regime Score',
        template='plotly_dark',
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,50,100,0.2)',
        font=dict(color='#ffffff'),
        xaxis=dict(gridcolor='rgba(255,255,255,0.1)'),
        yaxis=dict(gridcolor='rgba(255,255,255,0.1)'),
        height=400
    )

    st.plotly_chart(fig, use_container_width=True)

    st.markdown('</div>', unsafe_allow_html=True)


def main():
    """
    Main application entry point.
    """
    st.markdown("""
    <div style="text-align: center; padding: 20px 0;">
        <h1 class="neon-text" style="font-size: 48px; margin-bottom: 10px;">
            AI-Powered Investment Portfolio Dashboard
        </h1>
        <p style="color: #b0d4ff; font-size: 18px;">
            Live Market Data • AI Insights • Risk Analysis • Portfolio Optimization
        </p>
    </div>
    """, unsafe_allow_html=True)

    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📊 Market Overview",
        "💹 Portfolio Builder",
        "🤖 AI Insights & Allocation",
        "📉 Simulator & Risk Analysis",
        "🧠 Market Regime Trends"
    ])

    with tab1:
        render_market_overview_tab()

    with tab2:
        render_portfolio_builder_tab()

    with tab3:
        render_ai_insights_tab()

    with tab4:
        render_simulator_tab()

    with tab5:
        render_market_regime_tab()

    st.markdown("""
    <div style="text-align: center; padding: 30px 0; margin-top: 50px; border-top: 1px solid rgba(255,255,255,0.1);">
        <p style="color: #7090b0; font-size: 14px;">
            Powered by Yahoo Finance, LangChain & LLaMA | Real-time market data with 15-minute delay
        </p>
        <p style="color: #506080; font-size: 12px; margin-top: 10px;">
            Disclaimer: This is for educational purposes only. Not financial advice.
        </p>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
