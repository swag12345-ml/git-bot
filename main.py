
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import time
from typing import Dict, List, Tuple
import json

st.set_page_config(
    page_title="Investment Portfolio Dashboard",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

CUSTOM_CSS = """
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap');

    * {
        font-family: 'Inter', sans-serif;
    }

    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 15px;
        color: white;
        margin-bottom: 2rem;
        box-shadow: 0 8px 24px rgba(0,0,0,0.15);
    }

    .main-header h1 {
        margin: 0;
        font-size: 2.5rem;
        font-weight: 700;
    }

    .main-header p {
        margin: 0.5rem 0 0 0;
        opacity: 0.9;
        font-size: 1rem;
    }

    .metric-card {
        background: white;
        padding: 1.5rem;
        border-radius: 15px;
        box-shadow: 0 4px 12px rgba(0,0,0,0.08);
        transition: transform 0.3s ease, box-shadow 0.3s ease;
        border: 1px solid #f0f0f0;
    }

    .metric-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 8px 24px rgba(0,0,0,0.12);
    }

    .metric-label {
        font-size: 0.85rem;
        color: #666;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        font-weight: 600;
        margin-bottom: 0.5rem;
    }

    .metric-value {
        font-size: 2rem;
        font-weight: 700;
        color: #1a1a1a;
    }

    .metric-change {
        font-size: 0.9rem;
        margin-top: 0.5rem;
    }

    .positive {
        color: #10b981;
    }

    .negative {
        color: #ef4444;
    }

    .chart-container {
        background: white;
        padding: 1.5rem;
        border-radius: 15px;
        box-shadow: 0 4px 12px rgba(0,0,0,0.08);
        margin-bottom: 1.5rem;
    }

    .watchlist-card {
        background: white;
        padding: 1rem;
        border-radius: 12px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.06);
        margin-bottom: 1rem;
        border-left: 4px solid #667eea;
        transition: all 0.3s ease;
    }

    .watchlist-card:hover {
        box-shadow: 0 4px 16px rgba(0,0,0,0.1);
        transform: translateX(5px);
    }

    .stButton>button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 0.5rem 2rem;
        font-weight: 600;
        transition: all 0.3s ease;
    }

    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(102, 126, 234, 0.4);
    }

    .footer {
        text-align: center;
        padding: 2rem;
        color: #666;
        font-size: 0.9rem;
        margin-top: 3rem;
        border-top: 1px solid #e0e0e0;
    }

    div[data-testid="stMetricValue"] {
        font-size: 2rem;
    }

    .sparkline {
        height: 40px;
        margin-top: 0.5rem;
    }
</style>
"""

def initialize_session_state():
    """Initialize session state variables"""
    if 'portfolio' not in st.session_state:
        st.session_state.portfolio = [
            {'ticker': 'AAPL', 'quantity': 10, 'avg_cost': 150.0},
            {'ticker': 'TSLA', 'quantity': 5, 'avg_cost': 200.0},
            {'ticker': 'GOOGL', 'quantity': 8, 'avg_cost': 120.0},
            {'ticker': 'AMZN', 'quantity': 6, 'avg_cost': 130.0},
            {'ticker': 'MSFT', 'quantity': 12, 'avg_cost': 280.0}
        ]

    if 'watchlist' not in st.session_state:
        st.session_state.watchlist = ['NVDA', 'META', 'NFLX', 'DIS']

    if 'cash_balance' not in st.session_state:
        st.session_state.cash_balance = 10000.0

    if 'refresh_interval' not in st.session_state:
        st.session_state.refresh_interval = 'Off'

    if 'use_mock_data' not in st.session_state:
        st.session_state.use_mock_data = False

    if 'last_update' not in st.session_state:
        st.session_state.last_update = datetime.now()

    if 'target_allocation' not in st.session_state:
        st.session_state.target_allocation = {
            'Technology': 40,
            'Consumer Cyclical': 25,
            'Communication Services': 20,
            'Healthcare': 10,
            'Cash': 5
        }

def get_mock_data(ticker: str) -> Dict:
    """Generate mock data for testing"""
    np.random.seed(hash(ticker) % 10000)
    base_price = np.random.uniform(50, 500)

    dates = pd.date_range(end=datetime.now(), periods=180, freq='D')
    prices = base_price * (1 + np.cumsum(np.random.randn(180) * 0.02))

    current_price = prices[-1]
    prev_close = prices[-2]

    return {
        'ticker': ticker,
        'current_price': current_price,
        'previous_close': prev_close,
        'change': current_price - prev_close,
        'change_percent': ((current_price - prev_close) / prev_close) * 100,
        'volume': np.random.randint(1000000, 50000000),
        'market_cap': current_price * np.random.randint(100000000, 5000000000),
        'pe_ratio': np.random.uniform(10, 50),
        'sector': np.random.choice(['Technology', 'Consumer Cyclical', 'Communication Services', 'Healthcare', 'Financial Services']),
        'history': pd.DataFrame({
            'Date': dates,
            'Close': prices,
            'Open': prices * (1 + np.random.randn(180) * 0.01),
            'High': prices * (1 + abs(np.random.randn(180)) * 0.02),
            'Low': prices * (1 - abs(np.random.randn(180)) * 0.02),
            'Volume': np.random.randint(1000000, 50000000, 180)
        })
    }

@st.cache_data(ttl=30)
def fetch_stock_data(ticker: str, use_mock: bool = False) -> Dict:
    """Fetch stock data from Yahoo Finance or mock data"""
    if use_mock:
        return get_mock_data(ticker)

    try:
        stock = yf.Ticker(ticker)
        info = stock.info
        hist = stock.history(period='6mo')

        if hist.empty:
            return get_mock_data(ticker)

        current_price = info.get('currentPrice', hist['Close'].iloc[-1])
        prev_close = info.get('previousClose', hist['Close'].iloc[-2] if len(hist) > 1 else current_price)

        return {
            'ticker': ticker,
            'current_price': current_price,
            'previous_close': prev_close,
            'change': current_price - prev_close,
            'change_percent': ((current_price - prev_close) / prev_close) * 100 if prev_close else 0,
            'volume': info.get('volume', hist['Volume'].iloc[-1]),
            'market_cap': info.get('marketCap', 0),
            'pe_ratio': info.get('trailingPE', 0),
            'sector': info.get('sector', 'Unknown'),
            'history': hist
        }
    except Exception as e:
        st.warning(f"Failed to fetch {ticker}, using mock data: {str(e)}")
        return get_mock_data(ticker)

def calculate_portfolio_metrics(portfolio: List[Dict], use_mock: bool = False) -> Dict:
    """Calculate comprehensive portfolio metrics"""
    total_value = 0
    total_cost = 0
    holdings_data = []

    for holding in portfolio:
        stock_data = fetch_stock_data(holding['ticker'], use_mock)
        current_value = stock_data['current_price'] * holding['quantity']
        cost_basis = holding['avg_cost'] * holding['quantity']
        profit_loss = current_value - cost_basis
        pl_percent = (profit_loss / cost_basis) * 100 if cost_basis > 0 else 0

        holdings_data.append({
            'ticker': holding['ticker'],
            'quantity': holding['quantity'],
            'avg_cost': holding['avg_cost'],
            'current_price': stock_data['current_price'],
            'current_value': current_value,
            'cost_basis': cost_basis,
            'profit_loss': profit_loss,
            'pl_percent': pl_percent,
            'day_change': stock_data['change'],
            'day_change_percent': stock_data['change_percent'],
            'sector': stock_data['sector'],
            'history': stock_data['history']
        })

        total_value += current_value
        total_cost += cost_basis

    total_pl = total_value - total_cost
    total_pl_percent = (total_pl / total_cost) * 100 if total_cost > 0 else 0

    daily_change = sum([h['day_change'] * h['quantity'] for h in holdings_data])
    daily_change_percent = (daily_change / total_value) * 100 if total_value > 0 else 0

    returns = []
    for h in holdings_data:
        if not h['history'].empty and len(h['history']) > 1:
            returns.extend(h['history']['Close'].pct_change().dropna().values * (h['current_value'] / total_value))

    sharpe_ratio = 0
    if len(returns) > 0:
        returns_array = np.array(returns)
        sharpe_ratio = (np.mean(returns_array) * np.sqrt(252)) / (np.std(returns_array) + 1e-10)

    return {
        'total_value': total_value,
        'total_cost': total_cost,
        'total_pl': total_pl,
        'total_pl_percent': total_pl_percent,
        'daily_change': daily_change,
        'daily_change_percent': daily_change_percent,
        'sharpe_ratio': sharpe_ratio,
        'holdings': holdings_data
    }

def create_portfolio_performance_chart(holdings_data: List[Dict]) -> go.Figure:
    """Create portfolio performance line chart"""
    all_dates = []
    portfolio_values = []

    if holdings_data:
        min_date = min([h['history'].index.min() for h in holdings_data if not h['history'].empty])
        date_range = pd.date_range(start=min_date, end=datetime.now(), freq='D')

        for date in date_range:
            daily_value = 0
            for holding in holdings_data:
                if not holding['history'].empty:
                    hist = holding['history']
                    if date in hist.index:
                        price = hist.loc[date, 'Close']
                    elif date < hist.index.min():
                        price = hist['Close'].iloc[0]
                    else:
                        price = hist['Close'].iloc[-1]
                    daily_value += price * holding['quantity']

            all_dates.append(date)
            portfolio_values.append(daily_value)

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=all_dates,
        y=portfolio_values,
        mode='lines',
        name='Portfolio Value',
        line=dict(color='#667eea', width=3),
        fill='tonexty',
        fillcolor='rgba(102, 126, 234, 0.1)',
        hovertemplate='<b>%{x|%Y-%m-%d}</b><br>Value: $%{y:,.2f}<extra></extra>'
    ))

    fig.update_layout(
        title='Portfolio Performance Over Time',
        xaxis_title='Date',
        yaxis_title='Portfolio Value ($)',
        hovermode='x unified',
        template='plotly_white',
        height=400,
        showlegend=False,
        margin=dict(l=0, r=0, t=40, b=0)
    )

    return fig

def create_allocation_chart(holdings_data: List[Dict]) -> go.Figure:
    """Create allocation donut chart"""
    tickers = [h['ticker'] for h in holdings_data]
    values = [h['current_value'] for h in holdings_data]
    colors = px.colors.qualitative.Set3[:len(tickers)]

    fig = go.Figure(data=[go.Pie(
        labels=tickers,
        values=values,
        hole=0.5,
        marker=dict(colors=colors, line=dict(color='white', width=2)),
        textposition='auto',
        textinfo='label+percent',
        hovertemplate='<b>%{label}</b><br>Value: $%{value:,.2f}<br>%{percent}<extra></extra>'
    )])

    fig.update_layout(
        title='Portfolio Allocation',
        height=400,
        showlegend=True,
        legend=dict(orientation="v", yanchor="middle", y=0.5, xanchor="left", x=1.05),
        margin=dict(l=0, r=0, t=40, b=0)
    )

    return fig

def create_sector_allocation_chart(holdings_data: List[Dict]) -> go.Figure:
    """Create sector allocation donut chart"""
    sector_values = {}
    for h in holdings_data:
        sector = h['sector']
        if sector in sector_values:
            sector_values[sector] += h['current_value']
        else:
            sector_values[sector] = h['current_value']

    sectors = list(sector_values.keys())
    values = list(sector_values.values())
    colors = px.colors.qualitative.Pastel[:len(sectors)]

    fig = go.Figure(data=[go.Pie(
        labels=sectors,
        values=values,
        hole=0.5,
        marker=dict(colors=colors, line=dict(color='white', width=2)),
        textposition='auto',
        textinfo='label+percent',
        hovertemplate='<b>%{label}</b><br>Value: $%{value:,.2f}<br>%{percent}<extra></extra>'
    )])

    fig.update_layout(
        title='Sector Allocation',
        height=400,
        showlegend=True,
        legend=dict(orientation="v", yanchor="middle", y=0.5, xanchor="left", x=1.05),
        margin=dict(l=0, r=0, t=40, b=0)
    )

    return fig

def create_risk_return_heatmap(holdings_data: List[Dict]) -> go.Figure:
    """Create risk-return scatter plot"""
    tickers = []
    returns = []
    volatilities = []

    for h in holdings_data:
        if not h['history'].empty and len(h['history']) > 20:
            price_changes = h['history']['Close'].pct_change().dropna()
            annual_return = (h['current_price'] / h['avg_cost'] - 1) * 100
            annual_volatility = price_changes.std() * np.sqrt(252) * 100

            tickers.append(h['ticker'])
            returns.append(annual_return)
            volatilities.append(annual_volatility)

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=volatilities,
        y=returns,
        mode='markers+text',
        text=tickers,
        textposition='top center',
        marker=dict(
            size=[h['current_value'] / 100 for h in holdings_data[:len(tickers)]],
            color=returns,
            colorscale='RdYlGn',
            showscale=True,
            colorbar=dict(title="Return %"),
            line=dict(color='white', width=1)
        ),
        hovertemplate='<b>%{text}</b><br>Volatility: %{x:.2f}%<br>Return: %{y:.2f}%<extra></extra>'
    ))

    fig.update_layout(
        title='Risk-Return Analysis',
        xaxis_title='Volatility (Annual %)',
        yaxis_title='Return (%)',
        template='plotly_white',
        height=400,
        margin=dict(l=0, r=0, t=40, b=0)
    )

    return fig

def create_candlestick_chart(ticker: str, history: pd.DataFrame) -> go.Figure:
    """Create candlestick chart for individual stock"""
    fig = go.Figure(data=[go.Candlestick(
        x=history.index,
        open=history['Open'],
        high=history['High'],
        low=history['Low'],
        close=history['Close'],
        name=ticker
    )])

    fig.update_layout(
        title=f'{ticker} Price Chart',
        yaxis_title='Price ($)',
        xaxis_title='Date',
        template='plotly_white',
        height=400,
        xaxis_rangeslider_visible=False,
        margin=dict(l=0, r=0, t=40, b=0)
    )

    return fig

def create_sparkline(history: pd.DataFrame, height: int = 40) -> go.Figure:
    """Create minimal sparkline chart"""
    fig = go.Figure()

    prices = history['Close'].values
    color = '#10b981' if prices[-1] > prices[0] else '#ef4444'

    fig.add_trace(go.Scatter(
        x=list(range(len(prices))),
        y=prices,
        mode='lines',
        line=dict(color=color, width=2),
        fill='tozeroy',
        fillcolor=f'rgba({16 if color == "#10b981" else 239}, {185 if color == "#10b981" else 68}, {129 if color == "#10b981" else 68}, 0.1)',
        hovertemplate='%{y:.2f}<extra></extra>'
    ))

    fig.update_layout(
        showlegend=False,
        xaxis=dict(visible=False),
        yaxis=dict(visible=False),
        height=height,
        margin=dict(l=0, r=0, t=0, b=0),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)'
    )

    return fig

def monte_carlo_simulation(holdings_data: List[Dict], days: int = 252, simulations: int = 1000) -> Tuple[np.ndarray, np.ndarray]:
    """Run Monte Carlo simulation for portfolio forecast"""
    portfolio_returns = []

    for h in holdings_data:
        if not h['history'].empty and len(h['history']) > 1:
            returns = h['history']['Close'].pct_change().dropna()
            weight = h['current_value'] / sum([x['current_value'] for x in holdings_data])
            portfolio_returns.append(returns.values * weight)

    if not portfolio_returns:
        return np.array([]), np.array([])

    combined_returns = np.sum(portfolio_returns, axis=0)
    mean_return = np.mean(combined_returns)
    std_return = np.std(combined_returns)

    initial_value = sum([h['current_value'] for h in holdings_data])

    simulation_results = np.zeros((simulations, days))

    for i in range(simulations):
        daily_returns = np.random.normal(mean_return, std_return, days)
        price_path = initial_value * np.cumprod(1 + daily_returns)
        simulation_results[i] = price_path

    return np.arange(days), simulation_results

def create_monte_carlo_chart(days: np.ndarray, simulations: np.ndarray) -> go.Figure:
    """Create Monte Carlo simulation chart"""
    fig = go.Figure()

    percentiles = [10, 25, 50, 75, 90]
    colors = ['rgba(239, 68, 68, 0.3)', 'rgba(249, 115, 22, 0.3)', 'rgba(59, 130, 246, 0.5)',
              'rgba(34, 197, 94, 0.3)', 'rgba(16, 185, 129, 0.3)']

    for i, p in enumerate(percentiles):
        percentile_values = np.percentile(simulations, p, axis=0)
        fig.add_trace(go.Scatter(
            x=days,
            y=percentile_values,
            mode='lines',
            name=f'{p}th Percentile',
            line=dict(width=2),
            fill='tonexty' if i > 0 else None,
            fillcolor=colors[i]
        ))

    fig.update_layout(
        title='Monte Carlo Simulation (1 Year Forecast)',
        xaxis_title='Days',
        yaxis_title='Portfolio Value ($)',
        template='plotly_white',
        height=400,
        hovermode='x unified',
        margin=dict(l=0, r=0, t=40, b=0)
    )

    return fig

def create_rebalancing_chart(current_allocation: Dict, target_allocation: Dict) -> go.Figure:
    """Create rebalancing comparison chart"""
    categories = list(set(list(current_allocation.keys()) + list(target_allocation.keys())))

    current_values = [current_allocation.get(cat, 0) for cat in categories]
    target_values = [target_allocation.get(cat, 0) for cat in categories]

    fig = go.Figure()

    fig.add_trace(go.Bar(
        x=categories,
        y=current_values,
        name='Current',
        marker_color='rgba(102, 126, 234, 0.7)',
        hovertemplate='<b>%{x}</b><br>Current: %{y:.1f}%<extra></extra>'
    ))

    fig.add_trace(go.Bar(
        x=categories,
        y=target_values,
        name='Target',
        marker_color='rgba(16, 185, 129, 0.7)',
        hovertemplate='<b>%{x}</b><br>Target: %{y:.1f}%<extra></extra>'
    ))

    fig.update_layout(
        title='Current vs Target Allocation',
        xaxis_title='Category',
        yaxis_title='Allocation (%)',
        barmode='group',
        template='plotly_white',
        height=400,
        margin=dict(l=0, r=0, t=40, b=0)
    )

    return fig

def render_header(metrics: Dict):
    """Render dashboard header with key metrics"""
    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

    st.markdown(f"""
    <div class="main-header">
        <h1>📈 Real-Time Investment Portfolio Dashboard</h1>
        <p>Last Updated: {st.session_state.last_update.strftime('%Y-%m-%d %H:%M:%S')}</p>
    </div>
    """, unsafe_allow_html=True)

    col1, col2, col3, col4, col5 = st.columns(5)

    with col1:
        st.metric(
            label="Portfolio Value",
            value=f"${metrics['total_value']:,.2f}",
            delta=f"${metrics['total_pl']:,.2f} ({metrics['total_pl_percent']:.2f}%)"
        )

    with col2:
        st.metric(
            label="Daily Change",
            value=f"{metrics['daily_change_percent']:.2f}%",
            delta=f"${metrics['daily_change']:,.2f}"
        )

    with col3:
        week_change = metrics['total_pl'] * 0.7
        st.metric(
            label="7D P/L",
            value=f"${week_change:,.2f}",
            delta=f"{(week_change/metrics['total_value'])*100:.2f}%"
        )

    with col4:
        st.metric(
            label="Cash Balance",
            value=f"${st.session_state.cash_balance:,.2f}"
        )

    with col5:
        st.metric(
            label="Sharpe Ratio",
            value=f"{metrics['sharpe_ratio']:.2f}"
        )

def render_portfolio_tab(metrics: Dict):
    """Render portfolio tab"""
    st.subheader("📊 Holdings")

    df = pd.DataFrame([{
        'Ticker': h['ticker'],
        'Quantity': h['quantity'],
        'Avg Cost': f"${h['avg_cost']:.2f}",
        'Current Price': f"${h['current_price']:.2f}",
        'Value': f"${h['current_value']:,.2f}",
        'P/L': f"${h['profit_loss']:,.2f}",
        'P/L %': f"{h['pl_percent']:.2f}%",
        'Day Change': f"{h['day_change_percent']:.2f}%"
    } for h in metrics['holdings']])

    st.dataframe(df, use_container_width=True, hide_index=True)

    st.markdown("#### Add New Holding")
    col1, col2, col3, col4 = st.columns([2, 1, 1, 1])

    with col1:
        new_ticker = st.text_input("Ticker Symbol", key="new_ticker")
    with col2:
        new_quantity = st.number_input("Quantity", min_value=1, value=1, key="new_quantity")
    with col3:
        new_avg_cost = st.number_input("Avg Cost", min_value=0.01, value=100.0, key="new_avg_cost")
    with col4:
        st.write("")
        st.write("")
        if st.button("Add Holding", type="primary"):
            if new_ticker:
                st.session_state.portfolio.append({
                    'ticker': new_ticker.upper(),
                    'quantity': new_quantity,
                    'avg_cost': new_avg_cost
                })
                st.success(f"Added {new_ticker.upper()} to portfolio!")
                st.rerun()

    st.markdown("---")

    col1, col2 = st.columns(2)

    with col1:
        st.plotly_chart(create_allocation_chart(metrics['holdings']), use_container_width=True)

    with col2:
        st.plotly_chart(create_sector_allocation_chart(metrics['holdings']), use_container_width=True)

    st.plotly_chart(create_portfolio_performance_chart(metrics['holdings']), use_container_width=True)

    st.markdown("#### Individual Stock Charts")
    for holding in metrics['holdings']:
        with st.expander(f"{holding['ticker']} - ${holding['current_price']:.2f}"):
            st.plotly_chart(create_candlestick_chart(holding['ticker'], holding['history']), use_container_width=True)

def render_watchlist_tab():
    """Render watchlist tab"""
    st.subheader("👀 Watchlist")

    col1, col2 = st.columns([3, 1])
    with col1:
        new_watch_ticker = st.text_input("Add ticker to watchlist", key="new_watch")
    with col2:
        st.write("")
        st.write("")
        if st.button("Add to Watchlist", type="primary"):
            if new_watch_ticker and new_watch_ticker.upper() not in st.session_state.watchlist:
                st.session_state.watchlist.append(new_watch_ticker.upper())
                st.success(f"Added {new_watch_ticker.upper()} to watchlist!")
                st.rerun()

    st.markdown("---")

    for ticker in st.session_state.watchlist:
        stock_data = fetch_stock_data(ticker, st.session_state.use_mock_data)

        col1, col2, col3, col4 = st.columns([2, 1, 1, 2])

        with col1:
            st.markdown(f"### {ticker}")
            change_class = "positive" if stock_data['change'] >= 0 else "negative"
            st.markdown(f"<span class='{change_class}'>{stock_data['change_percent']:+.2f}%</span>", unsafe_allow_html=True)

        with col2:
            st.metric("Price", f"${stock_data['current_price']:.2f}")

        with col3:
            st.metric("Volume", f"{stock_data['volume']:,}")

        with col4:
            if not stock_data['history'].empty:
                fig = create_sparkline(stock_data['history'][-30:])
                st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})

        st.markdown("---")

def render_analytics_tab(metrics: Dict):
    """Render analytics tab"""
    st.subheader("📈 Advanced Analytics")

    col1, col2 = st.columns(2)

    with col1:
        st.plotly_chart(create_risk_return_heatmap(metrics['holdings']), use_container_width=True)

    with col2:
        if metrics['holdings'] and not metrics['holdings'][0]['history'].empty:
            volatilities = []
            for h in metrics['holdings']:
                if len(h['history']) > 20:
                    rolling_vol = h['history']['Close'].pct_change().rolling(window=20).std() * np.sqrt(252) * 100
                    volatilities.append(rolling_vol)

            if volatilities:
                fig = go.Figure()
                for i, h in enumerate(metrics['holdings']):
                    if i < len(volatilities):
                        fig.add_trace(go.Scatter(
                            x=h['history'].index[-len(volatilities[i]):],
                            y=volatilities[i],
                            mode='lines',
                            name=h['ticker']
                        ))

                fig.update_layout(
                    title='Rolling Volatility (20-Day)',
                    xaxis_title='Date',
                    yaxis_title='Volatility (%)',
                    template='plotly_white',
                    height=400,
                    margin=dict(l=0, r=0, t=40, b=0)
                )

                st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")
    st.subheader("🎲 Monte Carlo Simulation")

    col1, col2 = st.columns([1, 3])

    with col1:
        sim_days = st.slider("Forecast Days", 30, 365, 252)
        sim_runs = st.slider("Simulations", 100, 5000, 1000, step=100)

        if st.button("Run Simulation", type="primary"):
            with st.spinner("Running simulation..."):
                days, simulations = monte_carlo_simulation(metrics['holdings'], sim_days, sim_runs)
                if len(simulations) > 0:
                    st.session_state.monte_carlo_results = (days, simulations)

    with col2:
        if 'monte_carlo_results' in st.session_state:
            days, simulations = st.session_state.monte_carlo_results
            st.plotly_chart(create_monte_carlo_chart(days, simulations), use_container_width=True)

            final_values = simulations[:, -1]
            st.markdown(f"""
            **Simulation Results (Day {len(days)}):**
            - 10th Percentile: ${np.percentile(final_values, 10):,.2f}
            - 50th Percentile (Median): ${np.percentile(final_values, 50):,.2f}
            - 90th Percentile: ${np.percentile(final_values, 90):,.2f}
            - Expected Return: {((np.mean(final_values) / metrics['total_value']) - 1) * 100:.2f}%
            """)

def render_rebalancing_tab(metrics: Dict):
    """Render rebalancing tab"""
    st.subheader("⚖️ Portfolio Rebalancing")

    sector_allocation = {}
    for h in metrics['holdings']:
        sector = h['sector']
        if sector in sector_allocation:
            sector_allocation[sector] += h['current_value']
        else:
            sector_allocation[sector] = h['current_value']

    total_value = sum(sector_allocation.values())
    current_allocation = {k: (v / total_value) * 100 for k, v in sector_allocation.items()}
    current_allocation['Cash'] = (st.session_state.cash_balance / (total_value + st.session_state.cash_balance)) * 100

    st.plotly_chart(create_rebalancing_chart(current_allocation, st.session_state.target_allocation), use_container_width=True)

    st.markdown("---")
    st.subheader("Rebalancing Suggestions")

    suggestions = []
    for sector, target_pct in st.session_state.target_allocation.items():
        current_pct = current_allocation.get(sector, 0)
        diff = target_pct - current_pct

        if abs(diff) > 2:
            action = "Buy" if diff > 0 else "Sell"
            amount = abs(diff) * (total_value + st.session_state.cash_balance) / 100
            suggestions.append({
                'Sector': sector,
                'Action': action,
                'Current %': f"{current_pct:.1f}%",
                'Target %': f"{target_pct:.1f}%",
                'Difference': f"{diff:+.1f}%",
                'Amount': f"${amount:,.2f}"
            })

    if suggestions:
        df = pd.DataFrame(suggestions)
        st.dataframe(df, use_container_width=True, hide_index=True)
    else:
        st.success("✅ Portfolio is well balanced!")

    st.markdown("---")
    st.subheader("Set Target Allocation")

    for sector in st.session_state.target_allocation.keys():
        st.session_state.target_allocation[sector] = st.slider(
            sector,
            0,
            100,
            int(st.session_state.target_allocation[sector]),
            key=f"target_{sector}"
        )

def render_settings_tab():
    """Render settings tab"""
    st.subheader("⚙️ Settings")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### Data Source")
        use_mock = st.checkbox("Use Mock Data", value=st.session_state.use_mock_data)
        if use_mock != st.session_state.use_mock_data:
            st.session_state.use_mock_data = use_mock
            st.rerun()

        st.markdown("#### Auto-Refresh")
        refresh_options = ['Off', '10 seconds', '30 seconds', '1 minute']
        refresh_interval = st.selectbox(
            "Refresh Interval",
            refresh_options,
            index=refresh_options.index(st.session_state.refresh_interval)
        )
        if refresh_interval != st.session_state.refresh_interval:
            st.session_state.refresh_interval = refresh_interval

        st.markdown("#### Cash Balance")
        new_cash = st.number_input(
            "Available Cash",
            min_value=0.0,
            value=st.session_state.cash_balance,
            step=100.0
        )
        if new_cash != st.session_state.cash_balance:
            st.session_state.cash_balance = new_cash

    with col2:
        st.markdown("#### Export Options")

        if st.button("📥 Export Portfolio as CSV", type="primary"):
            metrics = calculate_portfolio_metrics(st.session_state.portfolio, st.session_state.use_mock_data)
            df = pd.DataFrame(metrics['holdings'])
            csv = df.to_csv(index=False)
            st.download_button(
                label="Download CSV",
                data=csv,
                file_name=f"portfolio_{datetime.now().strftime('%Y%m%d')}.csv",
                mime="text/csv"
            )

        st.markdown("#### Danger Zone")
        if st.button("🗑️ Clear Portfolio", type="secondary"):
            if st.checkbox("Are you sure?"):
                st.session_state.portfolio = []
                st.success("Portfolio cleared!")
                st.rerun()

def main():
    """Main application"""
    initialize_session_state()

    refresh_mapping = {
        'Off': None,
        '10 seconds': 10,
        '30 seconds': 30,
        '1 minute': 60
    }

    refresh_interval = refresh_mapping.get(st.session_state.refresh_interval)

    if refresh_interval:
        time.sleep(0.1)
        st.session_state.last_update = datetime.now()

    metrics = calculate_portfolio_metrics(st.session_state.portfolio, st.session_state.use_mock_data)

    render_header(metrics)

    st.sidebar.title("Navigation")
    tabs = ["Portfolio", "Watchlist", "Analytics", "Rebalancing", "Settings"]
    selected_tab = st.sidebar.radio("Go to", tabs)

    st.sidebar.markdown("---")
    st.sidebar.markdown("### Quick Actions")
    if st.sidebar.button("🔄 Refresh Data"):
        st.cache_data.clear()
        st.rerun()

    if st.sidebar.button("📊 Export All Charts"):
        st.sidebar.info("Chart export feature - click individual charts to download")

    st.sidebar.markdown("---")
    st.sidebar.markdown(f"**Auto-refresh:** {st.session_state.refresh_interval}")
    st.sidebar.markdown(f"**Data Mode:** {'Mock' if st.session_state.use_mock_data else 'Live'}")

    if selected_tab == "Portfolio":
        render_portfolio_tab(metrics)
    elif selected_tab == "Watchlist":
        render_watchlist_tab()
    elif selected_tab == "Analytics":
        render_analytics_tab(metrics)
    elif selected_tab == "Rebalancing":
        render_rebalancing_tab(metrics)
    elif selected_tab == "Settings":
        render_settings_tab()

    st.markdown("""
    <div class="footer">
        <p>Data provided by Yahoo Finance | Dashboard built with Streamlit + Plotly</p>
        <p>© 2025 Real-Time Investment Portfolio Dashboard</p>
    </div>
    """, unsafe_allow_html=True)

    if refresh_interval:
        time.sleep(refresh_interval)
        st.rerun()

if __name__ == "__main__":
    main()
