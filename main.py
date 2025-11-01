import streamlit as st
import yfinance as yf
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
import json
import os
import io
import base64
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
import requests
from xhtml2pdf import pisa
from groq import Groq
import feedparser
from dotenv import load_dotenv

load_dotenv()

st.set_page_config(
    page_title="Inclusive Investment Portfolio Builder",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="collapsed"
)

st.markdown("""
<style>
    .stApp {
        background-color: #0e1117;
        color: #ffffff;
    }
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #ffffff;
        text-align: center;
        margin-bottom: 1rem;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .metric-card {
        background: linear-gradient(135deg, #1f2937 0%, #374151 100%);
        padding: 1.5rem;
        border-radius: 12px;
        border-left: 5px solid #3b82f6;
        margin-bottom: 15px;
        box-shadow: 0 4px 16px rgba(0,0,0,0.2);
        color: #ffffff;
    }
    .stButton > button {
        background-color: #3b82f6;
        color: #ffffff;
        border-radius: 8px;
    }
    div[data-testid="stMetric"] {
        background-color: #1f2937;
        padding: 1rem;
        border-radius: 8px;
        border: 1px solid #4b5563;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_data(ttl=300)
def fetch_market_data():
    """Fetch live market data with caching"""
    try:
        indices = {
            '^GSPC': 'S&P 500',
            '^DJI': 'Dow Jones',
            '^IXIC': 'NASDAQ',
            '^FTSE': 'FTSE 100',
            '^N225': 'Nikkei 225',
            '^HSI': 'Hang Seng'
        }

        data = {}
        for ticker, name in indices.items():
            try:
                stock = yf.Ticker(ticker)
                hist = stock.history(period='5d')
                if not hist.empty:
                    current = hist['Close'].iloc[-1]
                    prev = hist['Close'].iloc[-2] if len(hist) > 1 else current
                    change = ((current - prev) / prev) * 100
                    data[name] = {'price': current, 'change': change}
            except:
                data[name] = {'price': 0, 'change': 0}

        return data
    except Exception as e:
        st.warning(f"Market data fetch error: {str(e)}")
        return {}

@st.cache_data(ttl=300)
def fetch_sector_data():
    """Fetch sector performance"""
    try:
        sectors = {
            'XLK': 'Technology',
            'XLF': 'Financials',
            'XLV': 'Healthcare',
            'XLE': 'Energy',
            'XLI': 'Industrials',
            'XLP': 'Consumer Staples',
            'XLY': 'Consumer Discretionary',
            'XLU': 'Utilities',
            'XLRE': 'Real Estate',
            'XLB': 'Materials'
        }

        data = {}
        for ticker, name in sectors.items():
            try:
                stock = yf.Ticker(ticker)
                hist = stock.history(period='5d')
                if not hist.empty:
                    current = hist['Close'].iloc[-1]
                    prev = hist['Close'].iloc[-2] if len(hist) > 1 else current
                    change = ((current - prev) / prev) * 100
                    data[name] = change
            except:
                data[name] = 0

        return data
    except Exception as e:
        st.warning(f"Sector data fetch error: {str(e)}")
        return {}

@st.cache_data(ttl=300)
def fetch_top_companies():
    """Fetch top companies data"""
    try:
        tickers = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 'TSLA', 'META', 'BRK-B']
        data = []

        for ticker in tickers:
            try:
                stock = yf.Ticker(ticker)
                info = stock.info
                hist = stock.history(period='5d')

                if not hist.empty:
                    current = hist['Close'].iloc[-1]
                    prev = hist['Close'].iloc[-2] if len(hist) > 1 else current
                    change = ((current - prev) / prev) * 100

                    data.append({
                        'Ticker': ticker,
                        'Price': f"${current:.2f}",
                        'Change %': f"{change:+.2f}%",
                        'Market Cap': f"${info.get('marketCap', 0) / 1e9:.1f}B"
                    })
            except:
                continue

        return pd.DataFrame(data)
    except Exception as e:
        st.warning(f"Company data fetch error: {str(e)}")
        return pd.DataFrame()

@st.cache_data(ttl=600)
def fetch_market_news():
    """Fetch market news via Yahoo Finance RSS"""
    try:
        feed = feedparser.parse('https://finance.yahoo.com/news/rssindex')
        news = []

        for entry in feed.entries[:10]:
            news.append({
                'title': entry.title,
                'link': entry.link,
                'published': entry.published if hasattr(entry, 'published') else ''
            })

        return news
    except Exception as e:
        st.warning(f"News fetch error: {str(e)}")
        return []

def analyze_news_sentiment(news_items: List[Dict], groq_api_key: Optional[str]) -> Dict:
    """Analyze news sentiment using LLaMA"""
    if not groq_api_key or not news_items:
        return {
            'sentiment': 'Neutral',
            'score': 50,
            'summary': 'Sentiment analysis unavailable'
        }

    try:
        client = Groq(api_key=groq_api_key)

        news_text = "\n".join([f"- {item['title']}" for item in news_items[:5]])

        prompt = f"""Analyze the sentiment of these market news headlines:

{news_text}

Return ONLY valid JSON with keys: sentiment (Positive/Neutral/Negative), score (0-100), summary (2 sentences)
Example: {{"sentiment": "Positive", "score": 65, "summary": "Markets show..."}}"""

        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3
        )

        result = json.loads(response.choices[0].message.content.strip())
        return result
    except:
        return {
            'sentiment': 'Neutral',
            'score': 50,
            'summary': 'Unable to analyze sentiment at this time'
        }

def generate_ai_portfolio(profile: Dict, tickers: List[str], esg_enabled: bool, groq_api_key: Optional[str]) -> Dict:
    """Generate AI portfolio allocation using LLaMA"""
    fallback = {
        'allocations': {t: round(100 / len(tickers), 2) for t in tickers} if tickers else {},
        'expected_return': 0.08,
        'volatility': 0.15,
        'diversification_index': 70,
        'risk_level': profile.get('risk_profile', 'Moderate'),
        'explanation': 'Using deterministic fallback allocation'
    }

    if not groq_api_key or not tickers:
        return fallback

    try:
        client = Groq(api_key=groq_api_key)

        esg_note = " CRITICAL: User requested ESG/inclusive options (ESGU, SHE, SUSA). You MUST include these if available in the ticker list." if esg_enabled else ""

        prompt = f"""You are a professional portfolio manager. Create an optimal portfolio allocation.

User Profile:
- Risk Profile: {profile.get('risk_profile', 'Moderate')}
- Time Horizon: {profile.get('time_horizon', 10)} years
- Investment Amount: ${profile.get('capital', 10000)}
- Age: {profile.get('age', 35)}

Available Tickers: {', '.join(tickers)}
{esg_note}

Return ONLY valid JSON with these exact keys:
{{
  "allocations": {{"TICKER1": 30.5, "TICKER2": 25.0, ...}},
  "expected_return": 0.08,
  "volatility": 0.15,
  "diversification_index": 75,
  "risk_level": "Moderate",
  "explanation": "Brief 2-sentence explanation"
}}

Allocations must sum to 100. Use realistic return (0.04-0.12) and volatility (0.05-0.25) estimates."""

        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
            max_tokens=1000
        )

        content = response.choices[0].message.content.strip()

        if content.startswith("```"):
            content = content.split("```")[1]
            if content.startswith("json"):
                content = content[4:]

        result = json.loads(content)

        required_keys = ['allocations', 'expected_return', 'volatility', 'diversification_index', 'risk_level', 'explanation']
        if not all(k in result for k in required_keys):
            return fallback

        total = sum(result['allocations'].values())
        if abs(total - 100) > 0.1:
            result['allocations'] = {k: (v / total) * 100 for k, v in result['allocations'].items()}

        return result
    except Exception as e:
        st.warning(f"AI allocation failed: {str(e)}")
        return fallback

def fetch_historical_prices(tickers: List[str], period: str = '1y') -> pd.DataFrame:
    """Fetch historical prices for multiple tickers"""
    try:
        data = yf.download(tickers, period=period, progress=False)['Close']
        if isinstance(data, pd.Series):
            data = data.to_frame(name=tickers[0])
        return data.fillna(method='ffill').fillna(method='bfill')
    except Exception as e:
        st.warning(f"Historical data fetch error: {str(e)}")
        return pd.DataFrame()

def plot_allocation_pie(allocations: Dict) -> go.Figure:
    """Create allocation pie chart"""
    fig = go.Figure(data=[go.Pie(
        labels=list(allocations.keys()),
        values=list(allocations.values()),
        hole=0.4,
        textposition='inside',
        textinfo='percent+label'
    )])

    fig.update_layout(
        title="Portfolio Allocation",
        height=400,
        paper_bgcolor='#1f2937',
        plot_bgcolor='#1f2937',
        font_color='white',
        showlegend=True
    )

    return fig

def plot_ai_vs_manual(ai_alloc: Dict, manual_alloc: Dict) -> go.Figure:
    """Compare AI vs manual allocation"""
    tickers = list(set(list(ai_alloc.keys()) + list(manual_alloc.keys())))

    ai_vals = [ai_alloc.get(t, 0) for t in tickers]
    manual_vals = [manual_alloc.get(t, 0) for t in tickers]

    fig = go.Figure(data=[
        go.Bar(name='AI Recommended', x=tickers, y=ai_vals, marker_color='#3b82f6'),
        go.Bar(name='Manual', x=tickers, y=manual_vals, marker_color='#10b981')
    ])

    fig.update_layout(
        title="AI vs Manual Allocation",
        xaxis_title="Ticker",
        yaxis_title="Allocation %",
        barmode='group',
        height=400,
        paper_bgcolor='#1f2937',
        plot_bgcolor='#1f2937',
        font_color='white'
    )

    return fig

def plot_historical_prices(prices_df: pd.DataFrame) -> go.Figure:
    """Plot historical price trends"""
    fig = go.Figure()

    for col in prices_df.columns:
        fig.add_trace(go.Scatter(
            x=prices_df.index,
            y=prices_df[col],
            mode='lines',
            name=col
        ))

    fig.update_layout(
        title="Historical Price Trends",
        xaxis_title="Date",
        yaxis_title="Price ($)",
        height=400,
        paper_bgcolor='#1f2937',
        plot_bgcolor='#1f2937',
        font_color='white',
        hovermode='x unified'
    )

    return fig

def plot_risk_return_scatter(allocations: Dict, prices_df: pd.DataFrame) -> go.Figure:
    """Risk vs return scatter plot"""
    returns = prices_df.pct_change().dropna()

    data_points = []
    for ticker in allocations.keys():
        if ticker in returns.columns:
            annual_return = returns[ticker].mean() * 252
            volatility = returns[ticker].std() * np.sqrt(252)
            allocation = allocations[ticker]

            data_points.append({
                'ticker': ticker,
                'return': annual_return * 100,
                'risk': volatility * 100,
                'allocation': allocation
            })

    df = pd.DataFrame(data_points)

    if df.empty:
        fig = go.Figure()
        fig.add_annotation(text="Insufficient data", x=0.5, y=0.5, showarrow=False)
    else:
        fig = px.scatter(
            df,
            x='risk',
            y='return',
            size='allocation',
            text='ticker',
            labels={'risk': 'Volatility (%)', 'return': 'Expected Return (%)'},
            title="Risk vs Return Profile"
        )

        fig.update_traces(textposition='top center')

    fig.update_layout(
        height=400,
        paper_bgcolor='#1f2937',
        plot_bgcolor='#1f2937',
        font_color='white'
    )

    return fig

def plot_volatility_heatmap(prices_df: pd.DataFrame) -> go.Figure:
    """Volatility heatmap"""
    returns = prices_df.pct_change().dropna()
    corr = returns.corr()

    fig = go.Figure(data=go.Heatmap(
        z=corr.values,
        x=corr.columns,
        y=corr.columns,
        colorscale='RdYlGn',
        zmid=0
    ))

    fig.update_layout(
        title="Asset Correlation Heatmap",
        height=400,
        paper_bgcolor='#1f2937',
        plot_bgcolor='#1f2937',
        font_color='white'
    )

    return fig

def plot_growth_projections(capital: float, expected_return: float, years: int = 10) -> go.Figure:
    """Growth projection chart"""
    time = list(range(years + 1))
    conservative = [capital * ((1 + expected_return * 0.7) ** t) for t in time]
    expected = [capital * ((1 + expected_return) ** t) for t in time]
    optimistic = [capital * ((1 + expected_return * 1.3) ** t) for t in time]

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=time, y=conservative, name='Conservative', line=dict(dash='dash', color='#f59e0b')))
    fig.add_trace(go.Scatter(x=time, y=expected, name='Expected', line=dict(width=3, color='#3b82f6')))
    fig.add_trace(go.Scatter(x=time, y=optimistic, name='Optimistic', line=dict(dash='dash', color='#10b981')))

    fig.update_layout(
        title=f"Portfolio Growth Projection ({years} Years)",
        xaxis_title="Years",
        yaxis_title="Portfolio Value ($)",
        height=400,
        paper_bgcolor='#1f2937',
        plot_bgcolor='#1f2937',
        font_color='white'
    )

    return fig

def plot_benchmark_comparison(portfolio_alloc: Dict, prices_df: pd.DataFrame, benchmark: str = 'SPY') -> go.Figure:
    """Benchmark comparison"""
    try:
        bench_data = yf.download(benchmark, period='1y', progress=False)['Close']
        bench_returns = (bench_data / bench_data.iloc[0]) * 100

        portfolio_value = pd.Series(0, index=prices_df.index)
        for ticker, weight in portfolio_alloc.items():
            if ticker in prices_df.columns:
                normalized = (prices_df[ticker] / prices_df[ticker].iloc[0]) * (weight / 100) * 100
                portfolio_value += normalized

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=portfolio_value.index, y=portfolio_value.values, name='Your Portfolio', line=dict(width=3)))
        fig.add_trace(go.Scatter(x=bench_returns.index, y=bench_returns.values, name=benchmark, line=dict(dash='dash')))

        fig.update_layout(
            title=f"Cumulative Returns vs {benchmark}",
            xaxis_title="Date",
            yaxis_title="Cumulative Return (%)",
            height=400,
            paper_bgcolor='#1f2937',
            plot_bgcolor='#1f2937',
            font_color='white'
        )

        return fig
    except Exception as e:
        st.warning(f"Benchmark comparison error: {str(e)}")
        fig = go.Figure()
        fig.add_annotation(text="Benchmark data unavailable", x=0.5, y=0.5, showarrow=False)
        fig.update_layout(height=400, paper_bgcolor='#1f2937', font_color='white')
        return fig

def generate_pdf_report(portfolio_data: Dict, charts: List[go.Figure]) -> bytes:
    """Generate PDF report using xhtml2pdf"""
    try:
        html_content = f"""
        <html>
        <head>
            <style>
                body {{ font-family: Arial; margin: 20px; }}
                h1 {{ color: #3b82f6; }}
                .metric {{ background: #f0f0f0; padding: 10px; margin: 10px 0; border-radius: 5px; }}
            </style>
        </head>
        <body>
            <h1>Investment Portfolio Report</h1>
            <p>Generated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</p>

            <h2>Portfolio Summary</h2>
            <div class="metric">
                <strong>Investment Amount:</strong> ${portfolio_data.get('capital', 0):,.2f}<br>
                <strong>Expected Return:</strong> {portfolio_data.get('expected_return', 0):.2%}<br>
                <strong>Risk Level:</strong> {portfolio_data.get('risk_level', 'N/A')}<br>
                <strong>Diversification Index:</strong> {portfolio_data.get('diversification_index', 0)}/100
            </div>

            <h2>Allocation</h2>
            <table border="1" cellpadding="5" style="border-collapse: collapse; width: 100%;">
                <tr><th>Ticker</th><th>Allocation %</th></tr>
                {''.join([f"<tr><td>{k}</td><td>{v:.2f}%</td></tr>" for k, v in portfolio_data.get('allocations', {}).items()])}
            </table>

            <h2>AI Commentary</h2>
            <p>{portfolio_data.get('explanation', 'No commentary available')}</p>
        </body>
        </html>
        """

        output = io.BytesIO()
        pisa.CreatePDF(html_content, dest=output)
        return output.getvalue()
    except Exception as e:
        st.error(f"PDF generation failed: {str(e)}")
        return b""

def main():
    st.markdown('<h1 class="main-header">📊 Inclusive Investment Portfolio Builder</h1>', unsafe_allow_html=True)

    groq_api_key = os.getenv("GROQ_API_KEY")

    if not groq_api_key:
        st.warning("⚠️ GROQ_API_KEY not found. AI features will use fallback mode.")

    st.markdown("---")
    st.header("🌍 Live Market Overview")

    col1, col2 = st.columns([3, 1])
    with col2:
        if st.button("🔄 Refresh Data", use_container_width=True):
            st.cache_data.clear()
            st.rerun()

    market_data = fetch_market_data()
    if market_data:
        cols = st.columns(6)
        for idx, (name, data) in enumerate(market_data.items()):
            with cols[idx % 6]:
                st.metric(
                    name,
                    f"${data['price']:,.2f}",
                    f"{data['change']:+.2f}%"
                )

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("📈 Sector Performance")
        sector_data = fetch_sector_data()
        if sector_data:
            sector_df = pd.DataFrame(list(sector_data.items()), columns=['Sector', 'Change %'])
            sector_df = sector_df.sort_values('Change %', ascending=False)

            fig_sector = px.bar(
                sector_df,
                x='Change %',
                y='Sector',
                orientation='h',
                color='Change %',
                color_continuous_scale=['red', 'yellow', 'green']
            )
            fig_sector.update_layout(
                height=400,
                paper_bgcolor='#1f2937',
                plot_bgcolor='#1f2937',
                font_color='white'
            )
            st.plotly_chart(fig_sector, use_container_width=True)

    with col2:
        st.subheader("🏢 Top Companies")
        companies_df = fetch_top_companies()
        if not companies_df.empty:
            st.dataframe(companies_df, use_container_width=True, hide_index=True)

    st.subheader("📰 Market News & Sentiment")
    news = fetch_market_news()
    sentiment = analyze_news_sentiment(news, groq_api_key)

    col1, col2 = st.columns([1, 2])
    with col1:
        st.metric("Market Sentiment", sentiment['sentiment'], f"{sentiment['score']}/100")
        st.caption(sentiment['summary'])

    with col2:
        if news:
            with st.expander("View Latest News"):
                for item in news[:5]:
                    st.markdown(f"**{item['title']}**")
                    st.caption(item.get('published', ''))
                    st.markdown("---")

    st.markdown("---")
    st.header("🤖 AI-Powered Portfolio Builder")

    if 'wizard_step' not in st.session_state:
        st.session_state.wizard_step = 1
    if 'profile_data' not in st.session_state:
        st.session_state.profile_data = {}
    if 'selected_tickers' not in st.session_state:
        st.session_state.selected_tickers = []
    if 'ai_allocation' not in st.session_state:
        st.session_state.ai_allocation = None
    if 'manual_allocation' not in st.session_state:
        st.session_state.manual_allocation = {}

    progress = st.session_state.wizard_step / 4
    st.progress(progress)
    st.caption(f"Step {st.session_state.wizard_step} of 4")

    if st.session_state.wizard_step == 1:
        st.subheader("Step 1: Investment Profile")

        col1, col2 = st.columns(2)
        with col1:
            risk_profile = st.selectbox(
                "Risk Profile",
                ["Conservative", "Moderate", "Aggressive"]
            )
            capital = st.number_input(
                "Investment Amount ($)",
                min_value=1000.0,
                value=10000.0,
                step=1000.0
            )

        with col2:
            time_horizon = st.slider(
                "Time Horizon (years)",
                1, 30, 10
            )
            age = st.number_input(
                "Your Age",
                min_value=18,
                max_value=80,
                value=35
            )

        if st.button("Next →", type="primary"):
            st.session_state.profile_data = {
                'risk_profile': risk_profile,
                'capital': capital,
                'time_horizon': time_horizon,
                'age': age
            }
            st.session_state.wizard_step = 2
            st.rerun()

    elif st.session_state.wizard_step == 2:
        st.subheader("Step 2: Select Investment Tickers")

        esg_options = ['ESGU', 'SHE', 'SUSA', 'VFTAX', 'NACP']
        traditional_options = ['SPY', 'QQQ', 'VTI', 'BND', 'GLD', 'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA']

        col1, col2 = st.columns(2)
        with col1:
            include_esg = st.checkbox("Include ESG/Inclusive Options", value=True)
            st.caption("ESG: ESGU, SHE, SUSA, VFTAX, NACP")

        with col2:
            st.caption("Select at least 3 tickers")

        available_tickers = esg_options + traditional_options if include_esg else traditional_options

        selected = st.multiselect(
            "Choose Tickers",
            available_tickers,
            default=available_tickers[:5] if include_esg else traditional_options[:5]
        )

        col1, col2 = st.columns(2)
        with col1:
            if st.button("← Back"):
                st.session_state.wizard_step = 1
                st.rerun()

        with col2:
            if st.button("Next →", type="primary", disabled=len(selected) < 3):
                st.session_state.selected_tickers = selected
                st.session_state.profile_data['esg_enabled'] = include_esg
                st.session_state.wizard_step = 3
                st.rerun()

    elif st.session_state.wizard_step == 3:
        st.subheader("Step 3: AI Recommendation")

        if st.session_state.ai_allocation is None:
            with st.spinner("Generating AI allocation..."):
                ai_result = generate_ai_portfolio(
                    st.session_state.profile_data,
                    st.session_state.selected_tickers,
                    st.session_state.profile_data.get('esg_enabled', False),
                    groq_api_key
                )
                st.session_state.ai_allocation = ai_result

        ai_alloc = st.session_state.ai_allocation

        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Expected Return", f"{ai_alloc['expected_return']:.2%}")
        with col2:
            st.metric("Volatility", f"{ai_alloc['volatility']:.2%}")
        with col3:
            st.metric("Diversification", f"{ai_alloc['diversification_index']}/100")

        st.info(f"**AI Commentary:** {ai_alloc['explanation']}")

        fig_pie = plot_allocation_pie(ai_alloc['allocations'])
        st.plotly_chart(fig_pie, use_container_width=True)

        st.subheader("Adjust Allocation (Optional)")
        manual_alloc = {}

        for ticker in st.session_state.selected_tickers:
            default_val = ai_alloc['allocations'].get(ticker, 0)
            manual_alloc[ticker] = st.slider(
                f"{ticker} (%)",
                0.0, 100.0,
                float(default_val),
                0.1,
                key=f"slider_{ticker}"
            )

        total = sum(manual_alloc.values())
        if abs(total - 100) > 0.1:
            st.warning(f"⚠️ Total: {total:.1f}%. Adjust to 100%.")
        else:
            st.success(f"✓ Total: {total:.1f}%")
            st.session_state.manual_allocation = manual_alloc

        col1, col2 = st.columns(2)
        with col1:
            if st.button("← Back"):
                st.session_state.wizard_step = 2
                st.rerun()

        with col2:
            if st.button("Next →", type="primary", disabled=abs(total - 100) > 0.1):
                st.session_state.wizard_step = 4
                st.rerun()

    elif st.session_state.wizard_step == 4:
        st.subheader("Step 4: Final Review & Analysis")

        final_alloc = st.session_state.manual_allocation if st.session_state.manual_allocation else st.session_state.ai_allocation['allocations']

        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Investment", f"${st.session_state.profile_data['capital']:,.0f}")
        with col2:
            st.metric("Risk Profile", st.session_state.profile_data['risk_profile'])
        with col3:
            st.metric("Time Horizon", f"{st.session_state.profile_data['time_horizon']} years")
        with col4:
            st.metric("Assets", len(final_alloc))

        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "📊 Allocation",
            "📈 Historical",
            "🎯 Risk/Return",
            "🔗 Correlation",
            "📉 Growth"
        ])

        with tab1:
            col1, col2 = st.columns(2)
            with col1:
                fig_pie = plot_allocation_pie(final_alloc)
                st.plotly_chart(fig_pie, use_container_width=True)
            with col2:
                if st.session_state.manual_allocation:
                    fig_compare = plot_ai_vs_manual(
                        st.session_state.ai_allocation['allocations'],
                        st.session_state.manual_allocation
                    )
                    st.plotly_chart(fig_compare, use_container_width=True)

        with tab2:
            prices_df = fetch_historical_prices(list(final_alloc.keys()))
            if not prices_df.empty:
                fig_hist = plot_historical_prices(prices_df)
                st.plotly_chart(fig_hist, use_container_width=True)

        with tab3:
            if not prices_df.empty:
                fig_risk = plot_risk_return_scatter(final_alloc, prices_df)
                st.plotly_chart(fig_risk, use_container_width=True)

        with tab4:
            if not prices_df.empty:
                fig_corr = plot_volatility_heatmap(prices_df)
                st.plotly_chart(fig_corr, use_container_width=True)

        with tab5:
            fig_growth = plot_growth_projections(
                st.session_state.profile_data['capital'],
                st.session_state.ai_allocation['expected_return'],
                st.session_state.profile_data['time_horizon']
            )
            st.plotly_chart(fig_growth, use_container_width=True)

        st.subheader("📊 Benchmark Comparison")
        benchmark_ticker = st.selectbox(
            "Select Benchmark",
            ["SPY", "ESGU", "^NSEI"],
            format_func=lambda x: {"SPY": "S&P 500", "ESGU": "ESG S&P 500", "^NSEI": "NIFTY 50"}[x]
        )

        if not prices_df.empty:
            fig_bench = plot_benchmark_comparison(final_alloc, prices_df, benchmark_ticker)
            st.plotly_chart(fig_bench, use_container_width=True)

            if groq_api_key:
                with st.spinner("Generating AI commentary..."):
                    try:
                        client = Groq(api_key=groq_api_key)
                        prompt = f"""Compare this portfolio to {benchmark_ticker}. Portfolio: {final_alloc}.
                        Provide 2-3 sentences of professional commentary."""

                        response = client.chat.completions.create(
                            model="llama-3.3-70b-versatile",
                            messages=[{"role": "user", "content": prompt}],
                            temperature=0.3
                        )

                        st.info(f"**AI Commentary:** {response.choices[0].message.content.strip()}")
                    except:
                        pass

        st.subheader("📥 Export Options")
        col1, col2 = st.columns(2)

        with col1:
            csv_data = pd.DataFrame(list(final_alloc.items()), columns=['Ticker', 'Allocation %'])
            csv = csv_data.to_csv(index=False)
            st.download_button(
                "Download CSV",
                csv,
                "portfolio_allocation.csv",
                "text/csv"
            )

        with col2:
            portfolio_data = {
                'capital': st.session_state.profile_data['capital'],
                'allocations': final_alloc,
                'expected_return': st.session_state.ai_allocation['expected_return'],
                'risk_level': st.session_state.ai_allocation['risk_level'],
                'diversification_index': st.session_state.ai_allocation['diversification_index'],
                'explanation': st.session_state.ai_allocation['explanation']
            }

            pdf_data = generate_pdf_report(portfolio_data, [])
            if pdf_data:
                st.download_button(
                    "Download PDF Report",
                    pdf_data,
                    "portfolio_report.pdf",
                    "application/pdf"
                )

        col1, col2 = st.columns(2)
        with col1:
            if st.button("← Back"):
                st.session_state.wizard_step = 3
                st.rerun()

        with col2:
            if st.button("🏁 Start New Portfolio", type="primary"):
                st.session_state.wizard_step = 1
                st.session_state.profile_data = {}
                st.session_state.selected_tickers = []
                st.session_state.ai_allocation = None
                st.session_state.manual_allocation = {}
                st.rerun()

if __name__ == "__main__":
    main()
