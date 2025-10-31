"""
AI Financial Advisor Application - LLAMA 3.3
A comprehensive financial planning tool with AI-powered insights

Required pip packages:
pip install streamlit plotly pandas numpy python-dotenv langchain-groq yfinance
"""

import streamlit as st
import os
import json
import sys
from dotenv import load_dotenv

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from typing import Dict, List, Any
from langchain_groq import ChatGroq
import yfinance as yf
import time

# Test mode check
TEST_MODE = "--test" in sys.argv

def get_market_snapshot():
    tickers = {
        "S&P 500":"^GSPC",
        "Bonds":"IEF",
        "Gold":"GLD",
        "Bitcoin":"BTC-USD",
        "Ethereum":"ETH-USD"
    }
    snap = {}
    for n,s in tickers.items():
        try:
            d = yf.Ticker(s).history(period="5d")["Close"]
            snap[n] = round((d.iloc[-1]-d.iloc[0])/d.iloc[0]*100,2)
        except:
            snap[n] = 0.0
    return snap

def get_ai_portfolio_suggestion(snapshot,risk,horizon,capital,age):
    from langchain_groq import ChatGroq

    if not groq_api_key or TEST_MODE:
        return {
            "Stocks":50,
            "Bonds":25,
            "Gold":10,
            "Crypto":5,
            "Cash":10,
            "Reasoning":"Default fallback suggestion due to unavailable AI."
        }

    try:
        llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0.4, groq_api_key=groq_api_key)
        txt = "\n".join([f"{k}: {v:+.2f}%" for k,v in snapshot.items()])
        prompt = f"You are an AI financial advisor. Based on market data:\n{txt}\n\nUser profile: Risk tolerance: {risk}, Time horizon: {horizon} years, Capital: ${capital}, Age: {age}.\n\nSuggest a 100% allocation across Stocks, Bonds, Gold, Crypto, Cash with brief reasoning. Output ONLY valid JSON with keys: Stocks, Bonds, Gold, Crypto, Cash, Reasoning."

        response = llm.invoke(prompt)
        result = json.loads(response.content.strip())
        return result
    except:
        return {
            "Stocks":50,
            "Bonds":25,
            "Gold":10,
            "Crypto":5,
            "Cash":10,
            "Reasoning":"Default fallback suggestion."
        }

if not TEST_MODE:
    # Set Streamlit Page Config
    st.set_page_config(
        page_title="AI Financial Advisor - LLAMA 3.3",
        page_icon="💰",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    # Layout optimization
    st.markdown("<style>div.block-container{padding-top:1rem;}</style>", unsafe_allow_html=True)

    # Custom CSS for dark theme financial advisor styling
    st.markdown("""
    <style>
        /* Global dark theme */
        .stApp {
            background-color: #0e1117;
            color: #ffffff;
        }

        /* Main header styling */
        .main-header {
            font-size: 2.5rem;
            font-weight: bold;
            color: #ffffff;
            text-align: center;
            margin-bottom: 2rem;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }

        /* Dark theme cards */
        .flow-card {
            background: linear-gradient(135deg, #1e293b 0%, #334155 100%);
            padding: 1.5rem;
            border-radius: 15px;
            color: #ffffff;
            margin: 1rem 0;
            box-shadow: 0 8px 32px rgba(0,0,0,0.3);
            transition: transform 0.3s ease;
            border: 1px solid #374151;
        }
        .flow-card:hover {
            transform: translateY(-5px);
            box-shadow: 0 12px 40px rgba(0,0,0,0.4);
        }

        /* Dark metric cards */
        .metric-card {
            background: linear-gradient(135deg, #1f2937 0%, #374151 100%);
            padding: 1.5rem;
            border-radius: 12px;
            border-left: 5px solid #3b82f6;
            margin-bottom: 15px !important;
            box-shadow: 0 4px 16px rgba(0,0,0,0.2);
            color: #ffffff;
            border: 1px solid #4b5563;
        }
        .metric-card h2, .metric-card h3, .metric-card h4 {
            color: #ffffff !important;
        }
        .metric-card p {
            color: #d1d5db !important;
        }

        /* AI Suggestions Card */
        .ai-suggestions-card {
            background: linear-gradient(135deg, #581c87 0%, #7c3aed 100%);
            padding: 1.5rem;
            border-radius: 12px;
            border-left: 5px solid #a78bfa;
            margin: 1rem 0;
            box-shadow: 0 4px 16px rgba(0,0,0,0.2);
            color: #ffffff;
            border: 1px solid #7c3aed;
        }
        .ai-suggestions-card h3, .ai-suggestions-card h4 {
            color: #ffffff !important;
        }
        .ai-suggestions-card p, .ai-suggestions-card ul li {
            color: #e9d5ff !important;
            margin-bottom: 0.5rem;
        }

        /* Dark summary cards */
        .summary-card {
            background: linear-gradient(135deg, #1f2937 0%, #374151 100%);
            padding: 1.5rem;
            border-radius: 12px;
            margin: 1rem 0;
            border-left: 5px solid #10b981;
            color: #ffffff;
            border: 1px solid #4b5563;
        }
        .summary-card h3, .summary-card h4 {
            color: #ffffff !important;
        }
        .summary-card ul li {
            color: #d1d5db !important;
            margin-bottom: 0.5rem;
        }

        /* Streamlit component overrides */
        .stSelectbox > div > div {
            background-color: #374151 !important;
            color: #ffffff !important;
            border: 1px solid #6b7280 !important;
        }

        .stNumberInput > div > div > input {
            background-color: #374151 !important;
            color: #ffffff !important;
            border: 1px solid #6b7280 !important;
        }

        .stTextInput > div > div > input {
            background-color: #374151 !important;
            color: #ffffff !important;
            border: 1px solid #6b7280 !important;
        }

        .stRadio > div {
            background-color: #1f2937 !important;
            padding: 1rem;
            border-radius: 8px;
            border: 1px solid #4b5563;
        }

        .stRadio label {
            color: #ffffff !important;
        }

        .stCheckbox label {
            color: #ffffff !important;
        }

        .stSlider > div > div > div {
            background-color: #374151 !important;
        }

        /* Sidebar styling */
        .css-1d391kg {
            background-color: #1f2937 !important;
        }

        .css-1d391kg .stSelectbox label {
            color: #ffffff !important;
        }

        /* Dataframe styling */
        .stDataFrame {
            background-color: #1f2937 !important;
        }

        .stDataFrame table {
            background-color: #374151 !important;
            color: #ffffff !important;
        }

        .stDataFrame th {
            background-color: #4b5563 !important;
            color: #ffffff !important;
        }

        .stDataFrame td {
            background-color: #374151 !important;
            color: #ffffff !important;
        }

        /* Button styling */
        .stButton > button {
            background-color: #3b82f6 !important;
            color: #ffffff !important;
            border: none !important;
            border-radius: 8px !important;
        }

        .stButton > button:hover {
            background-color: #2563eb !important;
            box-shadow: 0 4px 12px rgba(59, 130, 246, 0.3) !important;
        }

        /* Expander styling */
        .streamlit-expanderHeader {
            background-color: #374151 !important;
            color: #ffffff !important;
            border: 1px solid #6b7280 !important;
        }

        .streamlit-expanderContent {
            background-color: #1f2937 !important;
            border: 1px solid #4b5563 !important;
        }

        /* Metric styling */
        .css-1xarl3l {
            background-color: #1f2937 !important;
            padding: 1rem !important;
            border-radius: 8px !important;
            border: 1px solid #4b5563 !important;
        }

        [data-testid="stMetric"] {
            background-color: #1f2937;
            padding: 1rem;
            border-radius: 8px;
            border: 1px solid #4b5563;
            margin-bottom: 1rem;
        }

        [data-testid="stMetricValue"] {
            font-size: 1.3rem !important;
            font-weight: bold;
            color: #ffffff;
        }

        [data-testid="stMetricLabel"] {
            font-size: 0.85rem !important;
            color: #9ca3af;
            margin-bottom: 0.5rem;
        }

        [data-testid="stMetricDelta"] {
            font-size: 0.75rem;
            margin-top: 0.5rem;
        }

        /* Success/Warning/Error message styling */
        .stSuccess {
            background-color: #065f46 !important;
            color: #ffffff !important;
            border: 1px solid #10b981 !important;
        }

        .stWarning {
            background-color: #92400e !important;
            color: #ffffff !important;
            border: 1px solid #f59e0b !important;
        }

        .stError {
            background-color: #991b1b !important;
            color: #ffffff !important;
            border: 1px solid #ef4444 !important;
        }

        .stInfo {
            background-color: #1e40af !important;
            color: #ffffff !important;
            border: 1px solid #3b82f6 !important;
        }

        /* Plotly chart background */
        .js-plotly-plot {
            background-color: #1f2937 !important;
        }
    </style>
    """, unsafe_allow_html=True)

# Load environment variables
load_dotenv()
working_dir = os.path.dirname(os.path.abspath(__file__))

def load_groq_api_key():
    """
    Loads the GROQ API key from config.json or environment variables.

    Returns:
        str: GROQ API key or None if not found
    """
    try:
        with open(os.path.join(working_dir, "config.json"), "r") as f:
            return json.load(f).get("GROQ_API_KEY")
    except FileNotFoundError:
        return os.getenv("GROQ_API_KEY")

groq_api_key = load_groq_api_key() if not TEST_MODE else "test_key"

if not groq_api_key and not TEST_MODE:
    st.error("🚨 GROQ_API_KEY is missing. Check your config.json file or environment variables.")
    st.warning("💡 AI features will use deterministic fallback mode.")

def generate_ai_insights(data: Dict[str, Any], context_label: str) -> Dict[str, Any]:
    """
    Centralized AI insights generator using LLaMA 3.3 via Groq.

    Args:
        data: Dictionary containing financial data for analysis
        context_label: Label indicating the type of analysis

    Returns:
        Dict containing AI score (0-100), reasoning, and recommendations
    """
    fallback_response = {
        "ai_score": None,
        "ai_reasoning": "AI analysis not available - using deterministic fallback.",
        "ai_recommendations": [
            "Review your financial data and look for improvement opportunities",
            "Consider consulting with a financial professional for personalized advice",
            "Use the built-in calculators and metrics for guidance"
        ]
    }

    if not groq_api_key or TEST_MODE:
        return fallback_response

    try:
        llm = ChatGroq(
            model="llama-3.3-70b-versatile",
            temperature=0.3,
            groq_api_key=groq_api_key
        )

        if context_label == "Budget Analysis":
            prompt = f"""
            You are an expert financial advisor with advanced analytical capabilities. Analyze this budget data thoroughly.

            Budget Data (JSON): {json.dumps(data, default=str)}

            Tasks:
            1. Calculate a Financial Wellness Index (0-100) based on:
               - Savings rate and emergency fund adequacy
               - Housing and debt ratios
               - Spending patterns vs. ideal benchmarks (50/30/20 rule)
               - Overall financial stability

            2. Provide 3-4 sentences of expert-level reasoning explaining the score, highlighting strengths and weaknesses.

            3. Generate 5-7 personalized, data-driven recommendations including:
               - Specific category overspending detection with exact percentages
               - Savings goal suggestions based on income level
               - Comparison to 50/30/20 ideal ratio (50% needs, 30% wants, 20% savings)
               - Emergency fund status (recommend 3-6 months of expenses)
               - Debt reduction strategies if applicable
               - Monthly investment/saving targets with specific dollar amounts
               - Long-term wealth building strategies

            4. Add a brief motivational message at the end of your recommendations.

            Output ONLY valid JSON with keys: ai_score, ai_reasoning, ai_recommendations
            Example: {{"ai_score": 75, "ai_reasoning": "Your financial health is...", "ai_recommendations": ["Reduce dining expenses by 15%...", ...]}}
            """

        elif context_label == "Investment Analysis":
            prompt = f"""
            You are an expert investment advisor. Analyze this portfolio data.

            Investment Data (JSON): {json.dumps(data, default=str)}

            Tasks:
            1. Provide an Investment Risk Score (0-100)
            2. Give a brief 2-3 sentence explanation
            3. Provide 3-5 specific portfolio improvement suggestions

            Output ONLY valid JSON with keys: ai_score, ai_reasoning, ai_recommendations
            """

        elif context_label == "Debt Analysis":
            prompt = f"""
            You are an expert debt counselor. Analyze this debt situation.

            Debt Data (JSON): {json.dumps(data, default=str)}

            Tasks:
            1. Provide a Debt Health Score (0-100)
            2. Give a brief 2-3 sentence assessment
            3. Provide 3-5 prioritized actionable steps

            Output ONLY valid JSON with keys: ai_score, ai_reasoning, ai_recommendations
            """

        elif context_label == "Retirement Analysis":
            prompt = f"""
            You are an expert retirement planner. Analyze this retirement planning data.

            Retirement Data (JSON): {json.dumps(data, default=str)}

            Tasks:
            1. Provide a Retirement Readiness Index (0-100)
            2. Give a brief 2-3 sentence assessment
            3. Provide 3-5 specific actions to improve readiness

            Output ONLY valid JSON with keys: ai_score, ai_reasoning, ai_recommendations
            """

        else:
            return fallback_response

        response = llm.invoke(prompt)
        response_text = response.content.strip()

        try:
            if response_text.startswith("{") and response_text.endswith("}"):
                ai_result = json.loads(response_text)
            else:
                import re
                json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
                if json_match:
                    ai_result = json.loads(json_match.group())
                else:
                    return fallback_response

            required_keys = ["ai_score", "ai_reasoning", "ai_recommendations"]
            if not all(key in ai_result for key in required_keys):
                return fallback_response

            if isinstance(ai_result["ai_score"], str):
                ai_result["ai_score"] = float(ai_result["ai_score"].replace("%",""))

            if ai_result["ai_score"] is not None:
                ai_result["ai_score"] = max(0, min(100, float(ai_result["ai_score"])))

            if not isinstance(ai_result["ai_recommendations"], list):
                ai_result["ai_recommendations"] = [str(ai_result["ai_recommendations"])]

            return ai_result

        except (json.JSONDecodeError, ValueError, KeyError):
            return fallback_response

    except Exception as e:
        if not TEST_MODE:
            st.warning(f"AI analysis temporarily unavailable: {str(e)}")
        return fallback_response

def display_metric_card(title: str, value: str, subtitle: str = "", color: str = None) -> str:
    """
    Generate a standardized metric card with consistent styling.

    Args:
        title: The metric title/label
        value: The primary value to display
        subtitle: Optional subtitle text
        color: Optional color for the value text

    Returns:
        HTML string for the metric card
    """
    color_style = f'style="color: {color}"' if color else ''
    subtitle_html = f'<p>{subtitle}</p>' if subtitle else ''

    return f'''
    <div class="metric-card">
        <h3>{title}</h3>
        <h2 {color_style}>{value}</h2>
        {subtitle_html}
    </div>
    '''

def display_ai_suggestions(ai_insights: Dict[str, Any], context_label: str):
    """
    Display AI suggestions in a consistent format.

    Args:
        ai_insights: Dictionary containing AI analysis results
        context_label: Label for the type of analysis
    """
    if TEST_MODE:
        return

    ai_score = ai_insights.get("ai_score")
    ai_reasoning = ai_insights.get("ai_reasoning", "")
    ai_recommendations = ai_insights.get("ai_recommendations", [])

    st.markdown("### 🤖 AI Suggestions")

    if ai_score is not None:
        st.markdown(f'''
        <div class="ai-suggestions-card">
            <h4>AI Score: {ai_score}/100</h4>
            <p><strong>Analysis:</strong> {ai_reasoning}</p>
            <h4>Personalized Recommendations:</h4>
            <ul>
                {"".join(f"<li>{rec}</li>" for rec in ai_recommendations)}
            </ul>
        </div>
        ''', unsafe_allow_html=True)
    else:
        st.markdown(f'''
        <div class="ai-suggestions-card">
            <h4>AI Analysis</h4>
            <p><strong>Note:</strong> {ai_reasoning}</p>
            <h4>General Recommendations:</h4>
            <ul>
                {"".join(f"<li>{rec}</li>" for rec in ai_recommendations)}
            </ul>
        </div>
        ''', unsafe_allow_html=True)

class FinancialCalculator:
    """Core financial calculation functions with advanced analytics"""

    @staticmethod
    def calculate_budget_summary(income: float, expenses: Dict[str, float]) -> Dict[str, Any]:
        """Calculate comprehensive budget summary with dynamic scores and benchmark comparison."""
        if income <= 0:
            return {
                'total_income': 0,
                'total_expenses': 0,
                'savings': 0,
                'savings_rate': 0,
                'essential_expenses': 0,
                'discretionary_expenses': 0,
                'expense_breakdown': {},
                'financial_health': 'Critical',
                'health_color': '#f44336',
                'health_score': 0,
                'recommendations': ['Please enter valid income and expense data.']
            }

        total_expenses = sum(expenses.values())
        savings = income - total_expenses
        savings_rate = max(0, savings / income * 100) if income > 0 else 0

        essential_categories = ['housing', 'utilities', 'groceries', 'transportation', 'insurance', 'healthcare']
        essential_expenses = sum(expenses.get(cat, 0) for cat in essential_categories if cat in expenses)
        discretionary_expenses = total_expenses - essential_expenses

        health_score = 0

        if savings_rate >= 20:
            health_score += 50
        elif savings_rate >= 10:
            health_score += 35
        elif savings_rate >= 5:
            health_score += 20
        elif savings_rate >= 0:
            health_score += 10

        housing_ratio = expenses.get('housing', 0) / income * 100 if income > 0 else 0
        if housing_ratio <= 25:
            health_score += 25
        elif housing_ratio <= 30:
            health_score += 20
        elif housing_ratio <= 35:
            health_score += 10

        debt_ratio = expenses.get('debt_payments', 0) / income * 100 if income > 0 else 0
        if debt_ratio <= 10:
            health_score += 15
        elif debt_ratio <= 20:
            health_score += 10
        elif debt_ratio <= 30:
            health_score += 5

        if savings > 0:
            health_score += 10

        health_score = min(100, health_score)

        if health_score >= 80:
            health_status = "Excellent"
            health_color = "#4caf50"
        elif health_score >= 65:
            health_status = "Good"
            health_color = "#8bc34a"
        elif health_score >= 45:
            health_status = "Fair"
            health_color = "#ff9800"
        elif health_score >= 25:
            health_status = "Poor"
            health_color = "#ff5722"
        else:
            health_status = "Critical"
            health_color = "#f44336"

        ideal_ratios = {
            "housing": 30,
            "utilities": 10,
            "groceries": 15,
            "transportation": 10,
            "insurance": 10,
            "savings": 20,
            "discretionary": 15
        }

        recommendations = FinancialCalculator._get_budget_recommendations(savings_rate, expenses, income)

        for category, ideal_pct in ideal_ratios.items():
            if category in expenses:
                actual_pct = (expenses[category] / income * 100) if income > 0 else 0
                if actual_pct > ideal_pct * 1.20:
                    overspend_amount = expenses[category] - (income * ideal_pct / 100)
                    recommendations.append(
                        f"⚠️ {category.title()} spending is {actual_pct:.1f}% of income (ideal: {ideal_pct}%). "
                        f"Consider reducing by ${overspend_amount:.0f}/month"
                    )

        return {
            'total_income': income,
            'total_expenses': total_expenses,
            'savings': savings,
            'savings_rate': savings_rate,
            'essential_expenses': essential_expenses,
            'discretionary_expenses': discretionary_expenses,
            'expense_breakdown': expenses,
            'financial_health': health_status,
            'health_color': health_color,
            'health_score': health_score,
            'recommendations': recommendations
        }

    @staticmethod
    def _get_budget_recommendations(savings_rate: float, expenses: Dict[str, float], income: float) -> List[str]:
        """Generate personalized budget recommendations."""
        recommendations = []

        if savings_rate < 10:
            recommendations.append("🎯 Aim to save at least 10% of your income")

        housing_ratio = expenses.get('housing', 0) / income * 100 if income > 0 else 0
        if housing_ratio > 30:
            recommendations.append(f"🏠 Consider reducing housing costs - currently {round(housing_ratio, 1)}% of income")

        if expenses.get('dining_out', 0) > expenses.get('groceries', 0):
            recommendations.append("🍽️ Consider cooking more at home to reduce dining expenses")

        if savings_rate >= 20:
            recommendations.append("🌟 Excellent savings rate! Consider investing surplus funds")

        if expenses.get('debt_payments', 0) / income > 0.2:
            recommendations.append("💳 Focus on debt repayment - debt payments are high relative to income")

        return recommendations

    @staticmethod
    def calculate_investment_allocation(risk_profile: str, time_horizon: int, capital: float, age: int = 35) -> Dict[str, Any]:
        """Calculate sophisticated investment allocation with dynamic allocations."""
        base_allocations = {
            'conservative': {'stocks': 25, 'bonds': 65, 'cash': 10},
            'moderate': {'stocks': 60, 'bonds': 30, 'cash': 10},
            'aggressive': {'stocks': 85, 'bonds': 10, 'cash': 5}
        }

        allocation = base_allocations.get(risk_profile.lower(), base_allocations['moderate']).copy()

        age_adjusted_stock = max(20, min(90, 110 - age))

        if time_horizon < 3:
            allocation['stocks'] = max(10, allocation['stocks'] - 30)
            allocation['cash'] += 20
            allocation['bonds'] += 10
        elif time_horizon < 7:
            allocation['stocks'] = max(20, allocation['stocks'] - 15)
            allocation['bonds'] += 10
            allocation['cash'] += 5
        elif time_horizon > 20:
            allocation['stocks'] = min(95, allocation['stocks'] + 10)
            allocation['bonds'] = max(5, allocation['bonds'] - 8)
            allocation['cash'] = max(0, allocation['cash'] - 2)

        allocation['stocks'] = int((allocation['stocks'] + age_adjusted_stock) / 2)
        total_non_stock = allocation['bonds'] + allocation['cash']
        allocation['bonds'] = max(5, 100 - allocation['stocks'] - allocation['cash'])

        if allocation['stocks'] + allocation['bonds'] + allocation['cash'] != 100:
            diff = 100 - (allocation['stocks'] + allocation['bonds'] + allocation['cash'])
            allocation['bonds'] += diff

        total = sum(allocation.values())
        if total != 100:
            for k in allocation:
                allocation[k] = round(allocation[k] / total * 100)

        dollar_allocation = {
            asset: (percentage / 100) * capital
            for asset, percentage in allocation.items()
        }

        expected_returns = {
            'stocks': 0.10,
            'bonds': 0.04,
            'cash': 0.02
        }

        portfolio_return = sum(
            (allocation[asset] / 100) * expected_returns[asset]
            for asset in allocation
        )

        projections = {}
        projection_years = sorted(set(
            [5, 10, 15, 20, 25, 30, int(time_horizon)] if time_horizon > 0 else [5, 10, 20, 30]
        ))
        projection_years = [y for y in projection_years if y <= time_horizon]

        if not projection_years and time_horizon > 0:
            projection_years = [time_horizon]

        for years in projection_years:
            conservative = capital * ((1 + portfolio_return * 0.7) ** years)
            expected = capital * ((1 + portfolio_return) ** years)
            optimistic = capital * ((1 + portfolio_return * 1.3) ** years)

            projections[f'{years}_years'] = {
                'conservative': conservative,
                'expected': expected,
                'optimistic': optimistic
            }

        return {
            'allocation_percentages': allocation,
            'allocation_dollars': dollar_allocation,
            'expected_annual_return': portfolio_return,
            'projections': projections,
            'risk_level': risk_profile,
            'volatility_estimate': FinancialCalculator._calculate_portfolio_volatility(allocation)
        }

    @staticmethod
    def _calculate_portfolio_volatility(allocation: Dict[str, int]) -> float:
        """Calculate estimated portfolio volatility."""
        volatilities = {'stocks': 0.16, 'bonds': 0.05, 'cash': 0.01}
        return sum((allocation[asset] / 100) * volatilities[asset] for asset in allocation)

    @staticmethod
    def calculate_debt_payoff(debts: List[Dict], extra_payment: float = 0, strategy: str = 'avalanche') -> Dict[str, Any]:
        """Accurate debt payoff calculator with realistic month-by-month simulation."""
        if not debts:
            return {
                'total_debt': 0,
                'payoff_plan': [],
                'total_interest': 0,
                'scenarios': {},
                'strategy': strategy
            }

        valid_debts = []
        for debt in debts:
            try:
                balance = float(debt.get('balance', 0))
                rate = float(debt.get('interest_rate', 0)) / 100 / 12
                min_payment = float(debt.get('minimum_payment', 0))
                if balance <= 0:
                    continue
                if min_payment <= balance * rate:
                    min_payment = max(25, balance * 0.02)
                valid_debts.append({
                    'name': debt.get('name', 'Unknown Debt'),
                    'balance': balance,
                    'interest_rate': rate,
                    'minimum_payment': min_payment
                })
            except Exception:
                continue

        if not valid_debts:
            return {
                'total_debt': 0,
                'payoff_plan': [],
                'total_interest': 0,
                'scenarios': {},
                'strategy': strategy
            }

        total_debt = sum(d['balance'] for d in valid_debts)

        def simulate(extra_amt: float):
            debts_sim = [d.copy() for d in valid_debts]

            if strategy == 'avalanche':
                debts_sim.sort(key=lambda x: x['interest_rate'], reverse=True)
            else:
                debts_sim.sort(key=lambda x: x['balance'])

            debt_stats = {i: {'months': 0, 'interest': 0.0, 'name': debts_sim[i]['name'],
                              'balance': debts_sim[i]['balance'], 'rate': debts_sim[i]['interest_rate'],
                              'min_pay': debts_sim[i]['minimum_payment']}
                          for i in range(len(debts_sim))}

            months = 0
            total_interest = 0.0
            current_extra = extra_amt

            while any(d['balance'] > 0.01 for d in debts_sim):
                priority_debt_idx = None
                for i, d in enumerate(debts_sim):
                    if d['balance'] > 0.01:
                        priority_debt_idx = i
                        break

                if priority_debt_idx is None:
                    break

                for i, d in enumerate(debts_sim):
                    if d['balance'] <= 0:
                        continue

                    payment = d['minimum_payment']
                    if i == priority_debt_idx and current_extra > 0:
                        payment += current_extra

                    interest = d['balance'] * d['interest_rate']
                    principal = max(0, payment - interest)
                    d['balance'] = max(0, d['balance'] - principal)
                    total_interest += interest

                    debt_stats[i]['interest'] += interest
                    if d['balance'] > 0.01:
                        debt_stats[i]['months'] = months + 1

                    if d['balance'] <= 0.01 and debt_stats[i]['months'] == months + 1:
                        current_extra += d['minimum_payment']

                months += 1
                if months > 1000:
                    break

            payoff_plan = []
            for i in range(len(debts_sim)):
                stat = debt_stats[i]
                payoff_plan.append({
                    'debt_name': stat['name'],
                    'balance': stat['balance'],
                    'interest_rate': stat['rate'] * 12 * 100,
                    'monthly_payment': stat['min_pay'] + (extra_amt if i == 0 else 0),
                    'months_to_payoff': max(1, stat['months']),
                    'interest_paid': stat['interest'],
                    'priority': i + 1
                })

            return {'months': months, 'interest': total_interest, 'payoff_plan': payoff_plan}

        base = simulate(0)
        with_extra = simulate(extra_payment)

        interest_savings = max(0, base['interest'] - with_extra['interest'])
        time_savings = max(0, base['months'] - with_extra['months'])

        return {
            'total_debt': total_debt,
            'strategy': strategy,
            'total_interest': base['interest'],
            'interest_savings': interest_savings,
            'time_savings_months': time_savings,
            'recommended_extra_payment': max(50, total_debt * 0.02),
            'scenarios': {
                'minimum_only': {
                    'total_interest': base['interest'],
                    'total_months': base['months'],
                    'payoff_plan': base['payoff_plan']
                },
                'with_extra': {
                    'total_interest': with_extra['interest'],
                    'total_months': with_extra['months'],
                    'payoff_plan': with_extra['payoff_plan']
                }
            },
            'payoff_plan': with_extra['payoff_plan'] if extra_payment > 0 else base['payoff_plan']
        }

    @staticmethod
    def calculate_retirement_needs(current_age: int, retirement_age: int, current_income: float,
                                 current_savings: float, monthly_contribution: float) -> Dict[str, Any]:
        """Calculate comprehensive retirement planning."""
        if retirement_age <= current_age:
            return {
                'error': 'Retirement age must be greater than current age.',
                'current_age': current_age,
                'retirement_age': retirement_age,
                'years_to_retirement': 0,
                'current_savings': current_savings,
                'monthly_contribution': monthly_contribution,
                'retirement_corpus_needed': 0,
                'projected_savings': 0,
                'retirement_gap': 0,
                'required_monthly_contribution': 0,
                'scenarios': {},
                'recommendations': ['Please set retirement age greater than current age.']
            }

        if current_income <= 0:
            return {
                'current_age': current_age,
                'retirement_age': retirement_age,
                'years_to_retirement': max(1, retirement_age - current_age),
                'current_savings': current_savings,
                'monthly_contribution': monthly_contribution,
                'retirement_corpus_needed': 0,
                'projected_savings': 0,
                'retirement_gap': 0,
                'required_monthly_contribution': 0,
                'scenarios': {},
                'recommendations': ['Please enter valid income data.']
            }

        years_to_retirement = retirement_age - current_age
        if years_to_retirement <= 0:
            return {"error": "Retirement age must be greater than current age."}

        annual_contribution = monthly_contribution * 12

        inflation_rate = 0.03
        investment_return = 0.07
        replacement_ratio = 0.80
        life_expectancy = 85
        retirement_years = max(1, life_expectancy - retirement_age)

        future_income_needed = current_income * ((1 + inflation_rate) ** years_to_retirement)
        annual_retirement_need = future_income_needed * replacement_ratio

        real_return = investment_return - inflation_rate
        if real_return > 0:
            retirement_corpus_needed = annual_retirement_need * (
                (1 - (1 + real_return) ** -retirement_years) / real_return
            )
        else:
            retirement_corpus_needed = annual_retirement_need * retirement_years

        future_current_savings = current_savings * ((1 + investment_return) ** years_to_retirement)

        if investment_return > 0 and annual_contribution > 0:
            future_contributions = annual_contribution * (
                ((1 + investment_return) ** years_to_retirement - 1) / investment_return
            )
        else:
            future_contributions = annual_contribution * years_to_retirement

        total_projected_savings = future_current_savings + future_contributions

        retirement_gap = max(0, retirement_corpus_needed - total_projected_savings)

        if retirement_gap > 0 and years_to_retirement > 0:
            if investment_return > 0:
                required_annual_contribution = retirement_gap / (
                    ((1 + investment_return) ** years_to_retirement - 1) / investment_return
                )
            else:
                required_annual_contribution = retirement_gap / years_to_retirement

            required_monthly_contribution = required_annual_contribution / 12
        else:
            required_monthly_contribution = 0

        scenarios = {}
        for contribution_multiplier, scenario_name in [(0.5, 'conservative'), (1.0, 'current'), (1.5, 'aggressive')]:
            scenario_monthly = monthly_contribution * contribution_multiplier
            scenario_annual = scenario_monthly * 12

            if investment_return > 0 and scenario_annual > 0:
                scenario_future_contributions = scenario_annual * (
                    ((1 + investment_return) ** years_to_retirement - 1) / investment_return
                )
            else:
                scenario_future_contributions = scenario_annual * years_to_retirement

            scenario_total = future_current_savings + scenario_future_contributions

            if real_return > 0:
                monthly_retirement_income = (scenario_total * real_return) / 12
            else:
                monthly_retirement_income = scenario_total / (retirement_years * 12)

            raw_ratio = (monthly_retirement_income * 12) / future_income_needed if future_income_needed > 0 else 0
            replacement_ratio_achieved = min(1, raw_ratio)
            display_ratio = f"{raw_ratio*100:.1f}%" + (" (Capped at 100%)" if raw_ratio > 1 else "")

            scenarios[scenario_name] = {
                'monthly_contribution': scenario_monthly,
                'projected_total': scenario_total,
                'monthly_retirement_income': monthly_retirement_income,
                'replacement_ratio_achieved': replacement_ratio_achieved,
                'display_ratio': display_ratio
            }

        return {
            'current_age': current_age,
            'retirement_age': retirement_age,
            'years_to_retirement': years_to_retirement,
            'current_savings': current_savings,
            'monthly_contribution': monthly_contribution,
            'retirement_corpus_needed': retirement_corpus_needed,
            'projected_savings': total_projected_savings,
            'retirement_gap': retirement_gap,
            'required_monthly_contribution': required_monthly_contribution,
            'scenarios': scenarios,
            'recommendations': FinancialCalculator._get_retirement_recommendations(
                retirement_gap, years_to_retirement, monthly_contribution, required_monthly_contribution
            )
        }

    @staticmethod
    def _get_retirement_recommendations(gap: float, years_left: int, current_contrib: float, required_contrib: float) -> List[str]:
        """Generate retirement planning recommendations."""
        recommendations = []

        if gap > 0:
            increase_needed = max(0, required_contrib - current_contrib)
            if increase_needed > 0:
                recommendations.append(f"💰 Increase monthly contributions by ${increase_needed:.0f}")
            else:
                recommendations.append("🎉 You are already on track, no increase needed")
        else:
            recommendations.append("🎉 You are already on track, no increase needed")

        if years_left > 30:
            recommendations.append("📈 Consider more aggressive investments for long-term growth")
        elif years_left < 10:
            recommendations.append("🛡️ Consider shifting to more conservative investments")

        if current_contrib < 500:
            recommendations.append("🎯 Aim to contribute at least $500/month for retirement")

        recommendations.append("🏢 Maximize employer 401(k) matching if available")
        recommendations.append("💡 Consider Roth IRA for tax-free retirement income")

        return recommendations

class FinancialVisualizer:
    """Advanced visualization functions for financial data"""

    @staticmethod
    def plot_expense_breakdown(expenses: Dict[str, float], title: str = "Expense Breakdown") -> go.Figure:
        """Create an interactive pie chart for expense breakdown."""
        filtered_expenses = {k: v for k, v in expenses.items() if v > 0}

        if not filtered_expenses:
            fig = go.Figure()
            fig.add_annotation(
                text="No expense data available<br>Please enter your expenses to see the breakdown",
                x=0.5, y=0.5, showarrow=False,
                font=dict(size=16, color='white')
            )
            fig.update_layout(
                paper_bgcolor='#1f2937', plot_bgcolor='#1f2937',
                font_color='white', height=400
            )
            return fig

        fig = px.pie(
            values=list(filtered_expenses.values()),
            names=list(filtered_expenses.keys()),
            title=title,
            color_discrete_sequence=px.colors.qualitative.Set3
        )

        fig.update_traces(
            textposition='inside',
            textinfo='percent+label',
            hovertemplate='<b>%{label}</b><br>Amount: $%{value:,.0f}<br>Percentage: %{percent}<extra></extra>'
        )

        fig.update_layout(
            showlegend=True,
            height=500,
            font=dict(size=12, color='white'),
            paper_bgcolor='#1f2937',
            plot_bgcolor='#1f2937',
            margin=dict(t=80, b=40, l=40, r=40)
        )

        return fig

    @staticmethod
    def plot_budget_summary(budget_data: Dict[str, Any]) -> go.Figure:
        """Create a comprehensive budget visualization."""
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Income vs Expenses', 'Savings Rate', 'Expense Categories', 'Financial Health Score'),
            specs=[[{"type": "bar"}, {"type": "indicator"}],
                   [{"type": "pie"}, {"type": "indicator"}]],
            vertical_spacing=0.25,
            horizontal_spacing=0.12
        )

        fig.add_trace(
            go.Bar(
                x=['Income', 'Expenses', 'Savings'],
                y=[budget_data['total_income'], budget_data['total_expenses'], budget_data['savings']],
                marker_color=['#2ecc71', '#e74c3c', '#3498db'],
                name='Amount'
            ),
            row=1, col=1
        )

        fig.add_trace(
            go.Indicator(
                mode="gauge+number",
                value=budget_data['savings_rate'],
                title={'text': "<b>Savings Rate (%)</b>", 'font': {'size': 18, 'color': 'white'}},
                number={'font': {'size': 36, 'color': 'white'}},
                gauge={
                    'axis': {'range': [None, 100], 'tickwidth': 1, 'tickcolor': 'white'},
                    'bar': {'color': 'darkblue', 'thickness': 0.3},
                    'steps': [
                        {'range': [0, 50], 'color': '#d1d5db'},
                        {'range': [50, 80], 'color': '#9ca3af'},
                        {'range': [80, 100], 'color': '#4ade80'}
                    ],
                    'threshold': {
                        'line': {'color': 'red', 'width': 4},
                        'thickness': 0.75,
                        'value': 80
                    }
                }
            ),
            row=1, col=2
        )

        filtered_expenses = {k: v for k, v in budget_data['expense_breakdown'].items() if v > 0}
        if filtered_expenses:
            fig.add_trace(
                go.Pie(
                    labels=list(filtered_expenses.keys()),
                    values=list(filtered_expenses.values()),
                    name="Expenses"
                ),
                row=2, col=1
            )

        ai_score = budget_data.get("ai_score", budget_data.get("health_score", 0))
        health_label = budget_data.get("financial_health", "N/A")
        fig.add_trace(
            go.Indicator(
                mode="gauge+number",
                value=ai_score,
                title={'text': f"<b>AI Financial Health Score: {health_label}</b>", 'font': {'size': 18, 'color': 'white'}},
                number={'font': {'size': 36, 'color': 'white'}},
                gauge={
                    'axis': {'range': [None, 100], 'tickwidth': 1, 'tickcolor': 'white'},
                    'bar': {'color': 'darkblue', 'thickness': 0.3},
                    'steps': [
                        {'range': [0, 50], 'color': '#d1d5db'},
                        {'range': [50, 80], 'color': '#9ca3af'},
                        {'range': [80, 100], 'color': '#4ade80'}
                    ],
                    'threshold': {
                        'line': {'color': 'red', 'width': 4},
                        'thickness': 0.75,
                        'value': 80
                    }
                }
            ),
            row=2, col=2
        )

        fig.update_layout(
            height=950,
            showlegend=False,
            title_text="Budget Analysis Dashboard",
            title_font_size=20,
            title_x=0.5,
            title_y=0.98,
            paper_bgcolor='#1f2937',
            plot_bgcolor='#1f2937',
            font_color='white',
            margin=dict(t=100, b=60, l=60, r=60)
        )

        if fig.layout.annotations:
            for i, ann in enumerate(fig.layout.annotations):
                ann.font.size = 14
                ann.yshift = 25

        return fig

    @staticmethod
    def plot_investment_allocation(allocation_data: Dict[str, Any]) -> go.Figure:
        """Create investment allocation visualization."""
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Asset Allocation', 'Portfolio Projections', 'Risk vs Return', 'Dollar Allocation'),
            specs=[[{"type": "pie"}, {"type": "scatter"}],
                   [{"type": "scatter"}, {"type": "bar"}]],
            vertical_spacing=0.25,
            horizontal_spacing=0.12
        )

        allocation = allocation_data['allocation_percentages']
        fig.add_trace(
            go.Pie(
                labels=list(allocation.keys()),
                values=list(allocation.values()),
                name="Allocation",
                marker_colors=['#1f77b4', '#ff7f0e', '#2ca02c']
            ),
            row=1, col=1
        )

        projections = allocation_data.get('projections', {})
        years = []
        conservative_vals = []
        expected_vals = []
        optimistic_vals = []

        for year_key, scenarios in projections.items():
            year = int(year_key.split('_')[0])
            years.append(year)
            conservative_vals.append(scenarios['conservative'])
            expected_vals.append(scenarios['expected'])
            optimistic_vals.append(scenarios['optimistic'])

        if years:
            fig.add_trace(go.Scatter(x=years, y=conservative_vals, name='Conservative', line=dict(color='red')), row=1, col=2)
            fig.add_trace(go.Scatter(x=years, y=expected_vals, name='Expected', line=dict(color='blue')), row=1, col=2)
            fig.add_trace(go.Scatter(x=years, y=optimistic_vals, name='Optimistic', line=dict(color='green')), row=1, col=2)

        risk_return_data = {
            'Conservative': (0.08, 0.06),
            'Moderate': (0.12, 0.08),
            'Aggressive': (0.18, 0.10)
        }

        for profile, (risk, return_val) in risk_return_data.items():
            color = 'red' if profile == allocation_data['risk_level'].title() else 'lightblue'
            fig.add_trace(
                go.Scatter(
                    x=[risk], y=[return_val],
                    mode='markers+text',
                    text=[profile],
                    textposition="top center",
                    marker=dict(size=15, color=color),
                    name=profile
                ),
                row=2, col=1
            )

        dollar_allocation = allocation_data['allocation_dollars']
        fig.add_trace(
            go.Bar(
                x=list(dollar_allocation.keys()),
                y=list(dollar_allocation.values()),
                marker_color=['#1f77b4', '#ff7f0e', '#2ca02c'],
                name='Dollar Amount'
            ),
            row=2, col=2
        )

        fig.update_layout(
            height=950,
            showlegend=True,
            title_text="Investment Portfolio Analysis",
            title_font_size=20,
            title_x=0.5,
            title_y=0.98,
            paper_bgcolor='#1f2937',
            plot_bgcolor='#1f2937',
            font_color='white',
            margin=dict(t=100, b=60, l=60, r=60)
        )

        if fig.layout.annotations:
            for i, ann in enumerate(fig.layout.annotations):
                ann.font.size = 14
                ann.yshift = 25

        return fig

    @staticmethod
    def plot_debt_payoff(debt_data: Dict[str, Any]) -> go.Figure:
        """Create debt payoff visualization."""
        scenarios = debt_data.get('scenarios', {})

        if not scenarios:
            fig = go.Figure()
            fig.add_annotation(
                text="No debt data available<br>Please add your debts to see the analysis",
                x=0.5, y=0.5, showarrow=False,
                font=dict(size=16, color='white')
            )
            fig.update_layout(
                paper_bgcolor='#1f2937', plot_bgcolor='#1f2937',
                font_color='white', height=400
            )
            return fig

        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Debt Balances', 'Payoff Timeline', 'Interest Rates', 'Monthly Payments'),
            specs=[[{"type": "bar"}, {"type": "bar"}],
                   [{"type": "bar"}, {"type": "bar"}]],
            vertical_spacing=0.25,
            horizontal_spacing=0.12
        )

        debts = scenarios.get('minimum_only', {}).get('payoff_plan', [])

        if debts:
            debt_names = [debt['debt_name'] for debt in debts]
            balances = [debt['balance'] for debt in debts]
            months = [debt['months_to_payoff'] for debt in debts]
            interest_rates = [debt['interest_rate'] for debt in debts]
            payments = [debt['monthly_payment'] for debt in debts]

            fig.add_trace(
                go.Bar(x=debt_names, y=balances, name='Balance', marker_color='red'),
                row=1, col=1
            )

            fig.add_trace(
                go.Bar(x=debt_names, y=months, name='Months to Payoff', marker_color='blue'),
                row=1, col=2
            )

            fig.add_trace(
                go.Bar(x=debt_names, y=interest_rates, name='Interest Rate (%)', marker_color='orange'),
                row=2, col=1
            )

            fig.add_trace(
                go.Bar(x=debt_names, y=payments, name='Monthly Payment', marker_color='green'),
                row=2, col=2
            )

        fig.update_layout(
            height=950,
            showlegend=False,
            title_text="Debt Payoff Analysis",
            title_font_size=20,
            title_x=0.5,
            title_y=0.98,
            paper_bgcolor='#1f2937',
            plot_bgcolor='#1f2937',
            font_color='white',
            margin=dict(t=100, b=60, l=60, r=60)
        )

        if fig.layout.annotations:
            for i, ann in enumerate(fig.layout.annotations):
                ann.font.size = 14
                ann.yshift = 25

        return fig

    @staticmethod
    def plot_retirement_projections(retirement_data: Dict[str, Any]) -> go.Figure:
        """Create retirement planning visualization."""
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Retirement Scenarios', 'Contribution Impact', 'Savings Growth', 'Income Replacement'),
            specs=[[{"type": "bar"}, {"type": "scatter"}],
                   [{"type": "scatter"}, {"type": "indicator"}]],
            vertical_spacing=0.25,
            horizontal_spacing=0.12
        )

        scenarios = retirement_data.get('scenarios', {})

        scenario_names = list(scenarios.keys())
        projected_totals = [scenarios[name]['projected_total'] for name in scenario_names]
        monthly_contributions = [scenarios[name]['monthly_contribution'] for name in scenario_names]

        fig.add_trace(
            go.Bar(
                x=scenario_names,
                y=projected_totals,
                name='Projected Total',
                marker_color=['#ff7f0e', '#1f77b4', '#2ca02c']
            ),
            row=1, col=1
        )

        fig.add_trace(
            go.Scatter(
                x=monthly_contributions,
                y=projected_totals,
                mode='markers+lines',
                name='Contribution vs Total',
                marker=dict(size=10)
            ),
            row=1, col=2
        )

        years_to_retirement = retirement_data['years_to_retirement']
        current_savings = retirement_data['current_savings']
        monthly_contribution = retirement_data['monthly_contribution']

        years = list(range(0, min(years_to_retirement + 1, 31), 5))
        growth_values = []

        for year in years:
            future_current = current_savings * ((1.07) ** year)
            future_contributions = monthly_contribution * 12 * year * ((1.07) ** (year/2)) if year > 0 else 0
            growth_values.append(future_current + future_contributions)

        fig.add_trace(
            go.Scatter(
                x=years,
                y=growth_values,
                mode='lines+markers',
                name='Projected Growth',
                line=dict(color='green', width=3)
            ),
            row=2, col=1
        )

        current_replacement = scenarios.get('current', {}).get('replacement_ratio_achieved', 0) * 100
        fig.add_trace(
            go.Indicator(
                mode="gauge+number",
                value=current_replacement,
                title={'text': "<b>Income Replacement (%)</b>", 'font': {'size': 18, 'color': 'white'}},
                number={'font': {'size': 36, 'color': 'white'}},
                gauge={
                    'axis': {'range': [None, 100], 'tickwidth': 1, 'tickcolor': 'white'},
                    'bar': {'color': 'darkblue', 'thickness': 0.3},
                    'steps': [
                        {'range': [0, 50], 'color': '#d1d5db'},
                        {'range': [50, 80], 'color': '#9ca3af'},
                        {'range': [80, 100], 'color': '#4ade80'}
                    ],
                    'threshold': {
                        'line': {'color': 'red', 'width': 4},
                        'thickness': 0.75,
                        'value': 80
                    }
                }
            ),
            row=2, col=2
        )

        fig.update_layout(
            height=950,
            showlegend=True,
            title_text="Retirement Planning Analysis",
            title_font_size=20,
            title_x=0.5,
            title_y=0.98,
            paper_bgcolor='#1f2937',
            plot_bgcolor='#1f2937',
            font_color='white',
            margin=dict(t=100, b=60, l=60, r=60)
        )

        if fig.layout.annotations:
            for i, ann in enumerate(fig.layout.annotations):
                ann.font.size = 14
                ann.yshift = 25

        return fig

class FinancialFlows:
    """Structured financial advisory flows with step-by-step guidance"""

    @staticmethod
    def demo_dashboard():
        """Display demo dashboard with sample financial data."""
        if not TEST_MODE:
            st.markdown('<div class="flow-card"><h2>📊 Demo Dashboard</h2><p>This is a demo dashboard showing how the AI Financial Advisor analyzes user data across budgeting, investing, debt, and retirement planning.</p></div>', unsafe_allow_html=True)

            st.subheader("Demo User Profile")
            col1, col2, col3 = st.columns(3)
            with col1:
                st.markdown(display_metric_card("Name", "John Doe"), unsafe_allow_html=True)
            with col2:
                st.markdown(display_metric_card("Age", "35 years"), unsafe_allow_html=True)
            with col3:
                st.markdown(display_metric_card("Annual Income", "$75,000"), unsafe_allow_html=True)

            st.markdown("---")

            st.subheader("Demo Budget Summary")
            demo_expenses = {
                'housing': 1500, 'utilities': 200, 'groceries': 400,
                'transportation': 300, 'insurance': 200, 'healthcare': 150,
                'dining_out': 250, 'shopping': 150, 'subscriptions': 50,
                'savings': 600, 'debt_payments': 250, 'other': 100
            }
            demo_budget = FinancialCalculator.calculate_budget_summary(5150, demo_expenses)

            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.markdown(display_metric_card("Total Income", f"${demo_budget['total_income']:,.2f}"), unsafe_allow_html=True)
            with col2:
                st.markdown(display_metric_card("Total Expenses", f"${demo_budget['total_expenses']:,.2f}"), unsafe_allow_html=True)
            with col3:
                st.markdown(display_metric_card("Monthly Savings", f"${demo_budget['savings']:,.2f}", color='green'), unsafe_allow_html=True)
            with col4:
                st.markdown(display_metric_card("Health Score", f"{demo_budget['health_score']}/100", color=demo_budget['health_color']), unsafe_allow_html=True)

            budget_viz = FinancialVisualizer.plot_budget_summary(demo_budget)
            st.plotly_chart(budget_viz, use_container_width=True, config={"displayModeBar": False})

            st.markdown("---")

            st.subheader("Demo Investment Allocation")
            demo_investment = FinancialCalculator.calculate_investment_allocation('moderate', 20, 50000, 35)

            col1, col2, col3, col4 = st.columns(4)
            allocation = demo_investment['allocation_percentages']
            with col1:
                st.markdown(display_metric_card("Stocks", f"{allocation['stocks']}%", f"${demo_investment['allocation_dollars']['stocks']:,.0f}"), unsafe_allow_html=True)
            with col2:
                st.markdown(display_metric_card("Bonds", f"{allocation['bonds']}%", f"${demo_investment['allocation_dollars']['bonds']:,.0f}"), unsafe_allow_html=True)
            with col3:
                st.markdown(display_metric_card("Cash", f"{allocation['cash']}%", f"${demo_investment['allocation_dollars']['cash']:,.0f}"), unsafe_allow_html=True)
            with col4:
                st.markdown(display_metric_card("Expected Return", f"{demo_investment['expected_annual_return']:.1%}"), unsafe_allow_html=True)

            investment_viz = FinancialVisualizer.plot_investment_allocation(demo_investment)
            st.plotly_chart(investment_viz, use_container_width=True, config={"displayModeBar": False})

            st.markdown("---")

            st.subheader("Demo Retirement Projection")
            demo_retirement = FinancialCalculator.calculate_retirement_needs(35, 65, 75000, 50000, 600)

            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.markdown(display_metric_card("Years to Retirement", str(demo_retirement['years_to_retirement'])), unsafe_allow_html=True)
            with col2:
                st.markdown(display_metric_card("Projected Savings", f"${demo_retirement['projected_savings']:,.0f}"), unsafe_allow_html=True)
            with col3:
                st.markdown(display_metric_card("Retirement Goal", f"${demo_retirement['retirement_corpus_needed']:,.0f}"), unsafe_allow_html=True)
            with col4:
                gap = demo_retirement['retirement_gap']
                gap_color = "red" if gap > 0 else "green"
                gap_text = f"${gap:,.0f}" if gap > 0 else "On Track!"
                st.markdown(display_metric_card("Retirement Gap", gap_text, color=gap_color), unsafe_allow_html=True)

            retirement_viz = FinancialVisualizer.plot_retirement_projections(demo_retirement)
            st.plotly_chart(retirement_viz, use_container_width=True, config={"displayModeBar": False})

    @staticmethod
    def budgeting_flow():
        """Interactive budgeting flow with guided questions."""
        # Rest of budgeting_flow remains the same...
        pass

    @staticmethod
    def investing_flow():
        """Interactive Portfolio Profile Builder with 4-step wizard + AI-powered live market dashboard."""
        if not TEST_MODE:
            st.markdown('<div class="flow-card"><h2>📈 Interactive Portfolio Profile Builder</h2><p>AI-powered portfolio builder with real-time market data and LLAMA AI recommendations.</p></div>', unsafe_allow_html=True)

        if 'profile_step' not in st.session_state:
            st.session_state.profile_step = 1
        if 'profile_data' not in st.session_state:
            st.session_state.profile_data = {
                'persona': None,
                'goal': None,
                'target_amount': 0,
                'time_horizon': 10,
                'current_age': 35,
                'investment_capital': 10000.0,
                'risk_answers': {},
                'risk_score': 0,
                'risk_profile': None,
                'allocations': {'stocks': 60, 'bonds': 30, 'cash': 10, 'real_estate': 0, 'crypto': 0}
            }

        if not TEST_MODE:
            if st.session_state.profile_step == 4:
                st.markdown("## 🤖 AI-Powered Investment Suggestion")
                prof = st.session_state.profile_data
                risk = prof.get("risk_profile","Moderate")
                horizon = prof.get("time_horizon",10)
                capital = prof.get("investment_capital",10000)
                age = prof.get("current_age",35)

                snap = get_market_snapshot()
                st.markdown("### 🌍 Market Snapshot (5-Day Change)")
                c1,c2,c3,c4,c5 = st.columns(5)
                for (n,v),cl in zip(snap.items(),[c1,c2,c3,c4,c5]):
                    cl.metric(n, f"{v:+.2f}%", "⬆️" if v>0 else "⬇️")

                sug = get_ai_portfolio_suggestion(snap, risk, horizon, capital, age)
                st.markdown("### 💼 AI Suggested Allocation")
                cols = st.columns(5)
                for k,col in zip(["Stocks","Bonds","Gold","Crypto","Cash"], cols):
                    col.metric(k, f"{sug.get(k, 0)}%")

                st.info("💬 " + sug.get("Reasoning", "AI suggestion generated."))

                alloc_labels = [k for k in ["Stocks","Bonds","Gold","Crypto","Cash"] if sug.get(k, 0) > 0]
                alloc_values = [sug[k] for k in alloc_labels]

                fig = go.Figure(data=[go.Pie(labels=alloc_labels, values=alloc_values, hole=0.4)])
                fig.update_layout(title="AI Portfolio Mix", height=360, paper_bgcolor='#1f2937', plot_bgcolor='#1f2937', font_color='white')
                st.plotly_chart(fig, use_container_width=True)

                if st.button("🔄 Refresh Market Data"):
                    st.rerun()

                st.markdown("<p style='text-align:center; color:#9ca3af; margin-top:20px;'>Auto-refreshing in 60 seconds...</p>", unsafe_allow_html=True)
                time.sleep(60)
                st.rerun()

    @staticmethod
    def debt_repayment_flow():
        """Interactive debt repayment planning flow."""
        pass

    @staticmethod
    def retirement_planning_flow():
        """Interactive retirement planning flow."""
        pass

def main():
    """Main application function"""
    if TEST_MODE:
        return

    st.markdown('<h1 class="main-header">🦙 AI Financial Advisor - LLAMA 3.3</h1>', unsafe_allow_html=True)

    st.info("💡 **Disclaimer**: AI suggestions are educational only and not financial advice. Always consult with a qualified financial professional for personalized guidance.")

    st.sidebar.subheader("📊 Financial Tools")
    menu = st.sidebar.selectbox(
        "Choose Section",
        ["Demo Dashboard", "Budgeting", "Investments", "Debt", "Retirement"]
    )

    if "selected_page" not in st.session_state:
        st.session_state.selected_page = menu

    if menu != st.session_state.selected_page:
        st.session_state.selected_page = menu
        st.rerun()

    if st.session_state.selected_page == "Demo Dashboard":
        FinancialFlows.demo_dashboard()
    elif st.session_state.selected_page == "Budgeting":
        FinancialFlows.budgeting_flow()
    elif st.session_state.selected_page == "Investments":
        FinancialFlows.investing_flow()
    elif st.session_state.selected_page == "Debt":
        FinancialFlows.debt_repayment_flow()
    elif st.session_state.selected_page == "Retirement":
        FinancialFlows.retirement_planning_flow()

if __name__ == "__main__":
    main()
