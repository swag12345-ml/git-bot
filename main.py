"""
AI Financial Advisor Application - LLAMA 3.3
A comprehensive financial planning tool with AI-powered insights

Required pip packages:
pip install streamlit plotly pandas numpy python-dotenv langchain-groq yfinance feedparser
"""

import streamlit as st
import os
import json
import sys
from dotenv import load_dotenv
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from typing import Dict, List, Any
from langchain_groq import ChatGroq

try:
    import yfinance as yf
    import feedparser
    YFINANCE_AVAILABLE = True
except ImportError:
    YFINANCE_AVAILABLE = False

# Test mode check
TEST_MODE = "--test" in sys.argv

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

            # Handle AI score as string
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

        # Benchmark ratio comparison
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

        # Auto-flag overspending categories
        for category, ideal_pct in ideal_ratios.items():
            if category in expenses:
                actual_pct = (expenses[category] / income * 100) if income > 0 else 0
                if actual_pct > ideal_pct * 1.20:  # 20% above benchmark
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

        # Normalize allocation to ensure it sums to 100%
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
        # Dynamic projection intervals including user's chosen horizon
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

            # Sort debts based on strategy at the beginning
            if strategy == 'avalanche':
                debts_sim.sort(key=lambda x: x['interest_rate'], reverse=True)
            else:
                debts_sim.sort(key=lambda x: x['balance'])

            # Track per-debt metrics
            debt_stats = {i: {'months': 0, 'interest': 0.0, 'name': debts_sim[i]['name'],
                              'balance': debts_sim[i]['balance'], 'rate': debts_sim[i]['interest_rate'],
                              'min_pay': debts_sim[i]['minimum_payment']}
                          for i in range(len(debts_sim))}

            months = 0
            total_interest = 0.0
            current_extra = extra_amt

            while any(d['balance'] > 0.01 for d in debts_sim):
                # Find first unpaid debt (priority debt)
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

                    # Once a debt is paid, roll over its payment to extra
                    if d['balance'] <= 0.01 and debt_stats[i]['months'] == months + 1:
                        current_extra += d['minimum_payment']

                months += 1
                if months > 1000:
                    break

            # Build payoff plan from debt_stats
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

        # Run both cases: minimum payments only vs. with extra payment
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

        # Adjust subplot annotations positioning
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

        # Adjust subplot annotations positioning
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

        # Adjust subplot annotations positioning
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

        # Adjust subplot annotations positioning
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
        if not TEST_MODE:
            st.markdown('<div class="flow-card"><h2>💰 Smart Budgeting Assistant</h2><p>Let\'s create a comprehensive budget plan tailored to your financial situation.</p></div>', unsafe_allow_html=True)

        if 'budget_form_data' not in st.session_state:
            st.session_state.budget_form_data = {}

        # Reset form functionality
        if not TEST_MODE:
            if st.button("🔄 Reset Form"):
                for key in list(st.session_state.keys()):
                    if key.endswith("_form_data") or key.startswith("expense_"):
                        del st.session_state[key]
                st.rerun()

        if not TEST_MODE:
            with st.form("budget_form"):
                st.subheader("Step 1: Monthly Income")
                col1, col2 = st.columns(2)

                with col1:
                    primary_income = st.number_input("Primary Income (after taxes)", min_value=0.0, value=5000.0, step=100.0)
                    secondary_income = st.number_input("Secondary Income", min_value=0.0, value=0.0, step=100.0)

                with col2:
                    other_income = st.number_input("Other Income (investments, etc.)", min_value=0.0, value=0.0, step=100.0)
                    total_income = primary_income + secondary_income + other_income
                    st.metric("Total Monthly Income", f"${total_income:,.2f}")

                st.subheader("Step 2: Monthly Expenses")

                expense_categories = {
                    'housing': 'Housing (rent/mortgage, property tax)',
                    'utilities': 'Utilities (electricity, water, internet)',
                    'groceries': 'Groceries',
                    'transportation': 'Transportation (car payment, gas, public transit)',
                    'insurance': 'Insurance (health, auto, life)',
                    'healthcare': 'Healthcare (medical, dental)',
                    'dining_out': 'Dining Out & Entertainment',
                    'shopping': 'Shopping & Personal Care',
                    'subscriptions': 'Subscriptions & Memberships',
                    'savings': 'Savings & Investments',
                    'debt_payments': 'Debt Payments',
                    'other': 'Other Expenses'
                }

                expenses = {}
                col1, col2 = st.columns(2)

                for i, (key, label) in enumerate(expense_categories.items()):
                    with col1 if i % 2 == 0 else col2:
                        expenses[key] = st.number_input(label, min_value=0.0, value=0.0, step=50.0, key=f"expense_{key}")

                submitted = st.form_submit_button("Analyze My Budget", type="primary")
        else:
            submitted = True
            expenses = {
                'housing': 1500, 'utilities': 200, 'groceries': 400,
                'transportation': 300, 'insurance': 200, 'healthcare': 150,
                'dining_out': 300, 'shopping': 200, 'subscriptions': 50,
                'savings': 500, 'debt_payments': 300, 'other': 100
            }
            total_expenses = sum(expenses.values())
            total_income = total_expenses + 1000

        if submitted:
            st.session_state.budget_form_data = {
                'total_income': total_income,
                'expenses': expenses
            }

        if st.session_state.budget_form_data:
            form_data = st.session_state.budget_form_data
            budget_summary = FinancialCalculator.calculate_budget_summary(form_data['total_income'], form_data['expenses'])

            ai_insights = generate_ai_insights(budget_summary, "Budget Analysis")
            budget_summary["ai_score"] = ai_insights.get("ai_score")

            if not budget_summary["recommendations"]:
                budget_summary["recommendations"] = ["No personalized recommendations available."]

            if not TEST_MODE:
                # Aligned metrics in a single centered row
                col1, col2, col3, col4 = st.columns(4)

                with col1:
                    st.markdown(display_metric_card(
                        "Total Income",
                        f"${budget_summary['total_income']:,.2f}"
                    ), unsafe_allow_html=True)

                with col2:
                    st.markdown(display_metric_card(
                        "Total Expenses",
                        f"${budget_summary['total_expenses']:,.2f}"
                    ), unsafe_allow_html=True)

                with col3:
                    savings_color = 'green' if budget_summary["savings"] >= 0 else 'red'
                    st.markdown(display_metric_card(
                        "Monthly Savings",
                        f"${budget_summary['savings']:,.2f}",
                        color=savings_color
                    ), unsafe_allow_html=True)

                with col4:
                    ai_score = ai_insights.get("ai_score")
                    display_score = ai_score if ai_score is not None else budget_summary['health_score']
                    st.markdown(display_metric_card(
                        "AI Financial Score",
                        f"{display_score:.0f}/100"
                    ), unsafe_allow_html=True)

                # Combine AI and calculated recommendations
                all_recommendations = []
                ai_recommendations = ai_insights.get("ai_recommendations", [])
                if ai_recommendations:
                    all_recommendations.extend(ai_recommendations)
                if budget_summary["recommendations"]:
                    # Add calculated recommendations that aren't duplicates
                    for rec in budget_summary["recommendations"]:
                        if not any(rec.lower() in ai_rec.lower() for ai_rec in ai_recommendations):
                            all_recommendations.append(rec)

                if not all_recommendations:
                    all_recommendations = ["No personalized recommendations available."]

                recommendations_html = "".join([f"<li>{rec}</li>" for rec in all_recommendations])
                ai_display_score = ai_score if ai_score is not None else budget_summary["health_score"]
                ai_reasoning = ai_insights.get("ai_reasoning", "")

                st.markdown(f'''
                <div class="summary-card">
                    <h3>Financial Health: {budget_summary["financial_health"]} (AI Score: {ai_display_score:.0f}/100)</h3>
                    <p><strong>AI Analysis:</strong> {ai_reasoning}</p>
                    <h4>Personalized Recommendations:</h4>
                    <ul>
                        {recommendations_html}
                    </ul>
                </div>
                ''', unsafe_allow_html=True)

                # Enhanced Savings projection chart
                st.markdown("<div style='margin-top:20px;'></div>", unsafe_allow_html=True)
                st.subheader("📊 Savings Projection (Next 12 Months)")

                monthly_savings = budget_summary['savings']
                months = list(range(1, 13))
                month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']

                # Calculate projections with compound interest assumption (conservative 2% annual = 0.167% monthly)
                monthly_interest_rate = 0.02 / 12
                cumulative_savings = []
                cumulative_with_interest = []

                for month in months:
                    # Simple accumulation
                    cumulative_savings.append(monthly_savings * month)

                    # With conservative interest
                    if monthly_savings > 0:
                        # Future value of series with interest
                        fv = monthly_savings * (((1 + monthly_interest_rate) ** month - 1) / monthly_interest_rate)
                        cumulative_with_interest.append(fv)
                    else:
                        cumulative_with_interest.append(monthly_savings * month)

                # Create figure with detailed information
                fig_projection = go.Figure()

                # Add main savings line
                fig_projection.add_trace(go.Scatter(
                    x=months,
                    y=cumulative_savings,
                    mode='lines+markers',
                    name='Without Interest',
                    line=dict(color='#3b82f6', width=3),
                    marker=dict(size=10, symbol='circle'),
                    hovertemplate='<b>Month %{x}</b><br>' +
                                  'Cumulative: $%{y:,.2f}<br>' +
                                  '<extra></extra>'
                ))

                # Add savings with interest line
                fig_projection.add_trace(go.Scatter(
                    x=months,
                    y=cumulative_with_interest,
                    mode='lines+markers',
                    name='With 2% Interest',
                    line=dict(color='#4ade80', width=3, dash='dash'),
                    marker=dict(size=10, symbol='diamond'),
                    hovertemplate='<b>Month %{x}</b><br>' +
                                  'With Interest: $%{y:,.2f}<br>' +
                                  '<extra></extra>'
                ))

                # Add milestone markers at 3, 6, 9, and 12 months
                milestones = [3, 6, 9, 12]
                for milestone in milestones:
                    fig_projection.add_annotation(
                        x=milestone,
                        y=cumulative_with_interest[milestone-1],
                        text=f"${cumulative_with_interest[milestone-1]:,.0f}",
                        showarrow=True,
                        arrowhead=2,
                        arrowsize=1,
                        arrowwidth=2,
                        arrowcolor="#4ade80",
                        ax=0,
                        ay=-40,
                        font=dict(size=11, color='#4ade80', family='Arial Black'),
                        bgcolor='#1f2937',
                        bordercolor='#4ade80',
                        borderwidth=2,
                        borderpad=4
                    )

                fig_projection.update_layout(
                    title={
                        'text': "Estimated Savings Growth Based on Current Monthly Savings",
                        'x': 0.5,
                        'xanchor': 'center',
                        'font': {'size': 16, 'color': 'white'}
                    },
                    xaxis_title="Month",
                    yaxis_title="Cumulative Savings ($)",
                    height=500,
                    paper_bgcolor='#1f2937',
                    plot_bgcolor='#1f2937',
                    font_color='white',
                    hovermode='x unified',
                    showlegend=True,
                    legend=dict(
                        orientation="h",
                        yanchor="bottom",
                        y=1.02,
                        xanchor="center",
                        x=0.5,
                        font=dict(size=12, color='white')
                    ),
                    xaxis=dict(
                        tickmode='array',
                        tickvals=months,
                        ticktext=month_names,
                        gridcolor='#374151',
                        showgrid=True
                    ),
                    yaxis=dict(
                        gridcolor='#374151',
                        showgrid=True,
                        tickformat='$,.0f'
                    )
                )

                st.plotly_chart(fig_projection, use_container_width=True, config={"displayModeBar": False})

                # Add detailed breakdown table
                col1, col2, col3 = st.columns(3)

                with col1:
                    st.markdown(f'''
                    <div class="metric-card">
                        <h4>💰 Monthly Contribution</h4>
                        <h2>${monthly_savings:,.2f}</h2>
                        <p>Your current monthly savings</p>
                    </div>
                    ''', unsafe_allow_html=True)

                with col2:
                    year_total = cumulative_savings[-1]
                    st.markdown(f'''
                    <div class="metric-card">
                        <h4>📅 12-Month Total</h4>
                        <h2>${year_total:,.2f}</h2>
                        <p>Without interest earnings</p>
                    </div>
                    ''', unsafe_allow_html=True)

                with col3:
                    year_total_interest = cumulative_with_interest[-1]
                    interest_earned = year_total_interest - year_total
                    st.markdown(f'''
                    <div class="metric-card">
                        <h4>🎯 With Interest (2%)</h4>
                        <h2>${year_total_interest:,.2f}</h2>
                        <p>Interest earned: ${interest_earned:,.2f}</p>
                    </div>
                    ''', unsafe_allow_html=True)

                # Add quarterly breakdown
                st.markdown("<div style='margin-top:15px;'></div>", unsafe_allow_html=True)
                with st.expander("📋 View Quarterly Breakdown", expanded=False):
                    quarters_data = {
                        'Quarter': ['Q1 (Jan-Mar)', 'Q2 (Apr-Jun)', 'Q3 (Jul-Sep)', 'Q4 (Oct-Dec)'],
                        'Months': ['1-3', '4-6', '7-9', '10-12'],
                        'Savings (No Interest)': [
                            f"${cumulative_savings[2]:,.2f}",
                            f"${cumulative_savings[5]:,.2f}",
                            f"${cumulative_savings[8]:,.2f}",
                            f"${cumulative_savings[11]:,.2f}"
                        ],
                        'Savings (With Interest)': [
                            f"${cumulative_with_interest[2]:,.2f}",
                            f"${cumulative_with_interest[5]:,.2f}",
                            f"${cumulative_with_interest[8]:,.2f}",
                            f"${cumulative_with_interest[11]:,.2f}"
                        ]
                    }
                    quarters_df = pd.DataFrame(quarters_data)
                    st.dataframe(quarters_df, use_container_width=True, hide_index=True)

                st.markdown("<div style='margin-top:15px;'></div>", unsafe_allow_html=True)
                st.subheader("Budget Analysis Dashboard")
                budget_viz = FinancialVisualizer.plot_budget_summary(budget_summary)
                st.plotly_chart(budget_viz, use_container_width=True, config={"displayModeBar": False})

                st.session_state.budget_data = budget_summary
                st.session_state.budget_ai_insights = ai_insights

            return budget_summary

    @staticmethod
    def investing_flow():
        """Real-time AI-driven multi-asset investment dashboard using live Yahoo Finance data."""
        if not TEST_MODE:
            st.markdown('<div class="flow-card"><h2>📈 Investment Portfolio Dashboard</h2><p>Real-time AI-driven multi-asset investment tracker with live Yahoo Finance data and LLaMA insights.</p></div>', unsafe_allow_html=True)

        if not YFINANCE_AVAILABLE:
            st.error("Yahoo Finance integration requires 'yfinance' and 'feedparser' packages. Please install them: pip install yfinance feedparser")
            return None

        if 'portfolio_holdings' not in st.session_state:
            st.session_state.portfolio_holdings = []
        if 'last_refresh' not in st.session_state:
            st.session_state.last_refresh = None

        col_reset, col_refresh = st.columns([6, 1])
        with col_reset:
            st.subheader("Step 1: Build Your Portfolio")
        with col_refresh:
            if st.button("🔄 Refresh Data", key="refresh_portfolio"):
                st.session_state.last_refresh = datetime.now()
                st.rerun()

        if st.session_state.last_refresh:
            st.caption(f"Last updated: {st.session_state.last_refresh.strftime('%Y-%m-%d %H:%M:%S')}")

        with st.expander("➕ Add Asset to Portfolio", expanded=len(st.session_state.portfolio_holdings) == 0):
            col1, col2 = st.columns(2)

            with col1:
                asset_type = st.selectbox(
                    "Asset Type",
                    ["Stock", "Bond", "Cash", "Gold", "Crypto", "ETF"],
                    key="new_asset_type"
                )

                if asset_type == "Stock":
                    symbol_help = "Examples: AAPL, MSFT, TSLA, GOOGL"
                elif asset_type == "Bond":
                    symbol_help = "Examples: ^TNX (10Y US Treasury), ^IRX (3M T-Bill)"
                elif asset_type == "Cash":
                    symbol_help = "Examples: INR=X (USD/INR), DXY (USD Index)"
                elif asset_type == "Gold":
                    symbol_help = "Examples: XAUUSD=X (Gold), XAGUSD=X (Silver), CL=F (Crude Oil)"
                elif asset_type == "Crypto":
                    symbol_help = "Examples: BTC-USD, ETH-USD, SOL-USD"
                else:
                    symbol_help = "Examples: SPY, QQQ, VTI, VOO"

                asset_symbol = st.text_input(
                    "Symbol",
                    placeholder="e.g., AAPL",
                    help=symbol_help,
                    key="new_asset_symbol"
                )

            with col2:
                asset_amount = st.number_input(
                    "Amount Invested ($)",
                    min_value=0.0,
                    step=100.0,
                    value=1000.0,
                    key="new_asset_amount"
                )

                purchase_date = st.date_input(
                    "Purchase Date (optional)",
                    value=datetime.now().date(),
                    max_value=datetime.now().date(),
                    key="new_asset_date"
                )

            if st.button("Add to Portfolio", type="primary", key="add_asset_button"):
                if asset_symbol:
                    try:
                        ticker = yf.Ticker(asset_symbol.upper())
                        info = ticker.info
                        current_price = info.get('currentPrice') or info.get('regularMarketPrice') or info.get('previousClose', 0)

                        if current_price > 0:
                            units = asset_amount / current_price if current_price > 0 else 0
                            st.session_state.portfolio_holdings.append({
                                'type': asset_type,
                                'symbol': asset_symbol.upper(),
                                'amount_invested': asset_amount,
                                'units': units,
                                'purchase_date': purchase_date.strftime('%Y-%m-%d'),
                                'purchase_price': current_price
                            })
                            st.success(f"Added {asset_symbol.upper()} to portfolio!")
                            st.rerun()
                        else:
                            st.error(f"Could not fetch price for {asset_symbol}. Please verify the symbol.")
                    except Exception as e:
                        st.error(f"Error adding asset: {str(e)}")
                else:
                    st.warning("Please enter a symbol")

        if st.session_state.portfolio_holdings:
            st.markdown("---")
            st.subheader("Step 2: Real-Time Portfolio Overview")

            total_portfolio_value = 0
            total_invested = 0
            portfolio_data = []

            for holding in st.session_state.portfolio_holdings:
                try:
                    ticker = yf.Ticker(holding['symbol'])
                    info = ticker.info
                    hist = ticker.history(period="5d")

                    current_price = info.get('currentPrice') or info.get('regularMarketPrice') or info.get('previousClose', 0)
                    if current_price == 0 and not hist.empty:
                        current_price = hist['Close'].iloc[-1]

                    prev_close = info.get('previousClose', current_price)
                    if len(hist) >= 2:
                        prev_close = hist['Close'].iloc[-2]

                    pct_change = ((current_price - prev_close) / prev_close * 100) if prev_close > 0 else 0
                    current_value = holding['units'] * current_price
                    gain_loss = current_value - holding['amount_invested']
                    gain_loss_pct = (gain_loss / holding['amount_invested'] * 100) if holding['amount_invested'] > 0 else 0

                    market_cap = info.get('marketCap', 0)

                    portfolio_data.append({
                        'symbol': holding['symbol'],
                        'type': holding['type'],
                        'units': holding['units'],
                        'purchase_price': holding['purchase_price'],
                        'current_price': current_price,
                        'invested': holding['amount_invested'],
                        'current_value': current_value,
                        'daily_change_pct': pct_change,
                        'gain_loss': gain_loss,
                        'gain_loss_pct': gain_loss_pct,
                        'market_cap': market_cap
                    })

                    total_portfolio_value += current_value
                    total_invested += holding['amount_invested']
                except Exception as e:
                    st.warning(f"Could not fetch data for {holding['symbol']}: {str(e)}")

            total_gain_loss = total_portfolio_value - total_invested
            total_gain_loss_pct = (total_gain_loss / total_invested * 100) if total_invested > 0 else 0

            col1, col2, col3, col4 = st.columns(4)

            with col1:
                st.markdown(display_metric_card(
                    "Total Portfolio Value",
                    f"${total_portfolio_value:,.2f}"
                ), unsafe_allow_html=True)

            with col2:
                st.markdown(display_metric_card(
                    "Total Invested",
                    f"${total_invested:,.2f}"
                ), unsafe_allow_html=True)

            with col3:
                gain_color = 'green' if total_gain_loss >= 0 else 'red'
                st.markdown(display_metric_card(
                    "Total Gain/Loss",
                    f"${total_gain_loss:,.2f}",
                    f"{total_gain_loss_pct:+.2f}%",
                    color=gain_color
                ), unsafe_allow_html=True)

            with col4:
                num_assets = len(st.session_state.portfolio_holdings)
                num_types = len(set(h['type'] for h in st.session_state.portfolio_holdings))
                st.markdown(display_metric_card(
                    "Holdings",
                    f"{num_assets} assets",
                    f"{num_types} asset types"
                ), unsafe_allow_html=True)

            if st.button("🗑️ Clear Portfolio", key="clear_portfolio"):
                st.session_state.portfolio_holdings = []
                st.rerun()

            st.markdown("---")
            st.subheader("Step 3: Holdings Detail")

            holdings_df = pd.DataFrame(portfolio_data)
            if not holdings_df.empty:
                display_df = holdings_df.copy()
                display_df['Purchase Price'] = display_df['purchase_price'].apply(lambda x: f"${x:,.2f}")
                display_df['Current Price'] = display_df['current_price'].apply(lambda x: f"${x:,.2f}")
                display_df['Invested'] = display_df['invested'].apply(lambda x: f"${x:,.2f}")
                display_df['Current Value'] = display_df['current_value'].apply(lambda x: f"${x:,.2f}")
                display_df['Daily Change'] = display_df['daily_change_pct'].apply(lambda x: f"{x:+.2f}%")
                display_df['Total Gain/Loss'] = display_df.apply(
                    lambda row: f"${row['gain_loss']:,.2f} ({row['gain_loss_pct']:+.2f}%)", axis=1
                )

                display_cols = ['symbol', 'type', 'Purchase Price', 'Current Price', 'Invested', 'Current Value', 'Daily Change', 'Total Gain/Loss']
                display_df = display_df[display_cols]
                display_df.columns = ['Symbol', 'Type', 'Purchase Price', 'Current Price', 'Invested', 'Current Value', 'Daily Change', 'Total Gain/Loss']

                st.dataframe(display_df, use_container_width=True, hide_index=True)

            st.markdown("---")
            st.subheader("Step 4: Asset-Class Performance Breakdown")

            col_pie, col_bar = st.columns(2)

            with col_pie:
                type_allocation = holdings_df.groupby('type')['current_value'].sum().to_dict()

                fig_pie = go.Figure(data=[go.Pie(
                    labels=list(type_allocation.keys()),
                    values=list(type_allocation.values()),
                    hole=0.4,
                    textposition='inside',
                    textinfo='percent+label'
                )])
                fig_pie.update_layout(
                    title="Portfolio Allocation by Asset Type",
                    height=400,
                    paper_bgcolor='#1f2937',
                    plot_bgcolor='#1f2937',
                    font_color='white',
                    showlegend=True
                )
                st.plotly_chart(fig_pie, use_container_width=True, config={"displayModeBar": False})

            with col_bar:
                type_performance = holdings_df.groupby('type')['daily_change_pct'].mean().to_dict()

                fig_bar = go.Figure(data=[go.Bar(
                    x=list(type_performance.keys()),
                    y=list(type_performance.values()),
                    marker_color=['green' if v >= 0 else 'red' for v in type_performance.values()]
                )])
                fig_bar.update_layout(
                    title="Average Daily % Change by Asset Class",
                    xaxis_title="Asset Type",
                    yaxis_title="Daily Change (%)",
                    height=400,
                    paper_bgcolor='#1f2937',
                    plot_bgcolor='#1f2937',
                    font_color='white'
                )
                st.plotly_chart(fig_bar, use_container_width=True, config={"displayModeBar": False})

            st.markdown("---")
            st.subheader("Step 5: Technical Analysis")

            selected_symbol = st.selectbox(
                "Select Asset for Detailed Analysis",
                [h['symbol'] for h in st.session_state.portfolio_holdings],
                key="selected_analysis_symbol"
            )

            if selected_symbol:
                try:
                    ticker = yf.Ticker(selected_symbol)
                    hist_1y = ticker.history(period="1y")
                    hist_90d = ticker.history(period="3mo")
                    hist_30d = ticker.history(period="1mo")

                    if not hist_1y.empty:
                        hist_1y['SMA_20'] = hist_1y['Close'].rolling(window=20).mean()
                        hist_1y['SMA_50'] = hist_1y['Close'].rolling(window=50).mean()

                        delta = hist_1y['Close'].diff()
                        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
                        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
                        rs = gain / loss
                        hist_1y['RSI'] = 100 - (100 / (1 + rs))

                        fig_tech = make_subplots(
                            rows=2, cols=1,
                            shared_xaxes=True,
                            vertical_spacing=0.1,
                            row_heights=[0.7, 0.3],
                            subplot_titles=(f'{selected_symbol} Price & Moving Averages', 'RSI')
                        )

                        fig_tech.add_trace(
                            go.Candlestick(
                                x=hist_1y.index,
                                open=hist_1y['Open'],
                                high=hist_1y['High'],
                                low=hist_1y['Low'],
                                close=hist_1y['Close'],
                                name='Price'
                            ),
                            row=1, col=1
                        )

                        fig_tech.add_trace(
                            go.Scatter(x=hist_1y.index, y=hist_1y['SMA_20'], name='SMA 20', line=dict(color='orange')),
                            row=1, col=1
                        )

                        fig_tech.add_trace(
                            go.Scatter(x=hist_1y.index, y=hist_1y['SMA_50'], name='SMA 50', line=dict(color='blue')),
                            row=1, col=1
                        )

                        fig_tech.add_trace(
                            go.Scatter(x=hist_1y.index, y=hist_1y['RSI'], name='RSI', line=dict(color='purple')),
                            row=2, col=1
                        )

                        fig_tech.add_hline(y=70, line_dash="dash", line_color="red", row=2, col=1)
                        fig_tech.add_hline(y=30, line_dash="dash", line_color="green", row=2, col=1)

                        fig_tech.update_layout(
                            height=700,
                            paper_bgcolor='#1f2937',
                            plot_bgcolor='#1f2937',
                            font_color='white',
                            xaxis_rangeslider_visible=False,
                            showlegend=True
                        )

                        st.plotly_chart(fig_tech, use_container_width=True, config={"displayModeBar": False})

                        col_perf1, col_perf2, col_perf3 = st.columns(3)

                        with col_perf1:
                            ret_30d = ((hist_30d['Close'].iloc[-1] - hist_30d['Close'].iloc[0]) / hist_30d['Close'].iloc[0] * 100) if len(hist_30d) > 0 else 0
                            st.markdown(display_metric_card(
                                "30-Day Return",
                                f"{ret_30d:+.2f}%",
                                color='green' if ret_30d >= 0 else 'red'
                            ), unsafe_allow_html=True)

                        with col_perf2:
                            ret_90d = ((hist_90d['Close'].iloc[-1] - hist_90d['Close'].iloc[0]) / hist_90d['Close'].iloc[0] * 100) if len(hist_90d) > 0 else 0
                            st.markdown(display_metric_card(
                                "90-Day Return",
                                f"{ret_90d:+.2f}%",
                                color='green' if ret_90d >= 0 else 'red'
                            ), unsafe_allow_html=True)

                        with col_perf3:
                            ret_1y = ((hist_1y['Close'].iloc[-1] - hist_1y['Close'].iloc[0]) / hist_1y['Close'].iloc[0] * 100) if len(hist_1y) > 0 else 0
                            st.markdown(display_metric_card(
                                "1-Year Return",
                                f"{ret_1y:+.2f}%",
                                color='green' if ret_1y >= 0 else 'red'
                            ), unsafe_allow_html=True)

                        volatility = hist_1y['Close'].pct_change().std() * np.sqrt(252) * 100
                        returns_mean = hist_1y['Close'].pct_change().mean() * 252
                        sharpe = (returns_mean / (volatility/100)) if volatility > 0 else 0

                        sp500 = yf.Ticker("^GSPC")
                        sp500_hist = sp500.history(period="1y")
                        if not sp500_hist.empty and len(sp500_hist) == len(hist_1y):
                            asset_returns = hist_1y['Close'].pct_change().dropna()
                            sp500_returns = sp500_hist['Close'].pct_change().dropna()
                            covariance = np.cov(asset_returns, sp500_returns)[0][1]
                            sp500_variance = np.var(sp500_returns)
                            beta = covariance / sp500_variance if sp500_variance > 0 else 1.0
                        else:
                            beta = 1.0

                        col_metric1, col_metric2, col_metric3 = st.columns(3)

                        with col_metric1:
                            st.markdown(display_metric_card(
                                "Volatility",
                                f"{volatility:.2f}%",
                                "Annualized"
                            ), unsafe_allow_html=True)

                        with col_metric2:
                            st.markdown(display_metric_card(
                                "Sharpe Ratio",
                                f"{sharpe:.2f}",
                                "Risk-adjusted return"
                            ), unsafe_allow_html=True)

                        with col_metric3:
                            st.markdown(display_metric_card(
                                "Beta (vs S&P 500)",
                                f"{beta:.2f}",
                                "Market correlation"
                            ), unsafe_allow_html=True)

                except Exception as e:
                    st.error(f"Error fetching technical data: {str(e)}")

            st.markdown("---")
            st.subheader("Step 6: Global Market Benchmarks")

            benchmarks = {
                'S&P 500': '^GSPC',
                'NASDAQ': '^IXIC',
                'Gold': 'GC=F',
                'Bitcoin': 'BTC-USD',
                '10Y Treasury': '^TNX',
                'USD Index': 'DX-Y.NYB'
            }

            benchmark_data = []
            for name, symbol in benchmarks.items():
                try:
                    ticker = yf.Ticker(symbol)
                    hist = ticker.history(period="5d")
                    if not hist.empty and len(hist) >= 2:
                        current = hist['Close'].iloc[-1]
                        prev = hist['Close'].iloc[-2]
                        pct_change = ((current - prev) / prev * 100) if prev > 0 else 0
                        benchmark_data.append({
                            'name': name,
                            'current': current,
                            'change_pct': pct_change
                        })
                except:
                    pass

            if benchmark_data:
                cols = st.columns(len(benchmark_data))
                for idx, (col, bm) in enumerate(zip(cols, benchmark_data)):
                    with col:
                        color = 'green' if bm['change_pct'] >= 0 else 'red'
                        st.markdown(display_metric_card(
                            bm['name'],
                            f"${bm['current']:,.2f}" if bm['current'] < 10000 else f"${bm['current']:,.0f}",
                            f"{bm['change_pct']:+.2f}%",
                            color=color
                        ), unsafe_allow_html=True)

            st.markdown("---")
            st.subheader("Step 7: AI Portfolio Insights")

            if st.button("🤖 Generate AI Insights", type="primary", key="generate_ai_insights"):
                with st.spinner("Analyzing your portfolio with LLaMA..."):
                    portfolio_analysis_data = {
                        'total_value': total_portfolio_value,
                        'total_invested': total_invested,
                        'total_gain_loss': total_gain_loss,
                        'total_gain_loss_pct': total_gain_loss_pct,
                        'holdings': portfolio_data,
                        'asset_allocation': type_allocation,
                        'num_assets': num_assets,
                        'num_types': num_types
                    }

                    ai_insights = generate_ai_insights(portfolio_analysis_data, "Investment Analysis")
                    st.session_state.portfolio_ai_insights = ai_insights
                    st.rerun()

            if 'portfolio_ai_insights' in st.session_state and st.session_state.portfolio_ai_insights:
                display_ai_suggestions(st.session_state.portfolio_ai_insights, "Investment Analysis")

            st.markdown("---")
            st.subheader("Step 8: Portfolio News Feed")

            news_data = []
            for holding in st.session_state.portfolio_holdings[:5]:
                try:
                    ticker = yf.Ticker(holding['symbol'])
                    news = ticker.news
                    if news:
                        for item in news[:2]:
                            news_data.append({
                                'symbol': holding['symbol'],
                                'title': item.get('title', 'No title'),
                                'link': item.get('link', '#'),
                                'publisher': item.get('publisher', 'Unknown'),
                                'timestamp': datetime.fromtimestamp(item.get('providerPublishTime', 0)).strftime('%Y-%m-%d %H:%M') if item.get('providerPublishTime') else 'Unknown'
                            })
                except:
                    pass

            if news_data:
                for news in news_data[:10]:
                    with st.expander(f"[{news['symbol']}] {news['title']}", expanded=False):
                        st.write(f"**Publisher:** {news['publisher']}")
                        st.write(f"**Published:** {news['timestamp']}")
                        st.write(f"[Read more]({news['link']})")
            else:
                st.info("No recent news available for your portfolio holdings.")

            return {
                'portfolio_value': total_portfolio_value,
                'total_invested': total_invested,
                'gain_loss': total_gain_loss,
                'holdings': portfolio_data
            }
        else:
            st.info("Add assets to your portfolio to get started!")
            return None

    @staticmethod
    def debt_repayment_flow():
        """Interactive debt repayment planning flow."""
        if not TEST_MODE:
            st.markdown('<div class="flow-card"><h2>💳 Debt Freedom Planner</h2><p>Let\'s create a strategic plan to eliminate your debt efficiently.</p></div>', unsafe_allow_html=True)

        if 'debts' not in st.session_state:
            st.session_state.debts = []

        # Reset form functionality
        if not TEST_MODE:
            if st.button("🔄 Reset Form", key="reset_debt"):
                for key in list(st.session_state.keys()):
                    if key.endswith("_form_data") or key.startswith("expense_") or key == "debts":
                        del st.session_state[key]
                st.rerun()

        if not TEST_MODE:
            st.subheader("Step 1: Your Current Debts")

            with st.expander("Add New Debt", expanded=len(st.session_state.debts) == 0):
                col1, col2 = st.columns(2)

                with col1:
                    debt_name = st.text_input("Debt Name (e.g., Credit Card, Student Loan)")
                    debt_balance = st.number_input("Current Balance", min_value=0.0, step=100.0)

                with col2:
                    interest_rate = st.number_input("Interest Rate (%)", min_value=0.0, max_value=50.0, step=0.1)
                    minimum_payment = st.number_input("Minimum Monthly Payment", min_value=0.0, step=10.0)

                if st.button("Add Debt"):
                    if debt_name and debt_balance > 0:
                        st.session_state.debts.append({
                            'name': debt_name,
                            'balance': debt_balance,
                            'interest_rate': interest_rate,
                            'minimum_payment': minimum_payment
                        })
                        st.success(f"Added {debt_name} to your debt list!")
                        st.rerun()

            if st.session_state.debts:
                st.subheader("Your Current Debts")
                debt_df = pd.DataFrame(st.session_state.debts)
                debt_df['Balance'] = debt_df['balance'].apply(lambda x: f"${x:,.2f}")
                debt_df['Interest Rate'] = debt_df['interest_rate'].apply(lambda x: f"{x:.1f}%")
                debt_df['Min Payment'] = debt_df['minimum_payment'].apply(lambda x: f"${x:.2f}")

                display_df = debt_df[['name', 'Balance', 'Interest Rate', 'Min Payment']].copy()
                display_df.columns = ['Debt Name', 'Balance', 'Interest Rate', 'Min Payment']
                st.dataframe(display_df, use_container_width=True)

                if st.button("Clear All Debts"):
                    st.session_state.debts = []
                    st.rerun()
        else:
            st.session_state.debts = [
                {'name': 'Credit Card 1', 'balance': 5000, 'interest_rate': 18.0, 'minimum_payment': 150},
                {'name': 'Credit Card 2', 'balance': 3000, 'interest_rate': 22.0, 'minimum_payment': 100},
                {'name': 'Student Loan', 'balance': 15000, 'interest_rate': 6.0, 'minimum_payment': 180}
            ]

        if st.session_state.debts:
            if 'debt_form_data' not in st.session_state:
                st.session_state.debt_form_data = {}

            if not TEST_MODE:
                st.subheader("Step 2: Repayment Strategy")

                with st.form("debt_form"):
                    col1, col2 = st.columns(2)

                    with col1:
                        strategy = st.selectbox(
                            "Choose Repayment Strategy",
                            ["avalanche", "snowball"],
                            format_func=lambda x: "Debt Avalanche (Highest Interest First)" if x == "avalanche" else "Debt Snowball (Smallest Balance First)"
                        )

                    with col2:
                        extra_payment = st.number_input("Extra Monthly Payment Available", min_value=0.0, step=50.0)

                    submitted = st.form_submit_button("Create Debt Payoff Plan", type="primary")
            else:
                submitted = True
                strategy = 'avalanche'
                extra_payment = 200.0

            if submitted:
                st.session_state.debt_form_data = {
                    'strategy': strategy,
                    'extra_payment': extra_payment
                }

            if st.session_state.debt_form_data:
                form_data = st.session_state.debt_form_data
                debt_analysis = FinancialCalculator.calculate_debt_payoff(st.session_state.debts, form_data['extra_payment'], form_data['strategy'])

                ai_insights = generate_ai_insights(debt_analysis, "Debt Analysis")

                if not TEST_MODE:
                    col1, col2, col3, col4 = st.columns(4)

                    with col1:
                        st.markdown(display_metric_card(
                            "Total Debt",
                            f"${debt_analysis['total_debt']:,.2f}"
                        ), unsafe_allow_html=True)

                    with col2:
                        total_min_payment = sum(d['monthly_payment'] for d in debt_analysis['payoff_plan'])
                        st.markdown(display_metric_card(
                            "Min Payments",
                            f"${total_min_payment:,.2f}"
                        ), unsafe_allow_html=True)

                    with col3:
                        savings_color = 'green' if form_data['extra_payment'] > 0 else None
                        savings_value = f"${debt_analysis['interest_savings']:,.2f}" if form_data['extra_payment'] > 0 else "$0"
                        st.markdown(display_metric_card(
                            "Interest Savings",
                            savings_value,
                            color=savings_color
                        ), unsafe_allow_html=True)

                    with col4:
                        ai_score = ai_insights.get("ai_score")
                        if ai_score is not None:
                            st.markdown(display_metric_card(
                                "AI Score",
                                f"{ai_score}/100"
                            ), unsafe_allow_html=True)
                        else:
                            time_color = 'green' if form_data['extra_payment'] > 0 else None
                            time_value = f"{debt_analysis['time_savings_months']:.0f} months" if form_data['extra_payment'] > 0 else "0 months"
                            st.markdown(display_metric_card(
                                "Time Savings",
                                time_value,
                                color=time_color
                            ), unsafe_allow_html=True)

                    st.subheader("Debt Payoff Priority Order")
                    scenario_key = 'with_extra' if form_data['extra_payment'] > 0 else 'minimum_only'
                    payoff_plan = debt_analysis['scenarios'][scenario_key]['payoff_plan']

                    plan_df = pd.DataFrame(payoff_plan)
                    if not plan_df.empty:
                        plan_df['Balance'] = plan_df['balance'].apply(lambda x: f"${x:,.2f}")
                        plan_df['Interest Rate'] = plan_df['interest_rate'].apply(lambda x: f"{x:.1f}%")
                        plan_df['Monthly Payment'] = plan_df['monthly_payment'].apply(lambda x: f"${x:.2f}")
                        plan_df['Interest Paid'] = plan_df['interest_paid'].apply(lambda x: f"${x:,.2f}")

                        display_plan = plan_df[['priority', 'debt_name', 'Balance', 'Interest Rate', 'Monthly Payment', 'months_to_payoff', 'Interest Paid']].copy()
                        display_plan.columns = ['Priority', 'Debt Name', 'Balance', 'Interest Rate', 'Monthly Payment', 'Months to Payoff', 'Total Interest']
                        st.dataframe(display_plan, use_container_width=True)

                    strategy_text = "pay highest interest rates first" if form_data["strategy"] == "avalanche" else "pay smallest balances first"
                    st.markdown(f'''
                    <div class="summary-card">
                        <h3>Debt Payoff Recommendations</h3>
                        <ul>
                            <li>🎯 Focus on paying ${debt_analysis["recommended_extra_payment"]:.0f} extra per month if possible</li>
                            <li>📊 You're using the <strong>{form_data["strategy"].title()}</strong> method - {strategy_text}</li>
                            <li>💡 Consider debt consolidation if you have high-interest credit cards</li>
                            <li>🚫 Avoid taking on new debt during your payoff journey</li>
                            <li>📱 Set up automatic payments to stay on track</li>
                        </ul>
                    </div>
                    ''', unsafe_allow_html=True)

                    display_ai_suggestions(ai_insights, "Debt Analysis")

                    st.markdown("<div style='margin-top:15px;'></div>", unsafe_allow_html=True)
                    st.subheader("Debt Analysis Dashboard")
                    debt_viz = FinancialVisualizer.plot_debt_payoff(debt_analysis)
                    st.plotly_chart(debt_viz, use_container_width=True, config={"displayModeBar": False})

                    st.session_state.debt_data = debt_analysis
                    st.session_state.debt_ai_insights = ai_insights

                return debt_analysis

    @staticmethod
    def retirement_planning_flow():
        """Interactive retirement planning flow."""
        if not TEST_MODE:
            st.markdown('<div class="flow-card"><h2>🏖️ Retirement Planning Assistant</h2><p>Let\'s ensure you\'re on track for a comfortable retirement.</p></div>', unsafe_allow_html=True)

        if 'retirement_form_data' not in st.session_state:
            st.session_state.retirement_form_data = {}

        # Reset form functionality
        if not TEST_MODE:
            if st.button("🔄 Reset Form", key="reset_retirement"):
                for key in list(st.session_state.keys()):
                    if key.endswith("_form_data") or key.startswith("expense_"):
                        del st.session_state[key]
                st.rerun()

        if not TEST_MODE:
            with st.form("retirement_form"):
                st.subheader("Step 1: Current Financial Situation")
                col1, col2 = st.columns(2)

                with col1:
                    current_age = st.number_input("Current Age", min_value=18, max_value=80, value=35)
                    retirement_age = st.number_input("Desired Retirement Age", min_value=50, max_value=80, value=65)
                    current_income = st.number_input("Current Annual Income", min_value=0.0, value=75000.0, step=5000.0)

                with col2:
                    current_savings = st.number_input("Current Retirement Savings", min_value=0.0, value=50000.0, step=5000.0)
                    monthly_contribution = st.number_input("Current Monthly Contribution", min_value=0.0, value=500.0, step=50.0)
                    employer_match = st.number_input("Employer Match (monthly)", min_value=0.0, value=0.0, step=50.0)

                st.subheader("Step 2: Retirement Lifestyle Goals")

                lifestyle_choice = st.selectbox(
                    "Desired Retirement Lifestyle",
                    ["Basic (60% of current income)", "Comfortable (80% of current income)", "Luxurious (100% of current income)"]
                )

                col1, col2 = st.columns(2)

                with col1:
                    healthcare_inflation = st.checkbox("Account for higher healthcare costs", value=True)
                    social_security = st.checkbox("Include Social Security benefits", value=True)

                with col2:
                    inheritance_expected = st.number_input("Expected Inheritance", min_value=0.0, value=0.0, step=10000.0)
                    other_retirement_income = st.number_input("Other Retirement Income (monthly)", min_value=0.0, value=0.0, step=100.0)

                submitted = st.form_submit_button("Analyze Retirement Plan", type="primary")
        else:
            submitted = True
            current_age = 35
            retirement_age = 65
            current_income = 75000.0
            current_savings = 50000.0
            monthly_contribution = 500.0
            employer_match = 150.0

        if submitted:
            st.session_state.retirement_form_data = {
                'current_age': current_age,
                'retirement_age': retirement_age,
                'current_income': current_income,
                'current_savings': current_savings,
                'monthly_contribution': monthly_contribution,
                'employer_match': employer_match
            }

        if st.session_state.retirement_form_data:
            form_data = st.session_state.retirement_form_data
            total_monthly_contribution = form_data['monthly_contribution'] + form_data['employer_match']

            retirement_analysis = FinancialCalculator.calculate_retirement_needs(
                form_data['current_age'], form_data['retirement_age'], form_data['current_income'],
                form_data['current_savings'], total_monthly_contribution
            )

            # Check if there was an error in retirement calculation
            if "error" in retirement_analysis:
                if not TEST_MODE:
                    st.error(f"⚠️ {retirement_analysis['error']}")
                    st.warning("Please adjust your retirement age to be greater than your current age.")
                return retirement_analysis

            ai_insights = generate_ai_insights(retirement_analysis, "Retirement Analysis")

            if not TEST_MODE:
                col1, col2, col3, col4, col5 = st.columns(5)

                with col1:
                    st.markdown(display_metric_card(
                        "Years to Retirement",
                        str(retirement_analysis["years_to_retirement"])
                    ), unsafe_allow_html=True)

                with col2:
                    st.markdown(display_metric_card(
                        "Projected Savings",
                        f"${retirement_analysis['projected_savings']:,.0f}"
                    ), unsafe_allow_html=True)

                with col3:
                    st.markdown(display_metric_card(
                        "Retirement Goal",
                        f"${retirement_analysis['retirement_corpus_needed']:,.0f}"
                    ), unsafe_allow_html=True)

                # ✅ New Retirement Gap metric card
                gap = retirement_analysis.get("retirement_gap", 0)
                gap_text = f"${gap:,.0f}" if gap > 0 else "On Track!"
                gap_color = "red" if gap > 0 else "green"
                with col4:
                    st.markdown(display_metric_card("Retirement Gap", gap_text, color=gap_color), unsafe_allow_html=True)

                with col5:
                    ai_score = ai_insights.get("ai_score", 0)
                    st.markdown(display_metric_card("AI Score", f"{ai_score}/100"), unsafe_allow_html=True)

                st.subheader("Retirement Scenarios")
                scenarios = retirement_analysis['scenarios']

                scenario_df = pd.DataFrame({
                    'Scenario': ['Conservative', 'Current Plan', 'Aggressive'],
                    'Monthly Contribution': [f"${scenarios['conservative']['monthly_contribution']:.0f}",
                                           f"${scenarios['current']['monthly_contribution']:.0f}",
                                           f"${scenarios['aggressive']['monthly_contribution']:.0f}"],
                    'Projected Total': [f"${scenarios['conservative']['projected_total']:,.0f}",
                                      f"${scenarios['current']['projected_total']:,.0f}",
                                      f"${scenarios['aggressive']['projected_total']:,.0f}"],
                    'Monthly Retirement Income': [f"${scenarios['conservative']['monthly_retirement_income']:,.0f}",
                                                f"${scenarios['current']['monthly_retirement_income']:,.0f}",
                                                f"${scenarios['aggressive']['monthly_retirement_income']:,.0f}"],
                    'Income Replacement': [scenarios['conservative'].get('display_ratio', f"{scenarios['conservative']['replacement_ratio_achieved']:.1%}"),
                                         scenarios['current'].get('display_ratio', f"{scenarios['current']['replacement_ratio_achieved']:.1%}"),
                                         scenarios['aggressive'].get('display_ratio', f"{scenarios['aggressive']['replacement_ratio_achieved']:.1%}")]
                })

                st.dataframe(scenario_df, use_container_width=True)

                recommendations_html = "".join([f"<li>{rec}</li>" for rec in retirement_analysis["recommendations"]])
                st.markdown(f'''
                <div class="summary-card">
                    <h3>Retirement Planning Recommendations</h3>
                    <ul>
                        {recommendations_html}
                    </ul>
                </div>
                ''', unsafe_allow_html=True)

                display_ai_suggestions(ai_insights, "Retirement Analysis")

                gap = retirement_analysis["retirement_gap"]
                required_contrib = retirement_analysis["required_monthly_contribution"]
                current_contrib = total_monthly_contribution

                if gap > 0:
                    increase_needed = max(0, required_contrib - current_contrib)
                    if increase_needed > 0:
                        st.warning(f"⚠️ To meet your retirement goal, consider increasing your monthly contribution by ${increase_needed:.0f}")
                    else:
                        st.success("🎉 You are already on track, no increase needed")
                else:
                    st.success("🎉 Congratulations! You're on track to meet your retirement goals!")

                st.markdown("<div style='margin-top:15px;'></div>", unsafe_allow_html=True)
                st.subheader("Retirement Planning Dashboard")
                retirement_viz = FinancialVisualizer.plot_retirement_projections(retirement_analysis)
                st.plotly_chart(retirement_viz, use_container_width=True, config={"displayModeBar": False})

                st.session_state.retirement_data = retirement_analysis
                st.session_state.retirement_ai_insights = ai_insights

            return retirement_analysis

def run_tests():
    """Run test scenarios to validate functionality"""
    print("🧪 Running Financial App Tests...")

    print("\n📊 Test 1: Zero income budget")
    try:
        budget_result = FinancialCalculator.calculate_budget_summary(0, {'housing': 1000})
        assert budget_result['financial_health'] == 'Critical'
        assert budget_result['health_score'] == 0
        print("✅ PASS: Zero income handled correctly")
    except Exception as e:
        print(f"❌ FAIL: {e}")

    print("\n💰 Test 2: High savings budget")
    try:
        expenses = {'housing': 2000, 'utilities': 300, 'groceries': 400}
        budget_result = FinancialCalculator.calculate_budget_summary(8000, expenses)
        assert budget_result['savings_rate'] > 20
        assert budget_result['health_score'] >= 70
        print(f"✅ PASS: High savings rate {budget_result['savings_rate']:.1f}%, Health score: {budget_result['health_score']}")
    except Exception as e:
        print(f"❌ FAIL: {e}")

    print("\n💳 Test 3: High debt payoff analysis")
    try:
        debts = [
            {'name': 'Credit Card', 'balance': 10000, 'interest_rate': 24.0, 'minimum_payment': 300}
        ]
        debt_result = FinancialCalculator.calculate_debt_payoff(debts, 200, 'avalanche')
        assert debt_result['scenarios']['minimum_only']['total_months'] > 24
        assert debt_result['interest_savings'] > 0
        print(f"✅ PASS: Debt payoff time {debt_result['scenarios']['minimum_only']['total_months']} months")
    except Exception as e:
        print(f"❌ FAIL: {e}")

    print("\n📈 Test 4: Investment risk profile")
    try:
        allocation = FinancialCalculator.calculate_investment_allocation('aggressive', 25, 50000, 30)
        assert allocation['allocation_percentages']['stocks'] >= 70
        assert allocation['expected_annual_return'] > 0.08
        print(f"✅ PASS: Aggressive allocation - Stocks: {allocation['allocation_percentages']['stocks']}%")
    except Exception as e:
        print(f"❌ FAIL: {e}")

    print("\n🏖️ Test 5: Retirement planning")
    try:
        retirement = FinancialCalculator.calculate_retirement_needs(35, 65, 75000, 50000, 600)
        assert retirement['years_to_retirement'] == 30
        assert retirement['retirement_corpus_needed'] > 0
        print(f"✅ PASS: Retirement corpus needed: ${retirement['retirement_corpus_needed']:,.0f}")
    except Exception as e:
        print(f"❌ FAIL: {e}")

    print("\n🔄 Test 6: Financial flows")
    try:
        budget_data = FinancialFlows.budgeting_flow()
        assert budget_data is not None

        investment_data = FinancialFlows.investing_flow()
        assert investment_data is not None

        debt_data = FinancialFlows.debt_repayment_flow()
        assert debt_data is not None

        retirement_data = FinancialFlows.retirement_planning_flow()
        assert retirement_data is not None

        print("✅ PASS: All financial flows working correctly")
    except Exception as e:
        print(f"❌ FAIL: Flow test failed: {e}")

    print("\n🎉 Test suite completed!")

def main():
    """Main application function"""
    if TEST_MODE:
        run_tests()
        return

    st.markdown('<h1 class="main-header">🦙 AI Financial Advisor - LLAMA 3.3</h1>', unsafe_allow_html=True)

    st.info("💡 **Disclaimer**: AI suggestions are educational only and not financial advice. Always consult with a qualified financial professional for personalized guidance.")

    st.sidebar.subheader("📊 Financial Tools")
    menu = st.sidebar.selectbox(
        "Choose Section",
        ["Demo Dashboard", "Budgeting", "Investments", "Debt", "Retirement"]
    )

    # Initialize session state for navigation
    if "selected_page" not in st.session_state:
        st.session_state.selected_page = menu

    # Handle navigation changes with rerun
    if menu != st.session_state.selected_page:
        st.session_state.selected_page = menu
        st.rerun()

    # Route to appropriate section
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

    if any(key in st.session_state for key in ['budget_data', 'investment_data', 'debt_data', 'retirement_data']):
        st.markdown("---")
        st.subheader("📊 Your Financial Summary")

        summary_cols = st.columns(4)

        if 'budget_data' in st.session_state:
            with summary_cols[0]:
                budget = st.session_state.budget_data
                ai_insights = st.session_state.get('budget_ai_insights', {})
                ai_score = ai_insights.get('ai_score')
                ai_text = f" | AI: {ai_score}/100" if ai_score else ""

                st.markdown(f'''
                <div class="metric-card">
                    <h4>💰 Budget Health</h4>
                    <p><strong>{budget["financial_health"]}</strong></p>
                    <p>Savings Rate: {budget["savings_rate"]:.1f}%{ai_text}</p>
                </div>
                ''', unsafe_allow_html=True)

        if 'investment_data' in st.session_state:
            with summary_cols[1]:
                investment = st.session_state.investment_data
                ai_insights = st.session_state.get('investment_ai_insights', {})
                ai_score = ai_insights.get('ai_score')
                ai_text = f" | Risk: {ai_score}/100" if ai_score else ""

                st.markdown(f'''
                <div class="metric-card">
                    <h4>📈 Investment Profile</h4>
                    <p><strong>{investment["risk_level"]}</strong></p>
                    <p>Expected Return: {investment["expected_annual_return"]:.1%}{ai_text}</p>
                </div>
                ''', unsafe_allow_html=True)

        if 'debt_data' in st.session_state:
            with summary_cols[2]:
                debt = st.session_state.debt_data
                ai_insights = st.session_state.get('debt_ai_insights', {})
                ai_score = ai_insights.get('ai_score')
                ai_text = f" | Health: {ai_score}/100" if ai_score else ""

                st.markdown(f'''
                <div class="metric-card">
                    <h4>💳 Debt Status</h4>
                    <p><strong>${debt["total_debt"]:,.0f}</strong></p>
                    <p>Strategy: {debt["strategy"].title()}{ai_text}</p>
                </div>
                ''', unsafe_allow_html=True)

        if 'retirement_data' in st.session_state:
            with summary_cols[3]:
                retirement = st.session_state.retirement_data
                ai_insights = st.session_state.get('retirement_ai_insights', {})
                ai_score = ai_insights.get('ai_score')

                gap_status = "On Track" if retirement["retirement_gap"] <= 0 else f"${retirement['retirement_gap']:,.0f} gap"
                ai_text = f" | Readiness: {ai_score}/100" if ai_score else ""

                st.markdown(f'''
                <div class="metric-card">
                    <h4>🏖️ Retirement</h4>
                    <p><strong>{retirement["years_to_retirement"]} years left</strong></p>
                    <p>{gap_status}{ai_text}</p>
                </div>
                ''', unsafe_allow_html=True)

if __name__ == "__main__":
    main()
