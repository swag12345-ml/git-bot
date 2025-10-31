"""
AI Financial Advisor Application - LLAMA 3.3
A comprehensive financial planning tool with AI-powered insights

Required pip packages:
pip install streamlit plotly pandas numpy python-dotenv langchain-groq
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
        """Advanced Real-time Investment Portfolio Builder with live market data and Monte Carlo analysis."""
        if not TEST_MODE:
            st.markdown('<div class="flow-card"><h2>📈 Advanced Real-time Portfolio Builder</h2><p>AI-powered portfolio with live market data, Monte Carlo simulations, and dynamic allocations.</p></div>', unsafe_allow_html=True)

        # Initialize session state variables
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
        if 'saved_profiles' not in st.session_state:
            st.session_state.saved_profiles = []
        if 'ai_result' not in st.session_state:
            st.session_state.ai_result = None
        if 'market_data' not in st.session_state:
            st.session_state.market_data = {}
        if 'last_market_update' not in st.session_state:
            st.session_state.last_market_update = 0

        def fetch_market_data():
            """Fetch live market data for key assets."""
            try:
                tickers = ['^GSPC', 'IEF', 'GLD', 'BTC-USD', 'ETH-USD']
                data = yf.download(tickers, period='5d', progress=False)

                if data.empty:
                    return None

                market_info = {}
                for ticker in tickers:
                    try:
                        if len(tickers) > 1:
                            prices = data['Close'][ticker].dropna()
                        else:
                            prices = data['Close'].dropna()

                        if len(prices) >= 2:
                            current_price = prices.iloc[-1]
                            prev_price = prices.iloc[-2]
                            change = (current_price - prev_price) / prev_price * 100

                            # 5-day performance
                            if len(prices) >= 5:
                                week_change = (prices.iloc[-1] - prices.iloc[0]) / prices.iloc[0] * 100
                            else:
                                week_change = change

                            market_info[ticker] = {
                                'price': current_price,
                                'change': change,
                                'week_change': week_change,
                                'sentiment': 'Bullish' if week_change > 2 else 'Bearish' if week_change < -2 else 'Neutral'
                            }
                    except Exception:
                        # Fallback data if individual ticker fails
                        market_info[ticker] = {
                            'price': 0,
                            'change': 0,
                            'week_change': 0,
                            'sentiment': 'Neutral'
                        }

                return market_info

            except Exception:
                # Return fallback data if all fails
                return {
                    '^GSPC': {'price': 4500, 'change': 0, 'week_change': 0, 'sentiment': 'Neutral'},
                    'IEF': {'price': 100, 'change': 0, 'week_change': 0, 'sentiment': 'Neutral'},
                    'GLD': {'price': 180, 'change': 0, 'week_change': 0, 'sentiment': 'Neutral'},
                    'BTC-USD': {'price': 45000, 'change': 0, 'week_change': 0, 'sentiment': 'Neutral'},
                    'ETH-USD': {'price': 3000, 'change': 0, 'week_change': 0, 'sentiment': 'Neutral'}
                }

        def adjust_allocation_based_on_market(base_allocation, market_data):
            """Dynamically adjust allocation based on 5-day market performance."""
            adjusted = base_allocation.copy()

            try:
                # Get market performance
                sp500_perf = market_data.get('^GSPC', {}).get('week_change', 0)
                bond_perf = market_data.get('IEF', {}).get('week_change', 0)
                gold_perf = market_data.get('GLD', {}).get('week_change', 0)
                crypto_perf = (market_data.get('BTC-USD', {}).get('week_change', 0) +
                              market_data.get('ETH-USD', {}).get('week_change', 0)) / 2

                # Small adjustments based on momentum (max 5% shift)
                if sp500_perf > 3:  # Strong stock performance
                    adjustment = min(2, sp500_perf / 2)
                    adjusted['stocks'] = min(85, adjusted['stocks'] + adjustment)
                    adjusted['bonds'] = max(5, adjusted['bonds'] - adjustment/2)
                elif sp500_perf < -3:  # Weak stock performance
                    adjustment = min(2, abs(sp500_perf) / 2)
                    adjusted['stocks'] = max(15, adjusted['stocks'] - adjustment)
                    adjusted['bonds'] = min(70, adjusted['bonds'] + adjustment/2)

                if crypto_perf > 5 and adjusted['crypto'] > 0:  # Strong crypto momentum
                    adjustment = min(2, crypto_perf / 5)
                    adjusted['crypto'] = min(20, adjusted['crypto'] + adjustment)
                elif crypto_perf < -5 and adjusted['crypto'] > 0:  # Weak crypto
                    adjustment = min(adjusted['crypto'], abs(crypto_perf) / 5)
                    adjusted['crypto'] = max(0, adjusted['crypto'] - adjustment)
                    adjusted['cash'] += adjustment

                # Normalize to 100%
                total = sum(adjusted.values())
                if total != 100 and total > 0:
                    for key in adjusted:
                        adjusted[key] = round(adjusted[key] * 100 / total)

                return adjusted
            except Exception:
                return base_allocation

        def monte_carlo_simulation(allocations, capital, time_horizon):
            """Run 1000 Monte Carlo simulations for portfolio projections."""
            # Expected returns and volatilities
            expected_returns = {
                'stocks': 0.07, 'bonds': 0.04, 'cash': 0.02,
                'real_estate': 0.06, 'crypto': 0.15
            }
            volatilities = {
                'stocks': 0.15, 'bonds': 0.05, 'cash': 0.01,
                'real_estate': 0.08, 'crypto': 0.35
            }

            # Portfolio expected return and volatility
            portfolio_return = sum(allocations[asset]/100 * expected_returns[asset] for asset in allocations)
            portfolio_vol = np.sqrt(sum((allocations[asset]/100 * volatilities[asset])**2 for asset in allocations))

            # Run 1000 simulations
            np.random.seed(42)  # For reproducible results
            final_values = []

            for _ in range(1000):
                value = capital
                for year in range(time_horizon):
                    annual_return = np.random.normal(portfolio_return, portfolio_vol)
                    value *= (1 + annual_return)
                final_values.append(value)

            return {
                'mean': np.mean(final_values),
                'percentile_5': np.percentile(final_values, 5),
                'percentile_95': np.percentile(final_values, 95),
                'median': np.median(final_values)
            }

        # Update market data every 60 seconds
        current_time = time.time()
        if current_time - st.session_state.last_market_update > 60:
            if not TEST_MODE:
                with st.spinner("Fetching live market data..."):
                    st.session_state.market_data = fetch_market_data()
                    st.session_state.last_market_update = current_time
            else:
                # Test mode fallback
                st.session_state.market_data = {
                    '^GSPC': {'price': 4500, 'change': 1.2, 'week_change': 2.1, 'sentiment': 'Bullish'},
                    'IEF': {'price': 100, 'change': -0.3, 'week_change': 0.8, 'sentiment': 'Neutral'},
                    'GLD': {'price': 180, 'change': 0.5, 'week_change': -1.2, 'sentiment': 'Bearish'},
                    'BTC-USD': {'price': 45000, 'change': 3.2, 'week_change': 8.5, 'sentiment': 'Bullish'},
                    'ETH-USD': {'price': 3000, 'change': 2.8, 'week_change': 6.2, 'sentiment': 'Bullish'}
                }

        # Market Overview Dashboard
        if not TEST_MODE and st.session_state.profile_step == 4:
            st.markdown("### 📊 Live Market Overview")
            col1, col2, col3, col4, col5 = st.columns(5)

            market_data = st.session_state.market_data
            with col1:
                sp_data = market_data.get('^GSPC', {})
                st.markdown(display_metric_card(
                    "S&P 500",
                    f"{sp_data.get('change', 0):.1f}%",
                    f"{sp_data.get('sentiment', 'Neutral')}",
                    '#10b981' if sp_data.get('change', 0) > 0 else '#ef4444'
                ), unsafe_allow_html=True)

            with col2:
                bond_data = market_data.get('IEF', {})
                st.markdown(display_metric_card(
                    "Bonds (IEF)",
                    f"{bond_data.get('change', 0):.1f}%",
                    f"{bond_data.get('sentiment', 'Neutral')}",
                    '#10b981' if bond_data.get('change', 0) > 0 else '#ef4444'
                ), unsafe_allow_html=True)

            with col3:
                gold_data = market_data.get('GLD', {})
                st.markdown(display_metric_card(
                    "Gold (GLD)",
                    f"{gold_data.get('change', 0):.1f}%",
                    f"{gold_data.get('sentiment', 'Neutral')}",
                    '#10b981' if gold_data.get('change', 0) > 0 else '#ef4444'
                ), unsafe_allow_html=True)

            with col4:
                btc_data = market_data.get('BTC-USD', {})
                st.markdown(display_metric_card(
                    "Bitcoin",
                    f"{btc_data.get('change', 0):.1f}%",
                    f"{btc_data.get('sentiment', 'Neutral')}",
                    '#10b981' if btc_data.get('change', 0) > 0 else '#ef4444'
                ), unsafe_allow_html=True)

            with col5:
                eth_data = market_data.get('ETH-USD', {})
                st.markdown(display_metric_card(
                    "Ethereum",
                    f"{eth_data.get('change', 0):.1f}%",
                    f"{eth_data.get('sentiment', 'Neutral')}",
                    '#10b981' if eth_data.get('change', 0) > 0 else '#ef4444'
                ), unsafe_allow_html=True)

        # Reset functionality
        if not TEST_MODE:
            col_reset1, col_reset2 = st.columns([6, 1])
            with col_reset2:
                if st.button("🔄 Reset", key="reset_portfolio_wizard"):
                    st.session_state.profile_step = 1
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
                    st.session_state.ai_result = None
                    st.rerun()

        # Progress indicator
        if not TEST_MODE:
            progress_pct = (st.session_state.profile_step / 4) * 100
            st.progress(progress_pct / 100)
            st.markdown(f"<p style='text-align:center; color:#9ca3af;'>Step {st.session_state.profile_step} of 4</p>", unsafe_allow_html=True)
            st.markdown("---")

        # STEP 1: Investor Persona & Goal (unchanged from original)
        if st.session_state.profile_step == 1:
            if not TEST_MODE:
                st.subheader("Step 1: Investor Persona & Goal")

                st.session_state.profile_data['persona'] = st.selectbox(
                    "Select Your Investor Persona",
                    ["Conservative Saver", "Balanced Investor", "Growth Seeker", "Aggressive Trader"],
                    index=1 if st.session_state.profile_data['persona'] is None else
                          ["Conservative Saver", "Balanced Investor", "Growth Seeker", "Aggressive Trader"].index(st.session_state.profile_data['persona'])
                )

                st.session_state.profile_data['goal'] = st.selectbox(
                    "Primary Investment Goal",
                    ["Retirement", "House Down Payment", "Emergency Fund", "Wealth Building", "Education", "Other"],
                    index=0 if st.session_state.profile_data['goal'] is None else
                          ["Retirement", "House Down Payment", "Emergency Fund", "Wealth Building", "Education", "Other"].index(st.session_state.profile_data['goal'])
                )

                col1, col2 = st.columns(2)
                with col1:
                    st.session_state.profile_data['target_amount'] = st.number_input(
                        "Target Amount ($)",
                        min_value=0.0,
                        value=float(st.session_state.profile_data['target_amount']),
                        step=10000.0
                    )
                    st.session_state.profile_data['time_horizon'] = st.slider(
                        "Investment Time Horizon (years)",
                        1, 40,
                        st.session_state.profile_data['time_horizon']
                    )

                with col2:
                    st.session_state.profile_data['current_age'] = st.number_input(
                        "Your Current Age",
                        min_value=18,
                        max_value=80,
                        value=st.session_state.profile_data['current_age']
                    )
                    st.session_state.profile_data['investment_capital'] = st.number_input(
                        "Initial Investment Amount",
                        min_value=0.0,
                        value=float(st.session_state.profile_data['investment_capital']),
                        step=1000.0
                    )

                st.markdown("<div style='margin-top:20px;'></div>", unsafe_allow_html=True)
                if st.button("Next →", type="primary", key="step1_next"):
                    st.session_state.profile_step = 2
                    st.rerun()

        # STEP 2: Risk Profiling (unchanged from original)
        elif st.session_state.profile_step == 2:
            if not TEST_MODE:
                st.subheader("Step 2: Risk Profiling")

                st.markdown("Answer these questions to assess your risk tolerance:")
                st.markdown("<div style='margin-bottom:20px;'></div>", unsafe_allow_html=True)

                q1 = st.radio(
                    "If your portfolio dropped 20% in a month, you would:",
                    ["Panic and sell everything", "Feel uncomfortable but hold", "See it as a buying opportunity"],
                    index=0 if 'market_drop' not in st.session_state.profile_data['risk_answers'] else
                          ["Panic and sell everything", "Feel uncomfortable but hold", "See it as a buying opportunity"].index(st.session_state.profile_data['risk_answers'].get('market_drop', "Feel uncomfortable but hold")),
                    key="q1_market_drop"
                )
                st.session_state.profile_data['risk_answers']['market_drop'] = q1

                st.markdown("<div style='margin:15px 0;'></div>", unsafe_allow_html=True)

                q2 = st.radio(
                    "Your investment experience level:",
                    ["Beginner (< 2 years)", "Intermediate (2-10 years)", "Advanced (> 10 years)"],
                    index=0 if 'investment_experience' not in st.session_state.profile_data['risk_answers'] else
                          ["Beginner (< 2 years)", "Intermediate (2-10 years)", "Advanced (> 10 years)"].index(st.session_state.profile_data['risk_answers'].get('investment_experience', "Beginner (< 2 years)")),
                    key="q2_experience"
                )
                st.session_state.profile_data['risk_answers']['investment_experience'] = q2

                st.markdown("<div style='margin:15px 0;'></div>", unsafe_allow_html=True)

                q3 = st.radio(
                    "Your income stability:",
                    ["Unstable/Variable", "Stable", "Very Stable with Growth"],
                    index=0 if 'income_stability' not in st.session_state.profile_data['risk_answers'] else
                          ["Unstable/Variable", "Stable", "Very Stable with Growth"].index(st.session_state.profile_data['risk_answers'].get('income_stability', "Stable")),
                    key="q3_income"
                )
                st.session_state.profile_data['risk_answers']['income_stability'] = q3

                st.markdown("<div style='margin:15px 0;'></div>", unsafe_allow_html=True)

                q4 = st.radio(
                    "Regarding investment volatility:",
                    ["I need stable, predictable returns", "I can handle some ups and downs", "I'm comfortable with high volatility for higher returns"],
                    index=0 if 'sleep_factor' not in st.session_state.profile_data['risk_answers'] else
                          ["I need stable, predictable returns", "I can handle some ups and downs", "I'm comfortable with high volatility for higher returns"].index(st.session_state.profile_data['risk_answers'].get('sleep_factor', "I can handle some ups and downs")),
                    key="q4_volatility"
                )
                st.session_state.profile_data['risk_answers']['sleep_factor'] = q4

                # Calculate risk score
                risk_weights = {
                    "market_drop": {"Panic and sell everything": 1, "Feel uncomfortable but hold": 2, "See it as a buying opportunity": 3},
                    "investment_experience": {"Beginner (< 2 years)": 1, "Intermediate (2-10 years)": 2, "Advanced (> 10 years)": 3},
                    "income_stability": {"Unstable/Variable": 1, "Stable": 2, "Very Stable with Growth": 3},
                    "sleep_factor": {"I need stable, predictable returns": 1, "I can handle some ups and downs": 2, "I'm comfortable with high volatility for higher returns": 3}
                }

                risk_score = sum(risk_weights[q][a] for q, a in st.session_state.profile_data['risk_answers'].items())
                st.session_state.profile_data['risk_score'] = risk_score

                if risk_score <= 6:
                    risk_profile = "Conservative"
                    risk_color = "#f59e0b"
                elif risk_score <= 9:
                    risk_profile = "Moderate"
                    risk_color = "#3b82f6"
                else:
                    risk_profile = "Aggressive"
                    risk_color = "#ef4444"

                st.session_state.profile_data['risk_profile'] = risk_profile

                # Live risk gauge with proper spacing
                st.markdown("<div style='margin-top:40px;'></div>", unsafe_allow_html=True)
                st.markdown("### Your Risk Profile")

                col_gauge1, col_gauge2, col_gauge3 = st.columns([1, 2, 1])
                with col_gauge2:
                    fig_risk = go.Figure(go.Indicator(
                        mode="gauge+number",
                        value=risk_score,
                        title={'text': f"<b>Risk Score: {risk_profile}</b>", 'font': {'size': 20, 'color': 'white'}},
                        number={'font': {'size': 40, 'color': 'white'}},
                        gauge={
                            'axis': {'range': [None, 12], 'tickwidth': 1, 'tickcolor': 'white'},
                            'bar': {'color': risk_color, 'thickness': 0.6},
                            'steps': [
                                {'range': [0, 6], 'color': '#d1d5db'},
                                {'range': [6, 9], 'color': '#9ca3af'},
                                {'range': [9, 12], 'color': '#6b7280'}
                            ],
                            'threshold': {
                                'line': {'color': 'white', 'width': 2},
                                'thickness': 0.75,
                                'value': risk_score
                            }
                        }
                    ))
                    fig_risk.update_layout(
                        height=350,
                        paper_bgcolor='#1f2937',
                        plot_bgcolor='#1f2937',
                        font_color='white',
                        margin=dict(t=60, b=40, l=40, r=40)
                    )
                    st.plotly_chart(fig_risk, use_container_width=True, config={"displayModeBar": False})

                st.markdown("<div style='margin-top:30px;'></div>", unsafe_allow_html=True)
                col_nav1, col_nav2 = st.columns(2)
                with col_nav1:
                    if st.button("← Back", key="step2_back"):
                        st.session_state.profile_step = 1
                        st.rerun()
                with col_nav2:
                    if st.button("Next →", type="primary", key="step2_next"):
                        st.session_state.profile_step = 3
                        st.rerun()

        # STEP 3: Enhanced Asset Allocation with AI adjustments
        elif st.session_state.profile_step == 3:
            if not TEST_MODE:
                st.subheader("Step 3: Smart Asset Allocation")

                st.info(f"AI Profile: **{st.session_state.profile_data['risk_profile']}** — Adjusting for current market conditions")

                # Set default allocations based on risk profile
                if st.session_state.profile_data['risk_profile'] == "Conservative":
                    default_alloc = {'stocks': 25, 'bonds': 65, 'cash': 10, 'real_estate': 0, 'crypto': 0}
                elif st.session_state.profile_data['risk_profile'] == "Moderate":
                    default_alloc = {'stocks': 60, 'bonds': 30, 'cash': 5, 'real_estate': 5, 'crypto': 0}
                else:
                    default_alloc = {'stocks': 75, 'bonds': 15, 'cash': 5, 'real_estate': 5, 'crypto': 0}

                # Apply market-based adjustments
                if st.session_state.market_data:
                    market_adjusted = adjust_allocation_based_on_market(default_alloc, st.session_state.market_data)

                    # Show AI adjustment if different from default
                    if market_adjusted != default_alloc:
                        st.markdown("### 🤖 AI Market Adjustments")
                        col1, col2 = st.columns(2)

                        with col1:
                            st.markdown("**Original Allocation:**")
                            for asset, pct in default_alloc.items():
                                if pct > 0:
                                    st.write(f"{asset.title()}: {pct}%")

                        with col2:
                            st.markdown("**Market-Adjusted Allocation:**")
                            for asset, pct in market_adjusted.items():
                                if pct > 0:
                                    change = pct - default_alloc[asset]
                                    color = "🔴" if change < 0 else "🟢" if change > 0 else ""
                                    st.write(f"{asset.title()}: {pct}% {color}")

                        st.info("💡 AI adjusted your allocation based on recent 5-day market performance")
                        default_alloc = market_adjusted

                # Only update if allocations haven't been customized
                if sum(st.session_state.profile_data['allocations'].values()) == 0 or st.session_state.profile_data['allocations'] == {'stocks': 60, 'bonds': 30, 'cash': 10, 'real_estate': 0, 'crypto': 0}:
                    st.session_state.profile_data['allocations'] = default_alloc.copy()

                st.markdown("### Fine-tune Your Asset Mix")

                stocks_pct = st.slider(
                    "Stocks (%)",
                    0, 100,
                    st.session_state.profile_data['allocations']['stocks'],
                    key="slider_stocks"
                )

                bonds_pct = st.slider(
                    "Bonds (%)",
                    0, 100,
                    st.session_state.profile_data['allocations']['bonds'],
                    key="slider_bonds"
                )

                cash_pct = st.slider(
                    "Cash (%)",
                    0, 100,
                    st.session_state.profile_data['allocations']['cash'],
                    key="slider_cash"
                )

                real_estate_pct = st.slider(
                    "Real Estate / REITs (%)",
                    0, 100,
                    st.session_state.profile_data['allocations']['real_estate'],
                    key="slider_reits"
                )

                crypto_pct = st.slider(
                    "Crypto (%)",
                    0, 100,
                    st.session_state.profile_data['allocations']['crypto'],
                    key="slider_crypto"
                )

                total_alloc = stocks_pct + bonds_pct + cash_pct + real_estate_pct + crypto_pct

                # Crypto warning
                if crypto_pct > 10:
                    st.warning("⚠️ **High Crypto Allocation**: Cryptocurrency is highly volatile. Consider limiting to 5-10% of your portfolio.")

                st.markdown("<div style='margin-top:25px;'></div>", unsafe_allow_html=True)

                if total_alloc != 100:
                    st.warning(f"⚠️ Total allocation: {total_alloc}%. Please adjust to equal 100%.")
                else:
                    st.success(f"✓ Total allocation: {total_alloc}%")
                    st.session_state.profile_data['allocations'] = {
                        'stocks': stocks_pct,
                        'bonds': bonds_pct,
                        'cash': cash_pct,
                        'real_estate': real_estate_pct,
                        'crypto': crypto_pct
                    }

                # Live pie chart with proper spacing
                st.markdown("<div style='margin-top:30px;'></div>", unsafe_allow_html=True)
                if total_alloc > 0:
                    alloc_values = [stocks_pct, bonds_pct, cash_pct, real_estate_pct, crypto_pct]
                    alloc_labels = ['Stocks', 'Bonds', 'Cash', 'Real Estate', 'Crypto']
                    alloc_colors = ['#3b82f6', '#f59e0b', '#10b981', '#8b5cf6', '#ef4444']

                    col_chart1, col_chart2, col_chart3 = st.columns([0.5, 2, 0.5])
                    with col_chart2:
                        fig_pie = go.Figure(data=[go.Pie(
                            labels=[l for l, v in zip(alloc_labels, alloc_values) if v > 0],
                            values=[v for v in alloc_values if v > 0],
                            marker=dict(colors=[c for c, v in zip(alloc_colors, alloc_values) if v > 0]),
                            hole=0.4,
                            textposition='inside',
                            textinfo='percent+label'
                        )])
                        fig_pie.update_layout(
                            title={
                                'text': "Your Smart Portfolio Mix",
                                'x': 0.5,
                                'xanchor': 'center',
                                'font': {'size': 18, 'color': 'white'}
                            },
                            height=450,
                            paper_bgcolor='#1f2937',
                            plot_bgcolor='#1f2937',
                            font_color='white',
                            showlegend=True,
                            legend=dict(
                                orientation="h",
                                yanchor="bottom",
                                y=-0.15,
                                xanchor="center",
                                x=0.5
                            ),
                            margin=dict(t=80, b=80, l=60, r=60)
                        )
                        st.plotly_chart(fig_pie, use_container_width=True, config={"displayModeBar": False})

                st.markdown("<div style='margin-top:20px;'></div>", unsafe_allow_html=True)

                # Auto-Rebalance button
                if st.button("🔁 Reset to AI Suggestion", key="auto_rebalance"):
                    st.session_state.profile_data['allocations'] = default_alloc.copy()
                    st.success(f"Reset to AI-optimized {st.session_state.profile_data['risk_profile']} profile!")
                    st.rerun()

                st.markdown("<div style='margin-top:25px;'></div>", unsafe_allow_html=True)
                col_nav1, col_nav2 = st.columns(2)
                with col_nav1:
                    if st.button("← Back", key="step3_back"):
                        st.session_state.profile_step = 2
                        st.rerun()
                with col_nav2:
                    if st.button("Next →", type="primary", key="step3_next", disabled=(total_alloc != 100)):
                        st.session_state.profile_step = 4
                        st.rerun()

        # STEP 4: Enhanced Analysis with Monte Carlo and Live Data
        elif st.session_state.profile_step == 4:
            if not TEST_MODE:
                st.subheader("Step 4: Advanced Portfolio Analysis")

                # Create empty containers for live updates
                market_container = st.empty()
                metrics_container = st.empty()
                monte_carlo_container = st.empty()
                chart_container = st.empty()
                ai_container = st.empty()

                # Display profile summary
                st.markdown("### Your Portfolio Profile")
                col1, col2, col3, col4 = st.columns(4)

                with col1:
                    st.markdown(display_metric_card(
                        "Persona",
                        st.session_state.profile_data['persona'],
                        f"Risk: {st.session_state.profile_data['risk_profile']}"
                    ), unsafe_allow_html=True)

                with col2:
                    st.markdown(display_metric_card(
                        "Goal",
                        st.session_state.profile_data['goal'],
                        f"Horizon: {st.session_state.profile_data['time_horizon']} years"
                    ), unsafe_allow_html=True)

                with col3:
                    st.markdown(display_metric_card(
                        "Initial Capital",
                        f"${st.session_state.profile_data['investment_capital']:,.0f}"
                    ), unsafe_allow_html=True)

                with col4:
                    if st.session_state.profile_data['target_amount'] > 0:
                        st.markdown(display_metric_card(
                            "Target Amount",
                            f"${st.session_state.profile_data['target_amount']:,.0f}"
                        ), unsafe_allow_html=True)
                    else:
                        st.markdown(display_metric_card(
                            "Age",
                            f"{st.session_state.profile_data['current_age']} years"
                        ), unsafe_allow_html=True)

                # Calculate enhanced portfolio metrics
                capital = st.session_state.profile_data['investment_capital']
                allocations = st.session_state.profile_data['allocations']
                time_horizon = st.session_state.profile_data['time_horizon']

                # Run Monte Carlo simulation
                mc_results = monte_carlo_simulation(allocations, capital, time_horizon)

                # Enhanced expected return calculation
                enhanced_returns = {
                    'stocks': 0.07, 'bonds': 0.04, 'cash': 0.02,
                    'real_estate': 0.06, 'crypto': 0.15
                }
                portfolio_return = sum(allocations[asset]/100 * enhanced_returns[asset] for asset in allocations)

                # Portfolio volatility
                volatilities = {
                    'stocks': 0.15, 'bonds': 0.05, 'cash': 0.01,
                    'real_estate': 0.08, 'crypto': 0.35
                }
                portfolio_vol = np.sqrt(sum((allocations[asset]/100 * volatilities[asset])**2 for asset in allocations))

                # Display enhanced metrics
                with metrics_container.container():
                    st.markdown("### 📊 Enhanced Portfolio Metrics")
                    col1, col2, col3, col4 = st.columns(4)

                    with col1:
                        st.markdown(display_metric_card(
                            "Expected Return",
                            f"{portfolio_return:.2%}",
                            "AI-enhanced calculation"
                        ), unsafe_allow_html=True)

                    with col2:
                        st.markdown(display_metric_card(
                            "Volatility",
                            f"{portfolio_vol:.2%}",
                            "Risk measure"
                        ), unsafe_allow_html=True)

                    with col3:
                        # Enhanced diversification index
                        div_index = sum(1 for v in allocations.values() if v >= 5) * 20
                        div_index = min(100, div_index)
                        st.markdown(display_metric_card(
                            "Diversification",
                            f"{div_index}/100",
                            f"{sum(1 for v in allocations.values() if v >= 5)} assets"
                        ), unsafe_allow_html=True)

                    with col4:
                        # Sharpe ratio approximation
                        risk_free_rate = 0.02
                        sharpe_ratio = (portfolio_return - risk_free_rate) / portfolio_vol if portfolio_vol > 0 else 0
                        st.markdown(display_metric_card(
                            "Sharpe Ratio",
                            f"{sharpe_ratio:.2f}",
                            "Risk-adjusted return"
                        ), unsafe_allow_html=True)

                # Monte Carlo Results
                with monte_carlo_container.container():
                    st.markdown("### 🎯 Monte Carlo Simulation (1000 iterations)")
                    col_mc1, col_mc2, col_mc3 = st.columns(3)

                    with col_mc1:
                        st.markdown(display_metric_card(
                            "Pessimistic (5th %ile)",
                            f"${mc_results['percentile_5']:,.0f}",
                            "Conservative outcome",
                            '#ef4444'
                        ), unsafe_allow_html=True)

                    with col_mc2:
                        st.markdown(display_metric_card(
                            "Expected (Mean)",
                            f"${mc_results['mean']:,.0f}",
                            "Average outcome",
                            '#3b82f6'
                        ), unsafe_allow_html=True)

                    with col_mc3:
                        st.markdown(display_metric_card(
                            "Optimistic (95th %ile)",
                            f"${mc_results['percentile_95']:,.0f}",
                            "Best case scenario",
                            '#10b981'
                        ), unsafe_allow_html=True)

                # Enhanced Growth Chart with Monte Carlo Distribution
                with chart_container.container():
                    st.markdown("### 📈 Advanced Growth Projections")

                    # Create projection years
                    years_list = list(range(0, time_horizon + 1, max(1, time_horizon // 10)))
                    if time_horizon not in years_list:
                        years_list.append(time_horizon)
                    years_list = sorted(years_list)

                    # Calculate different scenarios
                    conservative_vals = [capital * ((1 + portfolio_return * 0.7) ** y) for y in years_list]
                    expected_vals = [capital * ((1 + portfolio_return) ** y) for y in years_list]
                    optimistic_vals = [capital * ((1 + portfolio_return * 1.3) ** y) for y in years_list]

                    # Create the chart
                    fig_growth = go.Figure()
                    fig_growth.add_trace(go.Scatter(x=years_list, y=conservative_vals, name='Conservative', line=dict(color='#f59e0b', dash='dash')))
                    fig_growth.add_trace(go.Scatter(x=years_list, y=expected_vals, name='Expected', line=dict(color='#3b82f6', width=3)))
                    fig_growth.add_trace(go.Scatter(x=years_list, y=optimistic_vals, name='Optimistic', line=dict(color='#10b981', dash='dash')))

                    # Add Monte Carlo confidence interval
                    fig_growth.add_trace(go.Scatter(
                        x=[time_horizon, time_horizon],
                        y=[mc_results['percentile_5'], mc_results['percentile_95']],
                        mode='lines',
                        line=dict(color='rgba(255, 255, 255, 0.3)', width=10),
                        name='Monte Carlo Range',
                        showlegend=False
                    ))

                    # Add Monte Carlo points
                    fig_growth.add_trace(go.Scatter(
                        x=[time_horizon],
                        y=[mc_results['percentile_5']],
                        mode='markers',
                        marker=dict(color='#ef4444', size=12, symbol='triangle-down'),
                        name='MC 5th percentile'
                    ))
                    fig_growth.add_trace(go.Scatter(
                        x=[time_horizon],
                        y=[mc_results['mean']],
                        mode='markers',
                        marker=dict(color='#3b82f6', size=12, symbol='circle'),
                        name='MC Mean'
                    ))
                    fig_growth.add_trace(go.Scatter(
                        x=[time_horizon],
                        y=[mc_results['percentile_95']],
                        mode='markers',
                        marker=dict(color='#10b981', size=12, symbol='triangle-up'),
                        name='MC 95th percentile'
                    ))

                    fig_growth.update_layout(
                        title="Advanced Portfolio Growth Projections with Monte Carlo Analysis",
                        xaxis_title="Years",
                        yaxis_title="Portfolio Value ($)",
                        height=500,
                        paper_bgcolor='#1f2937',
                        plot_bgcolor='#1f2937',
                        font_color='white',
                        hovermode='x unified',
                        showlegend=True
                    )
                    st.plotly_chart(fig_growth, use_container_width=True, config={"displayModeBar": False})

                # AI Analysis section
                with ai_container.container():
                    st.markdown("<div style='margin-top:30px;'></div>", unsafe_allow_html=True)
                    if st.button("🤖 Generate AI Insights", type="primary", key="analyze_ai", use_container_width=False):
                        with st.spinner("Analyzing your portfolio with AI..."):
                            # Create comprehensive data for AI analysis
                            analysis_data = {
                                'profile_data': st.session_state.profile_data,
                                'portfolio_return': portfolio_return,
                                'volatility': portfolio_vol,
                                'sharpe_ratio': sharpe_ratio,
                                'diversification': div_index,
                                'monte_carlo': mc_results,
                                'market_data': st.session_state.market_data,
                                'allocations': allocations,
                                'capital': capital
                            }

                            st.session_state.ai_result = generate_ai_insights(
                                analysis_data,
                                "Advanced Investment Analysis"
                            )
                        st.rerun()

                    # Display AI result if available
                    if st.session_state.ai_result:
                        st.markdown("<div style='margin-top:20px;'></div>", unsafe_allow_html=True)
                        display_ai_suggestions(st.session_state.ai_result, "Advanced Investment Analysis")

                # Auto-refresh functionality (live updates every 60 seconds)
                if not TEST_MODE:
                    # Use a placeholder for auto-refresh status
                    refresh_status = st.empty()
                    next_update = 60 - int((time.time() - st.session_state.last_market_update) % 60)
                    refresh_status.info(f"🔄 Live data updates every 60 seconds. Next update in {next_update}s")

                    # Auto-refresh trigger
                    if st.button("🔄 Refresh Market Data", key="manual_refresh"):
                        st.session_state.last_market_update = 0  # Force refresh
                        st.rerun()

                # Backend save
                from datetime import datetime
                profile_snapshot = {
                    'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    'persona': st.session_state.profile_data['persona'],
                    'goal': st.session_state.profile_data['goal'],
                    'risk_profile': st.session_state.profile_data['risk_profile'],
                    'risk_score': st.session_state.profile_data['risk_score'],
                    'allocations': allocations.copy(),
                    'capital': capital,
                    'expected_return': f"{portfolio_return:.2%}",
                    'volatility': f"{portfolio_vol:.2%}",
                    'sharpe_ratio': f"{sharpe_ratio:.2f}",
                    'monte_carlo_mean': f"${mc_results['mean']:,.0f}",
                    'monte_carlo_5th': f"${mc_results['percentile_5']:,.0f}",
                    'monte_carlo_95th': f"${mc_results['percentile_95']:,.0f}"
                }

                # Auto-save to session state
                if 'saved_profiles' not in st.session_state or not st.session_state.saved_profiles:
                    st.session_state.saved_profiles = [profile_snapshot]
                else:
                    if len(st.session_state.saved_profiles) > 0:
                        last_profile = st.session_state.saved_profiles[-1]
                        if (last_profile.get('risk_profile') != profile_snapshot['risk_profile'] or
                            last_profile.get('allocations') != profile_snapshot['allocations']):
                            st.session_state.saved_profiles.append(profile_snapshot)
                    else:
                        st.session_state.saved_profiles.append(profile_snapshot)

                # Navigation
                st.markdown("<div style='margin-top:40px;'></div>", unsafe_allow_html=True)
                col_nav1, col_nav2 = st.columns(2)
                with col_nav1:
                    if st.button("← Back", key="step4_back"):
                        st.session_state.profile_step = 3
                        st.rerun()
                with col_nav2:
                    if st.button("🏁 Complete Analysis", type="primary", key="finish"):
                        # Save final results
                        allocation_data = {
                            'allocation_percentages': allocations,
                            'expected_annual_return': portfolio_return,
                            'volatility_estimate': portfolio_vol,
                            'monte_carlo_results': mc_results
                        }
                        st.session_state.investment_data = allocation_data
                        st.session_state.investment_ai_insights = st.session_state.ai_result
                        st.success("🎉 Advanced portfolio analysis completed and saved!")
                        st.balloons()

        # Test mode return
        if TEST_MODE:
            test_allocation = FinancialCalculator.calculate_investment_allocation('moderate', 15, 25000.0, 35)
            # Add Monte Carlo results for test
            test_allocation['monte_carlo_results'] = {
                'mean': 45000,
                'percentile_5': 32000,
                'percentile_95': 62000,
                'median': 44500
            }
            return test_allocation

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
