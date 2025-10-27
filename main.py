"""
AI Financial Advisor Application - LLAMA 3.3
A comprehensive financial planning tool with AI-powered insights

Required pip packages:
pip install streamlit plotly pandas numpy python-dotenv langchain langchain-community langchain-groq langchain-huggingface langchain-text-splitters
"""

import streamlit as st
import os
import json
import sys
from dotenv import load_dotenv
from langchain_groq import ChatGroq

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from typing import Dict, List, Any

# Test mode check
TEST_MODE = "--test" in sys.argv

if not TEST_MODE:
    st.set_page_config(
        page_title="AI Financial Advisor - LLAMA 3.3",
        page_icon="💰",
        layout="wide",
        initial_sidebar_state="expanded"
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
            margin-bottom: 2rem;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }

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

        .metric-card {
            background: linear-gradient(135deg, #1f2937 0%, #374151 100%);
            padding: 1.5rem;
            border-radius: 12px;
            border-left: 5px solid #3b82f6;
            margin: 1rem 0;
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

        .streamlit-expanderHeader {
            background-color: #374151 !important;
            color: #ffffff !important;
            border: 1px solid #6b7280 !important;
        }

        .streamlit-expanderContent {
            background-color: #1f2937 !important;
            border: 1px solid #4b5563 !important;
        }

        [data-testid="stMetric"] {
            background-color: #1f2937;
            padding: 1rem;
            border-radius: 8px;
            border: 1px solid #4b5563;
            margin-bottom: 1.5rem;
        }

        [data-testid="stMetricValue"] {
            font-size: 1.5rem;
            font-weight: bold;
            color: #ffffff;
            margin-bottom: 0.5rem;
        }

        [data-testid="stMetricLabel"] {
            font-size: 0.875rem;
            color: #9ca3af;
            margin-bottom: 0.5rem;
        }

        [data-testid="stMetricDelta"] {
            font-size: 0.75rem;
            margin-top: 0.5rem;
        }

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

        .js-plotly-plot {
            background-color: #1f2937 !important;
        }
    </style>
    """, unsafe_allow_html=True)

load_dotenv()
working_dir = os.path.dirname(os.path.abspath(__file__))

def load_groq_api_key():
    try:
        with open(os.path.join(working_dir, "config.json"), "r") as f:
            return json.load(f).get("GROQ_API_KEY")
    except FileNotFoundError:
        return os.getenv("GROQ_API_KEY")

groq_api_key = load_groq_api_key() if not TEST_MODE else "test_key"

def generate_ai_insights(data: Dict[str, Any], context_label: str) -> Dict[str, Any]:
    """Centralized AI insights generator using LLaMA 3.3 via Groq."""
    fallback_response = {
        "ai_score": None,
        "ai_reasoning": "AI analysis not available - using deterministic fallback.",
        "ai_recommendations": [
            "Review your financial data and look for improvement opportunities",
            "Consider consulting with a financial professional for personalized advice"
        ]
    }

    if not groq_api_key or TEST_MODE:
        return fallback_response

    try:
        llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0.3, groq_api_key=groq_api_key)

        if context_label == "Budget Analysis":
            prompt = f"""
            You are an expert financial advisor. Analyze this budget data and provide insights.

            Budget Data (JSON): {json.dumps(data, default=str)}

            Tasks:
            1. Provide a Financial Health Score (0-100)
            2. Give a brief 2-3 sentence reasoning for the score
            3. Provide 3-5 concise, actionable recommendations

            Output ONLY valid JSON with keys: ai_score, ai_reasoning, ai_recommendations

            Example: {{"ai_score": 75, "ai_reasoning": "Good savings rate but high housing costs.", "ai_recommendations": ["Reduce housing costs", "Increase emergency fund"]}}
            """
        elif context_label == "Investment Analysis":
            prompt = f"""
            You are an expert investment advisor. Analyze this portfolio data.

            Investment Data (JSON): {json.dumps(data, default=str)}

            Tasks:
            1. Provide an Investment Risk Score (0-100)
            2. Give a brief 2-3 sentence explanation
            3. Provide 3-5 portfolio improvement suggestions

            Output ONLY valid JSON with keys: ai_score, ai_reasoning, ai_recommendations
            """
        elif context_label == "Debt Analysis":
            prompt = f"""
            You are an expert debt counselor. Analyze this debt situation.

            Debt Data (JSON): {json.dumps(data, default=str)}

            Tasks:
            1. Provide a Debt Health Score (0-100)
            2. Give a brief 2-3 sentence assessment
            3. Provide 3-5 actionable steps

            Output ONLY valid JSON with keys: ai_score, ai_reasoning, ai_recommendations
            """
        elif context_label == "Retirement Analysis":
            prompt = f"""
            You are an expert retirement planner. Analyze this data.

            Retirement Data (JSON): {json.dumps(data, default=str)}

            Tasks:
            1. Provide a Retirement Readiness Index (0-100)
            2. Give a brief 2-3 sentence assessment
            3. Provide 3-5 specific actions

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

def display_ai_suggestions(ai_insights: Dict[str, Any], context_label: str):
    """Display AI suggestions in a consistent format."""
    if TEST_MODE:
        return

    ai_score = ai_insights.get("ai_score")
    ai_reasoning = ai_insights.get("ai_reasoning", "")
    ai_recommendations = ai_insights.get("ai_recommendations", [])

    st.markdown("### 🤖 AI Suggestions")

    if ai_score is not None:
        st.markdown(f'''
        <div class="ai-suggestions-card">
            <h4>AI {context_label.split()[0]} Score: {ai_score}/100</h4>
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
    """Core financial calculation functions"""

    @staticmethod
    def calculate_budget_summary(income: float, expenses: Dict[str, float]) -> Dict[str, Any]:
        """Calculate comprehensive budget summary."""
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
        savings_rate = (savings / income * 100) if income > 0 else 0

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

        recommendations = []
        if savings_rate < 10:
            recommendations.append("🎯 Aim to save at least 10% of your income")
        if housing_ratio > 30:
            recommendations.append(f"🏠 Consider reducing housing costs - currently {housing_ratio:.1f}% of income")
        if expenses.get('dining_out', 0) > expenses.get('groceries', 0):
            recommendations.append("🍽️ Consider cooking more at home to reduce dining expenses")
        if savings_rate >= 20:
            recommendations.append("🌟 Excellent savings rate! Consider investing surplus funds")
        if expenses.get('debt_payments', 0) / income > 0.2:
            recommendations.append("💳 Focus on debt repayment - debt payments are high")

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
    def calculate_investment_allocation(risk_profile: str, time_horizon: int, capital: float, age: int = 35) -> Dict[str, Any]:
        """Calculate investment allocation."""
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
        allocation['bonds'] = max(5, 95 - allocation['stocks'] - allocation['cash'])

        dollar_allocation = {
            asset: (percentage / 100) * capital
            for asset, percentage in allocation.items()
        }

        expected_returns = {'stocks': 0.10, 'bonds': 0.04, 'cash': 0.02}

        portfolio_return = sum(
            (allocation[asset] / 100) * expected_returns[asset]
            for asset in allocation
        )

        projections = {}
        for years in [5, 10, 20, 30]:
            if years <= time_horizon:
                conservative = capital * ((1 + portfolio_return * 0.7) ** years)
                expected = capital * ((1 + portfolio_return) ** years)
                optimistic = capital * ((1 + portfolio_return * 1.3) ** years)

                projections[f'{years}_years'] = {
                    'conservative': conservative,
                    'expected': expected,
                    'optimistic': optimistic
                }

        volatilities = {'stocks': 0.16, 'bonds': 0.05, 'cash': 0.01}
        volatility = sum((allocation[asset] / 100) * volatilities[asset] for asset in allocation)

        return {
            'allocation_percentages': allocation,
            'allocation_dollars': dollar_allocation,
            'expected_annual_return': portfolio_return,
            'projections': projections,
            'risk_level': risk_profile,
            'volatility_estimate': volatility
        }

    @staticmethod
    def calculate_debt_payoff(debts: List[Dict], extra_payment: float = 0, strategy: str = 'avalanche') -> Dict[str, Any]:
        """Calculate debt payoff strategy."""
        if not debts:
            return {'total_debt': 0, 'payoff_plan': [], 'total_interest': 0, 'scenarios': {}, 'strategy': strategy}

        valid_debts = []
        for debt in debts:
            try:
                balance = float(debt.get('balance', 0))
                interest_rate = float(debt.get('interest_rate', 0))
                minimum_payment = float(debt.get('minimum_payment', 0))

                if balance <= 0:
                    continue

                monthly_interest = balance * (interest_rate / 100 / 12) if interest_rate > 0 else 0
                if minimum_payment <= monthly_interest and interest_rate > 0:
                    minimum_payment = monthly_interest * 1.2 + 25

                if minimum_payment <= 0:
                    minimum_payment = max(25, balance * 0.02)

                valid_debts.append({
                    'name': debt.get('name', 'Unknown Debt'),
                    'balance': balance,
                    'interest_rate': interest_rate,
                    'minimum_payment': minimum_payment
                })
            except (ValueError, TypeError):
                continue

        if not valid_debts:
            return {'total_debt': 0, 'payoff_plan': [], 'total_interest': 0, 'scenarios': {}, 'strategy': strategy}

        total_debt = sum(debt['balance'] for debt in valid_debts)
        total_minimum = sum(debt['minimum_payment'] for debt in valid_debts)

        scenarios = {}
        for scenario_name, extra in [('minimum_only', 0), ('with_extra', extra_payment)]:
            payoff_plan = []
            remaining_debts = [debt.copy() for debt in valid_debts]

            if strategy == 'avalanche':
                remaining_debts.sort(key=lambda x: x['interest_rate'], reverse=True)
            else:
                remaining_debts.sort(key=lambda x: x['balance'])

            total_interest = 0
            total_months = 0
            available_extra = extra

            for i, debt in enumerate(remaining_debts):
                monthly_payment = debt['minimum_payment']
                if scenario_name == 'with_extra' and i == 0 and available_extra > 0:
                    monthly_payment += available_extra

                balance = debt['balance']
                rate = debt['interest_rate'] / 100 / 12 if debt['interest_rate'] > 0 else 0

                if rate <= 0:
                    months = int(np.ceil(balance / monthly_payment)) if monthly_payment > 0 else 999
                elif monthly_payment <= balance * rate:
                    months = 999
                else:
                    months = -np.log(1 - (balance * rate) / monthly_payment) / np.log(1 + rate)
                    months = max(1, int(np.ceil(months)))

                if rate > 0:
                    total_payment = monthly_payment * months
                    interest_paid = max(0, total_payment - balance)
                else:
                    interest_paid = 0

                total_interest += interest_paid
                total_months = max(total_months, months)

                payoff_plan.append({
                    'debt_name': debt['name'],
                    'balance': balance,
                    'interest_rate': debt['interest_rate'],
                    'monthly_payment': monthly_payment,
                    'months_to_payoff': months,
                    'interest_paid': interest_paid,
                    'priority': i + 1
                })

                if i == 0 and scenario_name == 'with_extra':
                    available_extra = monthly_payment - debt['minimum_payment']

            scenarios[scenario_name] = {
                'payoff_plan': payoff_plan,
                'total_interest': total_interest,
                'total_months': total_months,
                'total_payments': total_minimum + (extra if scenario_name == 'with_extra' else 0)
            }

        interest_savings = max(0, scenarios['minimum_only']['total_interest'] - scenarios['with_extra']['total_interest'])
        time_savings = max(0, scenarios['minimum_only']['total_months'] - scenarios['with_extra']['total_months'])

        return {
            'total_debt': total_debt,
            'total_minimum_payment': total_minimum,
            'scenarios': scenarios,
            'strategy': strategy,
            'interest_savings': interest_savings,
            'time_savings_months': time_savings,
            'recommended_extra_payment': max(50, total_debt * 0.02)
        }

    @staticmethod
    def calculate_retirement_needs(current_age: int, retirement_age: int, current_income: float,
                                 current_savings: float, monthly_contribution: float) -> Dict[str, Any]:
        """Calculate retirement planning."""
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

        years_to_retirement = max(1, retirement_age - current_age)
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

            replacement_ratio_achieved = (monthly_retirement_income * 12) / future_income_needed if future_income_needed > 0 else 0

            scenarios[scenario_name] = {
                'monthly_contribution': scenario_monthly,
                'projected_total': scenario_total,
                'monthly_retirement_income': monthly_retirement_income,
                'replacement_ratio_achieved': replacement_ratio_achieved
            }

        recommendations = []
        if retirement_gap > 0:
            increase_needed = max(0, required_monthly_contribution - monthly_contribution)
            if increase_needed > 0:
                recommendations.append(f"💰 Increase monthly contributions by ${increase_needed:.0f}")
            else:
                recommendations.append("🎉 You are already on track, no increase needed")
        else:
            recommendations.append("🎉 You are already on track, no increase needed")

        if years_to_retirement > 30:
            recommendations.append("📈 Consider more aggressive investments for long-term growth")
        elif years_to_retirement < 10:
            recommendations.append("🛡️ Consider shifting to more conservative investments")

        if monthly_contribution < 500:
            recommendations.append("🎯 Aim to contribute at least $500/month for retirement")

        recommendations.append("🏢 Maximize employer 401(k) matching if available")
        recommendations.append("💡 Consider Roth IRA for tax-free retirement income")

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
            'recommendations': recommendations
        }

class FinancialVisualizer:
    """Advanced visualization functions"""

    @staticmethod
    def plot_budget_summary(budget_data: Dict[str, Any]) -> go.Figure:
        """Create comprehensive budget visualization."""
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Income vs Expenses', 'Savings Rate', 'Expense Categories', 'Financial Health Score'),
            specs=[[{"type": "bar"}, {"type": "indicator"}],
                   [{"type": "pie"}, {"type": "indicator"}]]
        )

        fig.add_trace(
            go.Bar(
                x=['Income', 'Expenses', 'Savings'],
                y=[budget_data['total_income'], budget_data['total_expenses'], budget_data['savings']],
                marker_color=['#2ecc71', '#e74c3c', '#3498db'],
                name='Amount',
                text=[f"${budget_data['total_income']:,.0f}", f"${budget_data['total_expenses']:,.0f}", f"${budget_data['savings']:,.0f}"],
                textposition='outside'
            ),
            row=1, col=1
        )

        fig.add_trace(
            go.Indicator(
                mode="gauge+number+delta",
                value=budget_data['savings_rate'],
                domain={'x': [0, 1], 'y': [0, 1]},
                title={'text': "Savings Rate (%)", 'font': {'size': 14}},
                gauge={
                    'axis': {'range': [None, 30]},
                    'bar': {'color': budget_data['health_color']},
                    'steps': [
                        {'range': [0, 10], 'color': "lightgray"},
                        {'range': [10, 20], 'color': "gray"}
                    ],
                    'threshold': {'line': {'color': "red", 'width': 4}, 'thickness': 0.75, 'value': 20}
                },
                number={'font': {'size': 28}}
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

        fig.add_trace(
            go.Indicator(
                mode="gauge+number",
                value=budget_data['health_score'],
                title={'text': f"Health Score: {budget_data['financial_health']}", 'font': {'size': 14}},
                gauge={
                    'axis': {'range': [None, 100]},
                    'bar': {'color': budget_data['health_color']},
                    'steps': [
                        {'range': [0, 25], 'color': "#ffebee"},
                        {'range': [25, 45], 'color': "#fff3e0"},
                        {'range': [45, 65], 'color': "#f3e5f5"},
                        {'range': [65, 80], 'color': "#e8f5e8"},
                        {'range': [80, 100], 'color': "#c8e6c9"}
                    ]
                },
                number={'font': {'size': 28}}
            ),
            row=2, col=2
        )

        fig.update_layout(
            height=800,
            showlegend=False,
            title_text="Budget Analysis Dashboard",
            paper_bgcolor='#1f2937',
            plot_bgcolor='#1f2937',
            font_color='white',
            margin=dict(t=100, b=60, l=50, r=50)
        )
        return fig

    @staticmethod
    def plot_investment_allocation(allocation_data: Dict[str, Any]) -> go.Figure:
        """Create investment allocation visualization."""
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Asset Allocation', 'Portfolio Projections', 'Risk vs Return', 'Dollar Allocation'),
            specs=[[{"type": "pie"}, {"type": "scatter"}],
                   [{"type": "scatter"}, {"type": "bar"}]]
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
            height=800,
            showlegend=True,
            title_text="Investment Portfolio Analysis",
            paper_bgcolor='#1f2937',
            plot_bgcolor='#1f2937',
            font_color='white',
            margin=dict(t=100, b=60, l=50, r=50)
        )
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
                   [{"type": "bar"}, {"type": "bar"}]]
        )

        debts = scenarios.get('minimum_only', {}).get('payoff_plan', [])

        if debts:
            debt_names = [debt['debt_name'] for debt in debts]
            balances = [debt['balance'] for debt in debts]
            months = [debt['months_to_payoff'] for debt in debts]
            interest_rates = [debt['interest_rate'] for debt in debts]
            payments = [debt['monthly_payment'] for debt in debts]

            fig.add_trace(go.Bar(x=debt_names, y=balances, name='Balance', marker_color='red'), row=1, col=1)
            fig.add_trace(go.Bar(x=debt_names, y=months, name='Months to Payoff', marker_color='blue'), row=1, col=2)
            fig.add_trace(go.Bar(x=debt_names, y=interest_rates, name='Interest Rate (%)', marker_color='orange'), row=2, col=1)
            fig.add_trace(go.Bar(x=debt_names, y=payments, name='Monthly Payment', marker_color='green'), row=2, col=2)

        fig.update_layout(
            height=800,
            showlegend=False,
            title_text="Debt Payoff Analysis",
            paper_bgcolor='#1f2937',
            plot_bgcolor='#1f2937',
            font_color='white',
            margin=dict(t=100, b=60, l=50, r=50)
        )
        return fig

    @staticmethod
    def plot_retirement_projections(retirement_data: Dict[str, Any]) -> go.Figure:
        """Create retirement planning visualization."""
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Retirement Scenarios', 'Contribution Impact', 'Savings Growth', 'Income Replacement'),
            specs=[[{"type": "bar"}, {"type": "scatter"}],
                   [{"type": "scatter"}, {"type": "indicator"}]]
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
                title={'text': "Income Replacement (%)", 'font': {'size': 14}},
                gauge={
                    'axis': {'range': [None, 100]},
                    'bar': {'color': "darkblue"},
                    'steps': [
                        {'range': [0, 50], 'color': "lightgray"},
                        {'range': [50, 80], 'color': "gray"}
                    ],
                    'threshold': {'line': {'color': "red", 'width': 4}, 'thickness': 0.75, 'value': 80}
                },
                number={'font': {'size': 28}}
            ),
            row=2, col=2
        )

        fig.update_layout(
            height=800,
            showlegend=True,
            title_text="Retirement Planning Analysis",
            paper_bgcolor='#1f2937',
            plot_bgcolor='#1f2937',
            font_color='white',
            margin=dict(t=100, b=60, l=50, r=50)
        )
        return fig

class FinancialFlows:
    """Structured financial advisory flows"""

    @staticmethod
    def budgeting_flow():
        """Interactive budgeting flow."""
        if not TEST_MODE:
            st.markdown('<div class="flow-card"><h2>💰 Smart Budgeting Assistant</h2><p>Let\'s create a comprehensive budget plan tailored to your financial situation.</p></div>', unsafe_allow_html=True)

        if 'budget_income' not in st.session_state:
            st.session_state.budget_income = 5000.0
        if 'budget_expenses' not in st.session_state:
            st.session_state.budget_expenses = {
                'housing': 0.0, 'utilities': 0.0, 'groceries': 0.0,
                'transportation': 0.0, 'insurance': 0.0, 'healthcare': 0.0,
                'dining_out': 0.0, 'shopping': 0.0, 'subscriptions': 0.0,
                'savings': 0.0, 'debt_payments': 0.0, 'other': 0.0
            }

        if not TEST_MODE:
            st.subheader("Step 1: Monthly Income")
            col1, col2 = st.columns(2)

            with col1:
                primary_income = st.number_input("Primary Income (after taxes)", min_value=0.0, value=st.session_state.budget_income, step=100.0, key="primary_income")
                secondary_income = st.number_input("Secondary Income", min_value=0.0, value=0.0, step=100.0, key="secondary_income")

            with col2:
                other_income = st.number_input("Other Income (investments, etc.)", min_value=0.0, value=0.0, step=100.0, key="other_income")
                total_income = primary_income + secondary_income + other_income
                st.metric("Total Monthly Income", f"${total_income:,.2f}")

            st.session_state.budget_income = total_income

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
                    expenses[key] = st.number_input(label, min_value=0.0, value=st.session_state.budget_expenses.get(key, 0.0), step=50.0, key=f"expense_{key}")

            st.session_state.budget_expenses = expenses

            if st.button("Analyze My Budget", type="primary"):
                budget_summary = FinancialCalculator.calculate_budget_summary(st.session_state.budget_income, st.session_state.budget_expenses)
                ai_insights = generate_ai_insights(budget_summary, "Budget Analysis")

                col1, col2, col3, col4 = st.columns(4)

                with col1:
                    st.markdown(f'''
                    <div class="metric-card">
                        <h3>Total Income</h3>
                        <h2>${budget_summary["total_income"]:,.2f}</h2>
                    </div>
                    ''', unsafe_allow_html=True)

                with col2:
                    st.markdown(f'''
                    <div class="metric-card">
                        <h3>Total Expenses</h3>
                        <h2>${budget_summary["total_expenses"]:,.2f}</h2>
                    </div>
                    ''', unsafe_allow_html=True)

                with col3:
                    st.markdown(f'''
                    <div class="metric-card">
                        <h3>Monthly Savings</h3>
                        <h2 style="color: {'green' if budget_summary["savings"] >= 0 else 'red'}">${budget_summary["savings"]:,.2f}</h2>
                    </div>
                    ''', unsafe_allow_html=True)

                with col4:
                    ai_score = ai_insights.get("ai_score")
                    if ai_score is not None:
                        st.markdown(f'''
                        <div class="metric-card">
                            <h3>Financial Score</h3>
                            <h2 style="color: {budget_summary["health_color"]}">{ai_score}/100</h2>
                        </div>
                        ''', unsafe_allow_html=True)
                    else:
                        st.markdown(f'''
                        <div class="metric-card">
                            <h3>Financial Score</h3>
                            <h2 style="color: {budget_summary["health_color"]}">{budget_summary["health_score"]}/100</h2>
                        </div>
                        ''', unsafe_allow_html=True)

                st.markdown(f'''
                <div class="summary-card">
                    <h3>Financial Health: {budget_summary["financial_health"]} (Score: {budget_summary["health_score"]}/100)</h3>
                    <h4>Recommendations:</h4>
                    <ul>
                        {"".join(f"<li>{rec}</li>" for rec in budget_summary["recommendations"])}
                    </ul>
                </div>
                ''', unsafe_allow_html=True)

                display_ai_suggestions(ai_insights, "Budget Analysis")

                st.subheader("Budget Analysis Dashboard")
                budget_viz = FinancialVisualizer.plot_budget_summary(budget_summary)
                st.plotly_chart(budget_viz, use_container_width=True)

                st.session_state.budget_data = budget_summary
                st.session_state.budget_ai_insights = ai_insights
        else:
            expenses = {'housing': 1500, 'utilities': 200, 'groceries': 400}
            total_income = 5000
            budget_summary = FinancialCalculator.calculate_budget_summary(total_income, expenses)
            return budget_summary

    @staticmethod
    def investing_flow():
        """Interactive investment planning flow."""
        if not TEST_MODE:
            st.markdown('<div class="flow-card"><h2>📈 Investment Portfolio Builder</h2><p>Let\'s create an optimal investment strategy based on your goals and risk tolerance.</p></div>', unsafe_allow_html=True)

        if 'investment_capital' not in st.session_state:
            st.session_state.investment_capital = 10000.0
        if 'investment_time_horizon' not in st.session_state:
            st.session_state.investment_time_horizon = 10
        if 'investment_age' not in st.session_state:
            st.session_state.investment_age = 35

        if not TEST_MODE:
            st.subheader("Step 1: Investment Goals & Timeline")
            col1, col2 = st.columns(2)

            with col1:
                investment_goal = st.selectbox(
                    "Primary Investment Goal",
                    ["Retirement", "House Down Payment", "Emergency Fund", "Wealth Building", "Education", "Other"],
                    key="investment_goal"
                )
                time_horizon = st.slider("Investment Time Horizon (years)", 1, 40, st.session_state.investment_time_horizon, key="time_horizon_slider")
                st.session_state.investment_time_horizon = time_horizon

            with col2:
                current_age = st.number_input("Your Current Age", min_value=18, max_value=80, value=st.session_state.investment_age, key="current_age")
                st.session_state.investment_age = current_age
                investment_capital = st.number_input("Initial Investment Amount", min_value=0.0, value=st.session_state.investment_capital, step=1000.0, key="investment_capital_input")
                st.session_state.investment_capital = investment_capital

            st.subheader("Step 2: Risk Tolerance Assessment")

            market_drop = st.radio(
                "If your portfolio dropped 20% in a month, you would:",
                ["Panic and sell everything", "Feel uncomfortable but hold", "See it as a buying opportunity"],
                key="market_drop"
            )

            investment_experience = st.radio(
                "Your investment experience level:",
                ["Beginner (< 2 years)", "Intermediate (2-10 years)", "Advanced (> 10 years)"],
                key="investment_experience"
            )

            income_stability = st.radio(
                "Your income stability:",
                ["Unstable/Variable", "Stable", "Very Stable with Growth"],
                key="income_stability"
            )

            sleep_factor = st.radio(
                "Regarding investment volatility:",
                ["I need stable, predictable returns", "I can handle some ups and downs", "I'm comfortable with high volatility for higher returns"],
                key="sleep_factor"
            )

            if st.button("Generate Investment Portfolio", type="primary"):
                risk_score = 0
                risk_weights = {
                    "market_drop": {"Panic and sell everything": 1, "Feel uncomfortable but hold": 2, "See it as a buying opportunity": 3},
                    "investment_experience": {"Beginner (< 2 years)": 1, "Intermediate (2-10 years)": 2, "Advanced (> 10 years)": 3},
                    "income_stability": {"Unstable/Variable": 1, "Stable": 2, "Very Stable with Growth": 3},
                    "sleep_factor": {"I need stable, predictable returns": 1, "I can handle some ups and downs": 2, "I'm comfortable with high volatility for higher returns": 3}
                }

                risk_answers = {
                    "market_drop": market_drop,
                    "investment_experience": investment_experience,
                    "income_stability": income_stability,
                    "sleep_factor": sleep_factor
                }

                for question, answer in risk_answers.items():
                    risk_score += risk_weights[question][answer]

                if risk_score <= 6:
                    risk_profile = "Conservative"
                elif risk_score <= 9:
                    risk_profile = "Moderate"
                else:
                    risk_profile = "Aggressive"

                st.info(f"Based on your responses, your risk profile is: **{risk_profile}**")

                allocation_data = FinancialCalculator.calculate_investment_allocation(
                    risk_profile, st.session_state.investment_time_horizon, st.session_state.investment_capital, st.session_state.investment_age
                )

                ai_insights = generate_ai_insights(allocation_data, "Investment Analysis")

                st.subheader("Recommended Portfolio Allocation")

                col1, col2, col3 = st.columns(3)
                allocation = allocation_data['allocation_percentages']

                with col1:
                    st.markdown(f'''
                    <div class="metric-card">
                        <h3>Stocks</h3>
                        <h2>{allocation["stocks"]}%</h2>
                        <p>${allocation_data["allocation_dollars"]["stocks"]:,.0f}</p>
                    </div>
                    ''', unsafe_allow_html=True)

                with col2:
                    st.markdown(f'''
                    <div class="metric-card">
                        <h3>Bonds</h3>
                        <h2>{allocation["bonds"]}%</h2>
                        <p>${allocation_data["allocation_dollars"]["bonds"]:,.0f}</p>
                    </div>
                    ''', unsafe_allow_html=True)

                with col3:
                    st.markdown(f'''
                    <div class="metric-card">
                        <h3>Cash</h3>
                        <h2>{allocation["cash"]}%</h2>
                        <p>${allocation_data["allocation_dollars"]["cash"]:,.0f}</p>
                    </div>
                    ''', unsafe_allow_html=True)

                ai_score = ai_insights.get("ai_score")
                ai_score_text = f" | AI Score: {ai_score}/100" if ai_score else ""

                st.markdown(f'''
                <div class="summary-card">
                    <h3>Portfolio Metrics{ai_score_text}</h3>
                    <p><strong>Expected Annual Return:</strong> {allocation_data["expected_annual_return"]:.1%}</p>
                    <p><strong>Estimated Volatility:</strong> {allocation_data["volatility_estimate"]:.1%}</p>
                    <p><strong>Risk Level:</strong> {allocation_data["risk_level"]}</p>
                </div>
                ''', unsafe_allow_html=True)

                display_ai_suggestions(ai_insights, "Investment Analysis")

                if allocation_data.get('projections'):
                    st.subheader("Portfolio Growth Projections")
                    projections_df = pd.DataFrame(allocation_data['projections']).T
                    st.dataframe(projections_df.style.format("${:,.0f}"))

                st.subheader("Investment Portfolio Analysis")
                investment_viz = FinancialVisualizer.plot_investment_allocation(allocation_data)
                st.plotly_chart(investment_viz, use_container_width=True)

                st.session_state.investment_data = allocation_data
                st.session_state.investment_ai_insights = ai_insights
        else:
            allocation_data = FinancialCalculator.calculate_investment_allocation('moderate', 15, 25000, 35)
            return allocation_data

    @staticmethod
    def debt_repayment_flow():
        """Interactive debt repayment planning flow."""
        if not TEST_MODE:
            st.markdown('<div class="flow-card"><h2>💳 Debt Freedom Planner</h2><p>Let\'s create a strategic plan to eliminate your debt efficiently.</p></div>', unsafe_allow_html=True)

        if 'debts' not in st.session_state:
            st.session_state.debts = []

        if not TEST_MODE:
            st.subheader("Step 1: Your Current Debts")

            with st.expander("Add New Debt", expanded=len(st.session_state.debts) == 0):
                col1, col2 = st.columns(2)

                with col1:
                    debt_name = st.text_input("Debt Name (e.g., Credit Card, Student Loan)", key="new_debt_name")
                    debt_balance = st.number_input("Current Balance", min_value=0.0, step=100.0, key="new_debt_balance")

                with col2:
                    interest_rate = st.number_input("Interest Rate (%)", min_value=0.0, max_value=50.0, step=0.1, key="new_debt_rate")
                    minimum_payment = st.number_input("Minimum Monthly Payment", min_value=0.0, step=10.0, key="new_debt_payment")

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
                {'name': 'Credit Card 2', 'balance': 3000, 'interest_rate': 22.0, 'minimum_payment': 100}
            ]

        if st.session_state.debts:
            if not TEST_MODE:
                st.subheader("Step 2: Repayment Strategy")

                col1, col2 = st.columns(2)

                with col1:
                    strategy = st.selectbox(
                        "Choose Repayment Strategy",
                        ["avalanche", "snowball"],
                        format_func=lambda x: "Debt Avalanche (Highest Interest First)" if x == "avalanche" else "Debt Snowball (Smallest Balance First)",
                        key="debt_strategy"
                    )

                with col2:
                    extra_payment = st.number_input("Extra Monthly Payment Available", min_value=0.0, step=50.0, key="extra_payment")

                if st.button("Create Debt Payoff Plan", type="primary"):
                    debt_analysis = FinancialCalculator.calculate_debt_payoff(st.session_state.debts, extra_payment, strategy)
                    ai_insights = generate_ai_insights(debt_analysis, "Debt Analysis")

                    col1, col2, col3, col4 = st.columns(4)

                    with col1:
                        st.markdown(f'''
                        <div class="metric-card">
                            <h3>Total Debt</h3>
                            <h2>${debt_analysis["total_debt"]:,.2f}</h2>
                        </div>
                        ''', unsafe_allow_html=True)

                    with col2:
                        st.markdown(f'''
                        <div class="metric-card">
                            <h3>Min Payments</h3>
                            <h2>${debt_analysis["total_minimum_payment"]:,.2f}</h2>
                        </div>
                        ''', unsafe_allow_html=True)

                    with col3:
                        if extra_payment > 0:
                            st.markdown(f'''
                            <div class="metric-card">
                                <h3>Interest Savings</h3>
                                <h2 style="color: green">${debt_analysis["interest_savings"]:,.2f}</h2>
                            </div>
                            ''', unsafe_allow_html=True)
                        else:
                            st.markdown(f'''
                            <div class="metric-card">
                                <h3>Interest Savings</h3>
                                <h2>$0</h2>
                            </div>
                            ''', unsafe_allow_html=True)

                    with col4:
                        if extra_payment > 0:
                            st.markdown(f'''
                            <div class="metric-card">
                                <h3>Time Savings</h3>
                                <h2 style="color: green">{debt_analysis["time_savings_months"]:.0f} months</h2>
                            </div>
                            ''', unsafe_allow_html=True)
                        else:
                            st.markdown(f'''
                            <div class="metric-card">
                                <h3>Time Savings</h3>
                                <h2>0 months</h2>
                            </div>
                            ''', unsafe_allow_html=True)

                    st.subheader("Debt Payoff Priority Order")
                    scenario_key = 'with_extra' if extra_payment > 0 else 'minimum_only'
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

                    st.markdown(f'''
                    <div class="summary-card">
                        <h3>Debt Payoff Recommendations</h3>
                        <ul>
                            <li>🎯 Focus on paying ${debt_analysis["recommended_extra_payment"]:.0f} extra per month if possible</li>
                            <li>📊 You're using the <strong>{strategy.title()}</strong> method</li>
                            <li>💡 Consider debt consolidation if you have high-interest credit cards</li>
                            <li>🚫 Avoid taking on new debt during your payoff journey</li>
                            <li>📱 Set up automatic payments to stay on track</li>
                        </ul>
                    </div>
                    ''', unsafe_allow_html=True)

                    display_ai_suggestions(ai_insights, "Debt Analysis")

                    st.subheader("Debt Analysis Dashboard")
                    debt_viz = FinancialVisualizer.plot_debt_payoff(debt_analysis)
                    st.plotly_chart(debt_viz, use_container_width=True)

                    st.session_state.debt_data = debt_analysis
                    st.session_state.debt_ai_insights = ai_insights
            else:
                debt_analysis = FinancialCalculator.calculate_debt_payoff(st.session_state.debts, 200, 'avalanche')
                return debt_analysis

    @staticmethod
    def retirement_planning_flow():
        """Interactive retirement planning flow."""
        if not TEST_MODE:
            st.markdown('<div class="flow-card"><h2>🏖️ Retirement Planning Assistant</h2><p>Let\'s ensure you\'re on track for a comfortable retirement.</p></div>', unsafe_allow_html=True)

        if 'retirement_current_age' not in st.session_state:
            st.session_state.retirement_current_age = 35
        if 'retirement_retirement_age' not in st.session_state:
            st.session_state.retirement_retirement_age = 65
        if 'retirement_current_income' not in st.session_state:
            st.session_state.retirement_current_income = 75000.0
        if 'retirement_current_savings' not in st.session_state:
            st.session_state.retirement_current_savings = 50000.0
        if 'retirement_monthly_contribution' not in st.session_state:
            st.session_state.retirement_monthly_contribution = 500.0
        if 'retirement_employer_match' not in st.session_state:
            st.session_state.retirement_employer_match = 0.0

        if not TEST_MODE:
            st.subheader("Step 1: Current Financial Situation")
            col1, col2 = st.columns(2)

            with col1:
                current_age = st.number_input("Current Age", min_value=18, max_value=80, value=st.session_state.retirement_current_age, key="ret_current_age")
                st.session_state.retirement_current_age = current_age

                retirement_age = st.number_input("Desired Retirement Age", min_value=50, max_value=80, value=st.session_state.retirement_retirement_age, key="ret_retirement_age")
                st.session_state.retirement_retirement_age = retirement_age

                current_income = st.number_input("Current Annual Income", min_value=0.0, value=st.session_state.retirement_current_income, step=5000.0, key="ret_current_income")
                st.session_state.retirement_current_income = current_income

            with col2:
                current_savings = st.number_input("Current Retirement Savings", min_value=0.0, value=st.session_state.retirement_current_savings, step=5000.0, key="ret_current_savings")
                st.session_state.retirement_current_savings = current_savings

                monthly_contribution = st.number_input("Current Monthly Contribution", min_value=0.0, value=st.session_state.retirement_monthly_contribution, step=50.0, key="ret_monthly_contribution")
                st.session_state.retirement_monthly_contribution = monthly_contribution

                employer_match = st.number_input("Employer Match (monthly)", min_value=0.0, value=st.session_state.retirement_employer_match, step=50.0, key="ret_employer_match")
                st.session_state.retirement_employer_match = employer_match

            st.subheader("Step 2: Retirement Lifestyle Goals")

            lifestyle_choice = st.selectbox(
                "Desired Retirement Lifestyle",
                ["Basic (60% of current income)", "Comfortable (80% of current income)", "Luxurious (100% of current income)"],
                key="lifestyle_choice"
            )

            col1, col2 = st.columns(2)

            with col1:
                healthcare_inflation = st.checkbox("Account for higher healthcare costs", value=True, key="healthcare_inflation")
                social_security = st.checkbox("Include Social Security benefits", value=True, key="social_security")

            with col2:
                inheritance_expected = st.number_input("Expected Inheritance", min_value=0.0, value=0.0, step=10000.0, key="inheritance_expected")
                other_retirement_income = st.number_input("Other Retirement Income (monthly)", min_value=0.0, value=0.0, step=100.0, key="other_retirement_income")

            if st.button("Analyze Retirement Plan", type="primary"):
                total_monthly_contribution = st.session_state.retirement_monthly_contribution + st.session_state.retirement_employer_match

                retirement_analysis = FinancialCalculator.calculate_retirement_needs(
                    st.session_state.retirement_current_age,
                    st.session_state.retirement_retirement_age,
                    st.session_state.retirement_current_income,
                    st.session_state.retirement_current_savings,
                    total_monthly_contribution
                )

                ai_insights = generate_ai_insights(retirement_analysis, "Retirement Analysis")

                col1, col2, col3, col4 = st.columns(4)

                with col1:
                    st.markdown(f'''
                    <div class="metric-card">
                        <h3>Years to Retirement</h3>
                        <h2>{retirement_analysis["years_to_retirement"]}</h2>
                    </div>
                    ''', unsafe_allow_html=True)

                with col2:
                    st.markdown(f'''
                    <div class="metric-card">
                        <h3>Projected Savings</h3>
                        <h2>${retirement_analysis["projected_savings"]:,.0f}</h2>
                    </div>
                    ''', unsafe_allow_html=True)

                with col3:
                    st.markdown(f'''
                    <div class="metric-card">
                        <h3>Retirement Goal</h3>
                        <h2>${retirement_analysis["retirement_corpus_needed"]:,.0f}</h2>
                    </div>
                    ''', unsafe_allow_html=True)

                with col4:
                    gap = retirement_analysis["retirement_gap"]
                    gap_color = "red" if gap > 0 else "green"
                    gap_text = f"${gap:,.0f}" if gap > 0 else "On Track!"

                    ai_score = ai_insights.get("ai_score")
                    if ai_score is not None:
                        st.markdown(f'''
                        <div class="metric-card">
                            <h3>Retirement Gap</h3>
                            <h2 style="color: {gap_color}">{gap_text}</h2>
                            <p>AI Score: {ai_score}/100</p>
                        </div>
                        ''', unsafe_allow_html=True)
                    else:
                        st.markdown(f'''
                        <div class="metric-card">
                            <h3>Retirement Gap</h3>
                            <h2 style="color: {gap_color}">{gap_text}</h2>
                        </div>
                        ''', unsafe_allow_html=True)

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
                    'Income Replacement': [f"{scenarios['conservative']['replacement_ratio_achieved']:.1%}",
                                         f"{scenarios['current']['replacement_ratio_achieved']:.1%}",
                                         f"{scenarios['aggressive']['replacement_ratio_achieved']:.1%}"]
                })

                st.dataframe(scenario_df, use_container_width=True)

                st.markdown(f'''
                <div class="summary-card">
                    <h3>Retirement Planning Recommendations</h3>
                    <ul>
                        {"".join(f"<li>{rec}</li>" for rec in retirement_analysis["recommendations"])}
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

                st.subheader("Retirement Planning Dashboard")
                retirement_viz = FinancialVisualizer.plot_retirement_projections(retirement_analysis)
                st.plotly_chart(retirement_viz, use_container_width=True)

                st.session_state.retirement_data = retirement_analysis
                st.session_state.retirement_ai_insights = ai_insights
        else:
            retirement_analysis = FinancialCalculator.calculate_retirement_needs(35, 65, 75000, 50000, 600)
            return retirement_analysis

def main():
    """Main application function"""
    if TEST_MODE:
        return

    st.markdown('<h1 class="main-header">🦙 AI Financial Advisor - LLAMA 3.3</h1>', unsafe_allow_html=True)

    st.info("💡 **Disclaimer**: AI suggestions are educational only and not financial advice. Always consult with a qualified financial professional.")

    st.sidebar.subheader("📊 Financial Tools")
    selected_flow = st.sidebar.selectbox(
        "Choose a Financial Flow",
        ["Demo Dashboard", "Smart Budgeting", "Investment Planning", "Debt Repayment", "Retirement Planning"]
    )

    if selected_flow == "Smart Budgeting":
        FinancialFlows.budgeting_flow()
    elif selected_flow == "Investment Planning":
        FinancialFlows.investing_flow()
    elif selected_flow == "Debt Repayment":
        FinancialFlows.debt_repayment_flow()
    elif selected_flow == "Retirement Planning":
        FinancialFlows.retirement_planning_flow()
    elif selected_flow == "Demo Dashboard":
        st.markdown('<div class="flow-card"><h2>📊 Demo Financial Dashboard</h2><p>Explore sample financial data and visualizations</p></div>', unsafe_allow_html=True)

        demo_income = 7000
        demo_expenses = {
            "housing": 1800,
            "utilities": 400,
            "food": 900,
            "transportation": 500,
            "entertainment": 300,
            "insurance": 350
        }
        demo_debts = [15000, 5000]
        demo_savings = 2500

        total_expenses = sum(demo_expenses.values())
        monthly_savings = demo_income - total_expenses
        savings_rate = (monthly_savings / demo_income * 100) if demo_income > 0 else 0
        total_debt = sum(demo_debts)
        dti_ratio = (total_debt / (demo_income * 12)) if demo_income > 0 else 0
        emergency_fund_months = (demo_savings / total_expenses) if total_expenses > 0 else 0

        health_score = 50
        if savings_rate > 20:
            health_score += 20
        elif savings_rate > 10:
            health_score += 10
        if dti_ratio < 0.36:
            health_score += 15
        if emergency_fund_months >= 3:
            health_score += 15
        health_score = min(100, health_score)

        st.markdown("### 📈 Key Financial Metrics")
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric("Savings Rate", f"{savings_rate:.1f}%", delta=f"{savings_rate - 20:.1f}% vs target")

        with col2:
            st.metric("Financial Health", f"{health_score:.0f}/100", delta="Good" if health_score >= 70 else "Needs Work")

        with col3:
            st.metric("DTI Ratio", f"{dti_ratio:.1%}", delta="Healthy" if dti_ratio < 0.36 else "High", delta_color="inverse")

        with col4:
            st.metric("Emergency Fund", f"{emergency_fund_months:.1f} mo", delta="Ready" if emergency_fund_months >= 3 else "Build Up")

        st.markdown("### 📊 Visual Analysis")

        chart_col1, chart_col2 = st.columns(2)

        with chart_col1:
            fig1 = go.Figure(data=[
                go.Bar(name='Income', x=['Monthly Cash Flow'], y=[demo_income], marker_color='#10b981', text=[f'${demo_income:,.0f}'], textposition='outside', textfont=dict(size=14)),
                go.Bar(name='Expenses', x=['Monthly Cash Flow'], y=[total_expenses], marker_color='#ef4444', text=[f'${total_expenses:,.0f}'], textposition='outside', textfont=dict(size=14))
            ])
            fig1.update_layout(
                title='Income vs Expenses',
                barmode='group',
                plot_bgcolor='#1f2937',
                paper_bgcolor='#1f2937',
                font=dict(color='#ffffff'),
                height=400,
                margin=dict(t=60, b=50, l=50, r=50),
                yaxis=dict(gridcolor='#374151')
            )
            st.plotly_chart(fig1, use_container_width=True)

            fig3 = go.Figure(go.Indicator(
                mode="gauge+number+delta",
                value=savings_rate,
                domain={'x': [0, 1], 'y': [0, 1]},
                title={'text': "Savings Rate (%)", 'font': {'size': 14}},
                delta={'reference': 20, 'increasing': {'color': "#10b981"}},
                gauge={
                    'axis': {'range': [None, 100], 'tickwidth': 1, 'tickcolor': "#ffffff"},
                    'bar': {'color': "#3b82f6"},
                    'steps': [
                        {'range': [0, 10], 'color': "#ef4444"},
                        {'range': [10, 20], 'color': "#f59e0b"},
                        {'range': [20, 100], 'color': "#10b981"}
                    ],
                    'threshold': {'line': {'color': "white", 'width': 2}, 'thickness': 0.75, 'value': 20}
                },
                number={'font': {'size': 28}}
            ))
            fig3.update_layout(
                plot_bgcolor='#1f2937',
                paper_bgcolor='#1f2937',
                font=dict(color='#ffffff'),
                height=350,
                margin=dict(t=80, b=40, l=40, r=40)
            )
            st.plotly_chart(fig3, use_container_width=True)

        with chart_col2:
            labels = list(demo_expenses.keys())
            values = list(demo_expenses.values())
            fig2 = go.Figure(data=[go.Pie(labels=labels, values=values, hole=.3)])
            fig2.update_layout(
                title='Expense Breakdown',
                plot_bgcolor='#1f2937',
                paper_bgcolor='#1f2937',
                font=dict(color='#ffffff'),
                height=400,
                margin=dict(t=60, b=40, l=40, r=40)
            )
            st.plotly_chart(fig2, use_container_width=True)

            dti_percent = dti_ratio * 100
            fig4 = go.Figure(go.Indicator(
                mode="gauge+number",
                value=dti_percent,
                domain={'x': [0, 1], 'y': [0, 1]},
                title={'text': "Debt-to-Income Ratio (%)", 'font': {'size': 14}},
                gauge={
                    'axis': {'range': [None, 50], 'tickwidth': 1, 'tickcolor': "#ffffff"},
                    'bar': {'color': "#3b82f6"},
                    'steps': [
                        {'range': [0, 20], 'color': "#10b981"},
                        {'range': [20, 36], 'color': "#f59e0b"},
                        {'range': [36, 50], 'color': "#ef4444"}
                    ],
                    'threshold': {'line': {'color': "white", 'width': 2}, 'thickness': 0.75, 'value': 36}
                },
                number={'font': {'size': 28}}
            ))
            fig4.update_layout(
                plot_bgcolor='#1f2937',
                paper_bgcolor='#1f2937',
                font=dict(color='#ffffff'),
                height=350,
                margin=dict(t=80, b=40, l=40, r=40)
            )
            st.plotly_chart(fig4, use_container_width=True)

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
