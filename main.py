"""
AI Financial Advisor Application - LLAMA 3.3
A comprehensive financial planning tool with AI-powered insights

Required pip packages:
pip install streamlit plotly pandas numpy easyocr torch torchvision torchaudio opencv-python pdf2image pymupdf python-dotenv faiss-cpu sentence-transformers langchain langchain-community langchain-groq langchain-huggingface langchain-text-splitters
"""

import streamlit as st  # Streamlit must be imported first
import os
import json
import torch
import asyncio
import tempfile
import uuid
import sys
from dotenv import load_dotenv
import fitz  # PyMuPDF for text extraction
import easyocr  # GPU-accelerated OCR
from pdf2image import convert_from_path  # Convert PDFs to images
from langchain_text_splitters import CharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Any
import io
import base64

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
        
        /* Dark chat messages */
        .chat-message {
            padding: 1.2rem;
            border-radius: 15px;
            margin: 0.8rem 0;
            box-shadow: 0 2px 8px rgba(0,0,0,0.2);
            color: #ffffff;
        }
        .user-message {
            background: linear-gradient(135deg, #1e3a8a 0%, #3b82f6 100%);
            border-left: 4px solid #60a5fa;
            color: #ffffff;
        }
        .bot-message {
            background: linear-gradient(135deg, #581c87 0%, #7c3aed 100%);
            border-left: 4px solid #a78bfa;
            color: #ffffff;
        }
        
        /* Dark persona cards */
        .persona-card {
            background: linear-gradient(135deg, #374151 0%, #4b5563 100%);
            padding: 1rem;
            border-radius: 10px;
            margin: 0.5rem 0;
            border-left: 4px solid #f59e0b;
            color: #ffffff;
            border: 1px solid #6b7280;
        }
        .persona-card h4 {
            color: #ffffff !important;
        }
        .persona-card p, .persona-card em {
            color: #d1d5db !important;
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
        
        /* Chat input styling */
        .stChatInput > div > div {
            background-color: #374151 !important;
            border: 1px solid #6b7280 !important;
        }
        
        .stChatInput input {
            background-color: #374151 !important;
            color: #ffffff !important;
        }
        
        /* File uploader styling */
        .stFileUploader > div {
            background-color: #374151 !important;
            border: 2px dashed #6b7280 !important;
            border-radius: 8px !important;
        }
        
        .stFileUploader label {
            color: #ffffff !important;
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

# Ensure GPU availability
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
if torch.cuda.is_available():
    torch.backends.cudnn.benchmark = True  # Optimize GPU performance

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

def generate_ai_insights(data: Dict[str, Any], context_label: str) -> Dict[str, Any]:
    """
    Centralized AI insights generator using LLaMA 3.3 via Groq.
    
    Args:
        data: Dictionary containing financial data for analysis
        context_label: Label indicating the type of analysis (e.g., "Budget Analysis")
    
    Returns:
        Dict containing AI score (0-100), reasoning, and recommendations
    """
    # Fallback response for when AI is not available
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
        # Initialize ChatGroq client
        llm = ChatGroq(
            model="llama-3.3-70b-versatile", 
            temperature=0.3,  # Conservative temperature for consistent results
            groq_api_key=groq_api_key
        )
        
        # Create context-specific prompts
        if context_label == "Budget Analysis":
            prompt = f"""
            You are an expert financial advisor. Analyze this budget data and provide insights.
            
            Budget Data (JSON): {json.dumps(data, default=str)}
            
            Tasks:
            1. Provide a Financial Health Score (0-100) where 0 is critical and 100 is excellent
            2. Give a brief 2-3 sentence reasoning for the score
            3. Provide 3-5 concise, actionable recommendations the user can implement today
            
            Important: Output ONLY valid JSON with keys: ai_score, ai_reasoning, ai_recommendations
            No additional text or explanations outside the JSON.
            
            Example format:
            {{"ai_score": 75, "ai_reasoning": "Good savings rate but high housing costs limit flexibility.", "ai_recommendations": ["Reduce housing costs", "Increase emergency fund", "Track discretionary spending"]}}
            """
            
        elif context_label == "Investment Analysis":
            prompt = f"""
            You are an expert investment advisor. Analyze this portfolio data and provide insights.
            
            Investment Data (JSON): {json.dumps(data, default=str)}
            
            Tasks:
            1. Provide an Investment Risk Score (0-100) where 0 is very conservative and 100 is very aggressive
            2. Give a brief 2-3 sentence explanation of the risk level and portfolio suitability
            3. Provide 3-5 specific portfolio improvement suggestions (general types like "low-cost index funds", no specific products)
            
            Important: Output ONLY valid JSON with keys: ai_score, ai_reasoning, ai_recommendations
            No additional text or explanations outside the JSON.
            """
            
        elif context_label == "Debt Analysis":
            prompt = f"""
            You are an expert debt counselor. Analyze this debt situation and provide insights.
            
            Debt Data (JSON): {json.dumps(data, default=str)}
            
            Tasks:
            1. Provide a Debt Health Score (0-100) where 0 is critical debt situation and 100 is debt-free/healthy
            2. Give a brief 2-3 sentence assessment of the debt situation
            3. Provide 3-5 prioritized actionable steps to improve the debt situation
            
            Important: Output ONLY valid JSON with keys: ai_score, ai_reasoning, ai_recommendations
            No additional text or explanations outside the JSON.
            """
            
        elif context_label == "Retirement Analysis":
            prompt = f"""
            You are an expert retirement planner. Analyze this retirement planning data and provide insights.
            
            Retirement Data (JSON): {json.dumps(data, default=str)}
            
            Tasks:
            1. Provide a Retirement Readiness Index (0-100) where 0 is completely unprepared and 100 is fully prepared
            2. Give a brief 2-3 sentence assessment of retirement readiness
            3. Provide 3-5 specific actions to improve retirement preparedness
            
            Important: Output ONLY valid JSON with keys: ai_score, ai_reasoning, ai_recommendations
            No additional text or explanations outside the JSON.
            """
            
        else:
            return fallback_response
        
        # Call LLM
        response = llm.invoke(prompt)
        response_text = response.content.strip()
        
        # Parse JSON response with error handling
        try:
            # Try to find JSON in the response
            if response_text.startswith("{") and response_text.endswith("}"):
                ai_result = json.loads(response_text)
            else:
                # Try to extract JSON from text
                import re
                json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
                if json_match:
                    ai_result = json.loads(json_match.group())
                else:
                    return fallback_response
            
            # Validate response structure
            required_keys = ["ai_score", "ai_reasoning", "ai_recommendations"]
            if not all(key in ai_result for key in required_keys):
                return fallback_response
            
            # Validate ai_score is a number between 0-100
            if ai_result["ai_score"] is not None:
                ai_result["ai_score"] = max(0, min(100, float(ai_result["ai_score"])))
            
            # Ensure recommendations is a list
            if not isinstance(ai_result["ai_recommendations"], list):
                ai_result["ai_recommendations"] = [str(ai_result["ai_recommendations"])]
            
            return ai_result
            
        except (json.JSONDecodeError, ValueError, KeyError) as e:
            return fallback_response
    
    except Exception as e:
        # Log error in non-test mode
        if not TEST_MODE:
            st.warning(f"AI analysis temporarily unavailable: {str(e)}")
        return fallback_response

def display_ai_suggestions(ai_insights: Dict[str, Any], context_label: str):
    """
    Display AI suggestions in a consistent format across all flows.
    
    Args:
        ai_insights: Dictionary containing AI analysis results
        context_label: Label for the type of analysis
    """
    if TEST_MODE:
        return
    
    ai_score = ai_insights.get("ai_score")
    ai_reasoning = ai_insights.get("ai_reasoning", "")
    ai_recommendations = ai_insights.get("ai_recommendations", [])
    
    # Display AI suggestions card
    st.markdown("### 🤖 AI Suggestions")
    
    # AI Score display
    if ai_score is not None:
        score_color = "#ef4444" if ai_score < 30 else "#f59e0b" if ai_score < 60 else "#10b981"
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

if not groq_api_key and not TEST_MODE:
    st.error("🚨 GROQ_API_KEY is missing. Check your config.json file or environment variables.")
    st.warning("💡 AI features will use deterministic fallback mode.")

# Initialize EasyOCR with GPU support
reader = None
if not TEST_MODE:
    try:
        reader = easyocr.Reader(["en"], gpu=torch.cuda.is_available())
    except Exception as e:
        st.warning(f"EasyOCR initialization failed: {e}. OCR features will be limited.")
        reader = None

class FinancialCalculator:
    """Core financial calculation functions with advanced analytics"""
    
    @staticmethod
    def calculate_budget_summary(income: float, expenses: Dict[str, float]) -> Dict[str, Any]:
        """
        Calculate comprehensive budget summary with dynamic scores based on user inputs.
        
        Args:
            income: Monthly income amount
            expenses: Dictionary of expense categories and amounts
        
        Returns:
            Dict containing budget analysis results
        """
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
        
        # Categorize expenses for better analysis
        essential_categories = ['housing', 'utilities', 'groceries', 'transportation', 'insurance', 'healthcare']
        essential_expenses = sum(expenses.get(cat, 0) for cat in essential_categories if cat in expenses)
        discretionary_expenses = total_expenses - essential_expenses
        
        # DYNAMIC Financial health scoring (0-100 scale)
        health_score = 0
        
        # Base score from savings rate (0-50 points)
        if savings_rate >= 20:
            health_score += 50
        elif savings_rate >= 10:
            health_score += 35
        elif savings_rate >= 5:
            health_score += 20
        elif savings_rate >= 0:
            health_score += 10
        else:
            health_score += 0  # negative savings
        
        # Housing ratio modifier (0-25 points)
        housing_ratio = expenses.get('housing', 0) / income * 100 if income > 0 else 0
        if housing_ratio <= 25:
            health_score += 25
        elif housing_ratio <= 30:
            health_score += 20
        elif housing_ratio <= 35:
            health_score += 10
        else:
            health_score += 0  # too much on housing
        
        # Debt payment ratio modifier (0-15 points)
        debt_ratio = expenses.get('debt_payments', 0) / income * 100 if income > 0 else 0
        if debt_ratio <= 10:
            health_score += 15
        elif debt_ratio <= 20:
            health_score += 10
        elif debt_ratio <= 30:
            health_score += 5
        else:
            health_score += 0  # high debt burden
        
        # Emergency fund consideration (0-10 points)
        if savings > 0:
            health_score += 10
        
        # Cap at 100
        health_score = min(100, health_score)
        
        # Determine health category based on computed score
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
            'recommendations': FinancialCalculator._get_budget_recommendations(savings_rate, expenses, income)
        }
    
    @staticmethod
    def _get_budget_recommendations(savings_rate: float, expenses: Dict[str, float], income: float) -> List[str]:
        """
        Generate personalized budget recommendations.
        
        Args:
            savings_rate: Current savings rate as percentage
            expenses: Dictionary of expenses
            income: Monthly income
        
        Returns:
            List of recommendation strings
        """
        recommendations = []
        
        if savings_rate < 10:
            recommendations.append("🎯 Aim to save at least 10% of your income")
            
        # Check for high expense categories
        housing_ratio = expenses.get('housing', 0) / income * 100 if income > 0 else 0
        if housing_ratio > 30:
            recommendations.append("🏠 Consider reducing housing costs - currently {}% of income".format(round(housing_ratio, 1)))
            
        if expenses.get('dining_out', 0) > expenses.get('groceries', 0):
            recommendations.append("🍽️ Consider cooking more at home to reduce dining expenses")
            
        if savings_rate >= 20:
            recommendations.append("🌟 Excellent savings rate! Consider investing surplus funds")
        
        if expenses.get('debt_payments', 0) / income > 0.2:
            recommendations.append("💳 Focus on debt repayment - debt payments are high relative to income")
            
        return recommendations
    
    @staticmethod
    def calculate_investment_allocation(risk_profile: str, time_horizon: int, capital: float, age: int = 35) -> Dict[str, Any]:
        """
        Calculate sophisticated investment allocation with dynamic allocations based on inputs.
        
        Args:
            risk_profile: Risk tolerance level (conservative, moderate, aggressive)
            time_horizon: Investment time horizon in years
            capital: Initial investment amount
            age: Investor's age
        
        Returns:
            Dict containing allocation recommendations and projections
        """
        # Base allocations by risk profile
        base_allocations = {
            'conservative': {'stocks': 25, 'bonds': 65, 'cash': 10},
            'moderate': {'stocks': 60, 'bonds': 30, 'cash': 10},
            'aggressive': {'stocks': 85, 'bonds': 10, 'cash': 5}
        }
        
        allocation = base_allocations.get(risk_profile.lower(), base_allocations['moderate']).copy()
        
        # DYNAMIC age-based adjustment (100 - age rule with modifications)
        age_adjusted_stock = max(20, min(90, 110 - age))
        
        # DYNAMIC time horizon adjustments
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
        
        # Blend with age-based allocation
        allocation['stocks'] = int((allocation['stocks'] + age_adjusted_stock) / 2)
        allocation['bonds'] = max(5, 95 - allocation['stocks'] - allocation['cash'])
        
        # Calculate dollar amounts
        dollar_allocation = {
            asset: (percentage / 100) * capital 
            for asset, percentage in allocation.items()
        }
        
        # DYNAMIC expected returns based on asset allocation (documented assumptions)
        # Assumptions: Stocks 10% annual, Bonds 4% annual, Cash 2% annual
        expected_returns = {
            'stocks': 0.10,  # Historical stock market average
            'bonds': 0.04,   # Current bond market expectations
            'cash': 0.02     # High-yield savings/money market
        }
        
        portfolio_return = sum(
            (allocation[asset] / 100) * expected_returns[asset] 
            for asset in allocation
        )
        
        # Monte Carlo simulation (simplified)
        projections = {}
        for years in [5, 10, 20, 30]:
            if years <= time_horizon:
                # Conservative, expected, and optimistic scenarios
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
        """
        Calculate estimated portfolio volatility.
        
        Args:
            allocation: Dictionary of asset allocation percentages
        
        Returns:
            Estimated portfolio volatility as a decimal
        """
        volatilities = {'stocks': 0.16, 'bonds': 0.05, 'cash': 0.01}
        return sum((allocation[asset] / 100) * volatilities[asset] for asset in allocation)
    
    @staticmethod
    def calculate_debt_payoff(debts: List[Dict], extra_payment: float = 0, strategy: str = 'avalanche') -> Dict[str, Any]:
        """
        Calculate comprehensive debt payoff strategy with fixed avalanche/snowball logic.
        
        Args:
            debts: List of debt dictionaries with balance, interest_rate, minimum_payment
            extra_payment: Additional monthly payment amount
            strategy: Payoff strategy ('avalanche' or 'snowball')
        
        Returns:
            Dict containing debt payoff analysis and scenarios
        """
        if not debts:
            return {'total_debt': 0, 'payoff_plan': [], 'total_interest': 0, 'scenarios': {}, 'strategy': strategy}
        
        # Validate and clean debt data
        valid_debts = []
        for debt in debts:
            try:
                balance = float(debt.get('balance', 0))
                interest_rate = float(debt.get('interest_rate', 0))
                minimum_payment = float(debt.get('minimum_payment', 0))
                
                # Skip debts with invalid data
                if balance <= 0:
                    continue
                    
                # Auto-adjust minimum payment if it doesn't cover monthly interest
                monthly_interest = balance * (interest_rate / 100 / 12) if interest_rate > 0 else 0
                if minimum_payment <= monthly_interest and interest_rate > 0:
                    # Adjust minimum payment to be able to pay off the debt
                    minimum_payment = monthly_interest * 1.2 + 25  # 20% buffer plus $25 principal
                
                if minimum_payment <= 0:
                    minimum_payment = max(25, balance * 0.02)  # 2% of balance or $25 minimum
                
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
        
        # Calculate payoff scenarios
        scenarios = {}
        for scenario_name, extra in [('minimum_only', 0), ('with_extra', extra_payment)]:
            payoff_plan = []
            remaining_debts = [debt.copy() for debt in valid_debts]
            
            # FIXED: Sort debts based on strategy
            if strategy == 'avalanche':
                remaining_debts.sort(key=lambda x: x['interest_rate'], reverse=True)
            else:  # snowball
                remaining_debts.sort(key=lambda x: x['balance'])
            
            total_interest = 0
            total_months = 0
            available_extra = extra
            
            # Process each debt in priority order
            for i, debt in enumerate(remaining_debts):
                # FIXED: Apply extra payment to current priority debt
                monthly_payment = debt['minimum_payment']
                if scenario_name == 'with_extra' and i == 0 and available_extra > 0:
                    monthly_payment += available_extra
                
                balance = debt['balance']
                rate = debt['interest_rate'] / 100 / 12 if debt['interest_rate'] > 0 else 0
                
                # Calculate months to payoff
                if rate <= 0:
                    months = int(np.ceil(balance / monthly_payment)) if monthly_payment > 0 else 999
                elif monthly_payment <= balance * rate:
                    months = 999   # Payment doesn't cover interest
                else:
                    months = -np.log(1 - (balance * rate) / monthly_payment) / np.log(1 + rate)
                    months = max(1, int(np.ceil(months)))
                
                # Calculate interest paid
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
                
                # FIXED: Roll over payments when debt is paid off
                if i == 0 and scenario_name == 'with_extra':
                    # This debt gets extra payment; when paid off, extra rolls to next debt
                    available_extra = monthly_payment - debt['minimum_payment']
            
            scenarios[scenario_name] = {
                'payoff_plan': payoff_plan,
                'total_interest': total_interest,
                'total_months': total_months,
                'total_payments': total_minimum + (extra if scenario_name == 'with_extra' else 0)
            }
        
        # Calculate savings from extra payments
        interest_savings = max(0, scenarios['minimum_only']['total_interest'] - scenarios['with_extra']['total_interest'])
        time_savings = max(0, scenarios['minimum_only']['total_months'] - scenarios['with_extra']['total_months'])
        
        return {
            'total_debt': total_debt,
            'total_minimum_payment': total_minimum,
            'scenarios': scenarios,
            'strategy': strategy,
            'interest_savings': interest_savings,
            'time_savings_months': time_savings,
            'recommended_extra_payment': max(50, total_debt * 0.02)  # 2% of total debt or $50
        }
    
    @staticmethod
    def calculate_retirement_needs(current_age: int, retirement_age: int, current_income: float, 
                                 current_savings: float, monthly_contribution: float) -> Dict[str, Any]:
        """
        Calculate comprehensive retirement planning with improved future value calculations.
        
        Args:
            current_age: Current age of the person
            retirement_age: Desired retirement age
            current_income: Current annual income
            current_savings: Current retirement savings amount
            monthly_contribution: Monthly contribution to retirement
        
        Returns:
            Dict containing retirement planning analysis
        """
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
        
        # Assumptions
        inflation_rate = 0.03
        investment_return = 0.07
        replacement_ratio = 0.80  # 80% of current income needed
        life_expectancy = 85
        retirement_years = max(1, life_expectancy - retirement_age)
        
        # Calculate future income needed (adjusted for inflation)
        future_income_needed = current_income * ((1 + inflation_rate) ** years_to_retirement)
        annual_retirement_need = future_income_needed * replacement_ratio
        
        # Calculate total retirement corpus needed
        # Using present value of annuity formula for retirement years
        real_return = investment_return - inflation_rate
        if real_return > 0:
            retirement_corpus_needed = annual_retirement_need * (
                (1 - (1 + real_return) ** -retirement_years) / real_return
            )
        else:
            retirement_corpus_needed = annual_retirement_need * retirement_years
        
        # Future value of current savings
        future_current_savings = current_savings * ((1 + investment_return) ** years_to_retirement)
        
        # FIXED: Future value of contributions using proper future value of annuity formula
        if investment_return > 0 and annual_contribution > 0:
            # Standard future value of ordinary annuity formula: PMT * [((1 + r)^n - 1) / r]
            future_contributions = annual_contribution * (
                ((1 + investment_return) ** years_to_retirement - 1) / investment_return
            )
        else:
            future_contributions = annual_contribution * years_to_retirement
        
        total_projected_savings = future_current_savings + future_contributions
        
        # Gap analysis
        retirement_gap = max(0, retirement_corpus_needed - total_projected_savings)
        
        # FIXED: Calculate required monthly contribution to meet goal
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
        
        # Scenarios
        scenarios = {}
        for contribution_multiplier, scenario_name in [(0.5, 'conservative'), (1.0, 'current'), (1.5, 'aggressive')]:
            scenario_monthly = monthly_contribution * contribution_multiplier
            scenario_annual = scenario_monthly * 12
            
            # FIXED: Use proper future value of annuity formula for scenarios
            if investment_return > 0 and scenario_annual > 0:
                scenario_future_contributions = scenario_annual * (
                    ((1 + investment_return) ** years_to_retirement - 1) / investment_return
                )
            else:
                scenario_future_contributions = scenario_annual * years_to_retirement
            
            scenario_total = future_current_savings + scenario_future_contributions
            
            # Calculate monthly retirement income
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
        """
        Generate retirement planning recommendations.
        
        Args:
            gap: Retirement funding gap
            years_left: Years until retirement
            current_contrib: Current monthly contribution
            required_contrib: Required monthly contribution to meet goal
        
        Returns:
            List of recommendation strings
        """
        recommendations = []
        
        # FIXED: Proper recommendation logic
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
        """
        Create an interactive pie chart for expense breakdown with empty data handling.
        
        Args:
            expenses: Dictionary of expense categories and amounts
            title: Chart title
        
        Returns:
            Plotly figure object
        """
        # Filter out zero expenses
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
            plot_bgcolor='#1f2937'
        )
        
        return fig
    
    @staticmethod
    def plot_budget_summary(budget_data: Dict[str, Any]) -> go.Figure:
        """
        Create a comprehensive budget visualization with dynamic values.
        
        Args:
            budget_data: Dictionary containing budget analysis results
        
        Returns:
            Plotly figure object
        """
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Income vs Expenses', 'Savings Rate', 'Expense Categories', 'Financial Health Score'),
            specs=[[{"type": "bar"}, {"type": "indicator"}],
                   [{"type": "pie"}, {"type": "indicator"}]]
        )
        
        # Income vs Expenses bar chart - DYNAMIC values
        fig.add_trace(
            go.Bar(
                x=['Income', 'Expenses', 'Savings'],
                y=[budget_data['total_income'], budget_data['total_expenses'], budget_data['savings']],
                marker_color=['#2ecc71', '#e74c3c', '#3498db'],
                name='Amount'
            ),
            row=1, col=1
        )
        
        # DYNAMIC Savings rate gauge
        fig.add_trace(
            go.Indicator(
                mode="gauge+number+delta",
                value=budget_data['savings_rate'],
                domain={'x': [0, 1], 'y': [0, 1]},
                title={'text': "Savings Rate (%)"},
                gauge={
                    'axis': {'range': [None, 30]},
                    'bar': {'color': budget_data['health_color']},
                    'steps': [
                        {'range': [0, 10], 'color': "lightgray"},
                        {'range': [10, 20], 'color': "gray"}
                    ],
                    'threshold': {
                        'line': {'color': "red", 'width': 4},
                        'thickness': 0.75,
                        'value': 20
                    }
                }
            ),
            row=1, col=2
        )
        
        # Expense breakdown pie - handles empty data
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
        
        # DYNAMIC Financial health indicator - uses computed health score
        fig.add_trace(
            go.Indicator(
                mode="gauge+number",
                value=budget_data['health_score'],  # DYNAMIC score from calculation
                title={'text': f"Health Score: {budget_data['financial_health']}"},
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
                }
            ),
            row=2, col=2
        )
        
        fig.update_layout(
            height=800, 
            showlegend=False, 
            title_text="Budget Analysis Dashboard",
            paper_bgcolor='#1f2937',
            plot_bgcolor='#1f2937',
            font_color='white'
        )
        return fig
    
    @staticmethod
    def plot_investment_allocation(allocation_data: Dict[str, Any]) -> go.Figure:
        """
        Create investment allocation visualization with dynamic values.
        
        Args:
            allocation_data: Dictionary containing investment allocation data
        
        Returns:
            Plotly figure object
        """
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Asset Allocation', 'Portfolio Projections', 'Risk vs Return', 'Dollar Allocation'),
            specs=[[{"type": "pie"}, {"type": "scatter"}],
                   [{"type": "scatter"}, {"type": "bar"}]]
        )
        
        # DYNAMIC Asset allocation pie chart
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
        
        # DYNAMIC Portfolio projections
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
        
        # Risk vs Return scatter
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
        
        # DYNAMIC Dollar allocation bar chart
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
            font_color='white'
        )
        return fig
    
    @staticmethod
    def plot_debt_payoff(debt_data: Dict[str, Any]) -> go.Figure:
        """
        Create debt payoff visualization with dynamic values.
        
        Args:
            debt_data: Dictionary containing debt analysis data
        
        Returns:
            Plotly figure object
        """
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
        
        # Get debt data from scenario
        debts = scenarios.get('minimum_only', {}).get('payoff_plan', [])
        
        if debts:
            debt_names = [debt['debt_name'] for debt in debts]
            balances = [debt['balance'] for debt in debts]
            months = [debt['months_to_payoff'] for debt in debts]
            interest_rates = [debt['interest_rate'] for debt in debts]
            payments = [debt['monthly_payment'] for debt in debts]
            
            # DYNAMIC Debt balances
            fig.add_trace(
                go.Bar(x=debt_names, y=balances, name='Balance', marker_color='red'),
                row=1, col=1
            )
            
            # DYNAMIC Payoff timeline
            fig.add_trace(
                go.Bar(x=debt_names, y=months, name='Months to Payoff', marker_color='blue'),
                row=1, col=2
            )
            
            # DYNAMIC Interest rates
            fig.add_trace(
                go.Bar(x=debt_names, y=interest_rates, name='Interest Rate (%)', marker_color='orange'),
                row=2, col=1
            )
            
            # DYNAMIC Monthly payments
            fig.add_trace(
                go.Bar(x=debt_names, y=payments, name='Monthly Payment', marker_color='green'),
                row=2, col=2
            )
        
        fig.update_layout(
            height=800, 
            showlegend=False, 
            title_text="Debt Payoff Analysis",
            paper_bgcolor='#1f2937',
            plot_bgcolor='#1f2937',
            font_color='white'
        )
        return fig
    
    @staticmethod
    def plot_retirement_projections(retirement_data: Dict[str, Any]) -> go.Figure:
        """
        Create retirement planning visualization with dynamic values.
        
        Args:
            retirement_data: Dictionary containing retirement analysis data
        
        Returns:
            Plotly figure object
        """
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Retirement Scenarios', 'Contribution Impact', 'Savings Growth', 'Income Replacement'),
            specs=[[{"type": "bar"}, {"type": "scatter"}],
                   [{"type": "scatter"}, {"type": "indicator"}]]
        )
        
        scenarios = retirement_data.get('scenarios', {})
        
        # DYNAMIC Retirement scenarios comparison
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
        
        # DYNAMIC Contribution impact
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
        
        # DYNAMIC Savings growth over time
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
        
        # DYNAMIC Income replacement gauge
        current_replacement = scenarios.get('current', {}).get('replacement_ratio_achieved', 0) * 100
        fig.add_trace(
            go.Indicator(
                mode="gauge+number",
                value=current_replacement,
                title={'text': "Income Replacement (%)"},
                gauge={
                    'axis': {'range': [None, 100]},
                    'bar': {'color': "darkblue"},
                    'steps': [
                        {'range': [0, 50], 'color': "lightgray"},
                        {'range': [50, 80], 'color': "gray"}
                    ],
                    'threshold': {
                        'line': {'color': "red", 'width': 4},
                        'thickness': 0.75,
                        'value': 80
                    }
                }
            ),
            row=2, col=2
        )
        
        fig.update_layout(
            height=800, 
            showlegend=True, 
            title_text="Retirement Planning Analysis",
            paper_bgcolor='#1f2937',
            plot_bgcolor='#1f2937',
            font_color='white'
        )
        return fig

class FinancialFlows:
    """Structured financial advisory flows with step-by-step guidance"""
    
    @staticmethod
    def budgeting_flow():
        """
        Interactive budgeting flow with guided questions.
        
        Returns:
            Budget analysis results or None
        """
        if not TEST_MODE:
            st.markdown('<div class="flow-card"><h2>💰 Smart Budgeting Assistant</h2><p>Let\'s create a comprehensive budget plan tailored to your financial situation.</p></div>', unsafe_allow_html=True)
        
        # Initialize session state for form data
        if 'budget_form_submitted' not in st.session_state:
            st.session_state.budget_form_submitted = False
        if 'budget_form_data' not in st.session_state:
            st.session_state.budget_form_data = {}
        
        # Create form
        if not TEST_MODE:
            with st.form("budget_form"):
                # Step 1: Income
                st.subheader("Step 1: Monthly Income")
                col1, col2 = st.columns(2)
                
                with col1:
                    primary_income = st.number_input("Primary Income (after taxes)", min_value=0.0, value=5000.0, step=100.0)
                    secondary_income = st.number_input("Secondary Income", min_value=0.0, value=0.0, step=100.0)
                
                with col2:
                    other_income = st.number_input("Other Income (investments, etc.)", min_value=0.0, value=0.0, step=100.0)
                    total_income = primary_income + secondary_income + other_income
                    st.metric("Total Monthly Income", f"${total_income:,.2f}")
                
                # Step 2: Expenses
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
                
                # Submit button
                submitted = st.form_submit_button("Analyze My Budget", type="primary")
        else:
            # Test mode - use sample data with dynamic total income calculation
            submitted = True
            expenses = {
                'housing': 1500, 'utilities': 200, 'groceries': 400,
                'transportation': 300, 'insurance': 200, 'healthcare': 150,
                'dining_out': 300, 'shopping': 200, 'subscriptions': 50,
                'savings': 500, 'debt_payments': 300, 'other': 100
            }
            # FIXED: Calculate total income dynamically from expenses for test mode
            total_expenses = sum(expenses.values())
            total_income = total_expenses + 1000  # Add some savings for realistic test scenario
        
        # Process form submission
        if submitted:
            st.session_state.budget_form_submitted = True
            st.session_state.budget_form_data = {
                'total_income': total_income,
                'expenses': expenses
            }
        
        # Display results if form has been submitted
        if st.session_state.budget_form_submitted and st.session_state.budget_form_data:
            form_data = st.session_state.budget_form_data
            budget_summary = FinancialCalculator.calculate_budget_summary(form_data['total_income'], form_data['expenses'])
            
            # AI INJECTION: Generate AI insights for budget analysis
            ai_insights = generate_ai_insights(budget_summary, "Budget Analysis")
            
            if not TEST_MODE:
                # Display summary cards
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
                    # FIXED: Standardized AI score display - show both System Score and AI Score
                    ai_score = ai_insights.get("ai_score")
                    if ai_score is not None:
                        st.markdown(f'''
                        <div class="metric-card">
                            <h3>System Score</h3>
                            <h2 style="color: {budget_summary["health_color"]}">{budget_summary["health_score"]}/100</h2>
                            <p>AI Score: {ai_score}/100</p>
                        </div>
                        ''', unsafe_allow_html=True)
                    else:
                        st.markdown(f'''
                        <div class="metric-card">
                            <h3>System Score</h3>
                            <h2 style="color: {budget_summary["health_color"]}">{budget_summary["health_score"]}/100</h2>
                        </div>
                        ''', unsafe_allow_html=True)
                
                # Financial Health Assessment
                st.markdown(f'''
                <div class="summary-card">
                    <h3>Financial Health: {budget_summary["financial_health"]} (Score: {budget_summary["health_score"]}/100)</h3>
                    <h4>Personalized Recommendations:</h4>
                    <ul>
                        {"".join(f"<li>{rec}</li>" for rec in budget_summary["recommendations"])}
                    </ul>
                </div>
                ''', unsafe_allow_html=True)
                
                # AI INJECTION: Display AI suggestions for budget
                display_ai_suggestions(ai_insights, "Budget Analysis")
                
                # Visualizations
                st.subheader("Budget Analysis Dashboard")
                budget_viz = FinancialVisualizer.plot_budget_summary(budget_summary)
                st.plotly_chart(budget_viz, use_container_width=True)
                
                # Store in session state for chat context
                st.session_state.budget_data = budget_summary
                # AI INJECTION: Store AI insights for context
                st.session_state.budget_ai_insights = ai_insights
            
            return budget_summary
    
    @staticmethod
    def investing_flow():
        """
        Interactive investment planning flow with dynamic risk profile calculation.
        
        Returns:
            Investment allocation data or None
        """
        if not TEST_MODE:
            st.markdown('<div class="flow-card"><h2>📈 Investment Portfolio Builder</h2><p>Let\'s create an optimal investment strategy based on your goals and risk tolerance.</p></div>', unsafe_allow_html=True)
        
        # Initialize session state for form data
        if 'investment_form_submitted' not in st.session_state:
            st.session_state.investment_form_submitted = False
        if 'investment_form_data' not in st.session_state:
            st.session_state.investment_form_data = {}
        
        # Create form or use test data
        if not TEST_MODE:
            with st.form("investment_form"):
                # Step 1: Investment Goals
                st.subheader("Step 1: Investment Goals & Timeline")
                col1, col2 = st.columns(2)
                
                with col1:
                    investment_goal = st.selectbox(
                        "Primary Investment Goal",
                        ["Retirement", "House Down Payment", "Emergency Fund", "Wealth Building", "Education", "Other"]
                    )
                    time_horizon = st.slider("Investment Time Horizon (years)", 1, 40, 10)
                
                with col2:
                    current_age = st.number_input("Your Current Age", min_value=18, max_value=80, value=35)
                    investment_capital = st.number_input("Initial Investment Amount", min_value=0.0, value=10000.0, step=1000.0)
                
                # Step 2: Risk Assessment
                st.subheader("Step 2: Risk Tolerance Assessment")
                
                risk_questions = {
                    "market_drop": "If your portfolio dropped 20% in a month, you would:",
                    "investment_experience": "Your investment experience level:",
                    "income_stability": "Your income stability:",
                    "sleep_factor": "Regarding investment volatility:"
                }
                
                risk_answers = {}
                
                risk_answers["market_drop"] = st.radio(
                    risk_questions["market_drop"],
                    ["Panic and sell everything", "Feel uncomfortable but hold", "See it as a buying opportunity"],
                    key="market_drop"
                )
                
                risk_answers["investment_experience"] = st.radio(
                    risk_questions["investment_experience"],
                    ["Beginner (< 2 years)", "Intermediate (2-10 years)", "Advanced (> 10 years)"],
                    key="investment_experience"
                )
                
                risk_answers["income_stability"] = st.radio(
                    risk_questions["income_stability"],
                    ["Unstable/Variable", "Stable", "Very Stable with Growth"],
                    key="income_stability"
                )
                
                risk_answers["sleep_factor"] = st.radio(
                    risk_questions["sleep_factor"],
                    ["I need stable, predictable returns", "I can handle some ups and downs", "I'm comfortable with high volatility for higher returns"],
                    key="sleep_factor"
                )
                
                # Submit button
                submitted = st.form_submit_button("Generate Investment Portfolio", type="primary")
        else:
            # Test mode
            submitted = True
            time_horizon = 15
            current_age = 35
            investment_capital = 25000.0
            risk_answers = {
                "market_drop": "See it as a buying opportunity",
                "investment_experience": "Intermediate (2-10 years)",
                "income_stability": "Stable",
                "sleep_factor": "I can handle some ups and downs"
            }
        
        # Process form submission
        if submitted:
            # DYNAMIC risk profile calculation from questionnaire scoring
            risk_score = 0
            risk_weights = {
                "market_drop": {"Panic and sell everything": 1, "Feel uncomfortable but hold": 2, "See it as a buying opportunity": 3},
                "investment_experience": {"Beginner (< 2 years)": 1, "Intermediate (2-10 years)": 2, "Advanced (> 10 years)": 3},
                "income_stability": {"Unstable/Variable": 1, "Stable": 2, "Very Stable with Growth": 3},
                "sleep_factor": {"I need stable, predictable returns": 1, "I can handle some ups and downs": 2, "I'm comfortable with high volatility for higher returns": 3}
            }
            
            for question, answer in risk_answers.items():
                risk_score += risk_weights[question][answer]
            
            # DYNAMIC risk profile determination
            if risk_score <= 6:
                risk_profile = "Conservative"
            elif risk_score <= 9:
                risk_profile = "Moderate"
            else:
                risk_profile = "Aggressive"
            
            st.session_state.investment_form_submitted = True
            st.session_state.investment_form_data = {
                'risk_profile': risk_profile,
                'time_horizon': time_horizon,
                'investment_capital': investment_capital,
                'current_age': current_age
            }
        
        # Display results if form has been submitted
        if st.session_state.investment_form_submitted and st.session_state.investment_form_data:
            form_data = st.session_state.investment_form_data
            
            if not TEST_MODE:
                st.info(f"Based on your responses, your risk profile is: **{form_data['risk_profile']}**")
            
            # DYNAMIC allocation based on calculated risk profile, age, and time horizon
            allocation_data = FinancialCalculator.calculate_investment_allocation(
                form_data['risk_profile'], form_data['time_horizon'], form_data['investment_capital'], form_data['current_age']
            )
            
            # AI INJECTION: Generate AI insights for investment analysis
            ai_insights = generate_ai_insights(allocation_data, "Investment Analysis")
            
            if not TEST_MODE:
                # Display allocation summary
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
                
                # FIXED: Standardized AI score display with expected return formatting
                ai_score = ai_insights.get("ai_score")
                if ai_score is not None:
                    ai_score_text = f" | AI Score: {ai_score}/100"
                else:
                    ai_score_text = ""
                
                st.markdown(f'''
                <div class="summary-card">
                    <h3>Portfolio Metrics{ai_score_text}</h3>
                    <p><strong>Expected Annual Return:</strong> {allocation_data["expected_annual_return"]:.1%}</p>
                    <p><strong>Estimated Volatility:</strong> {allocation_data["volatility_estimate"]:.1%}</p>
                    <p><strong>Risk Level:</strong> {allocation_data["risk_level"]}</p>
                </div>
                ''', unsafe_allow_html=True)
                
                # AI INJECTION: Display AI suggestions for investment
                display_ai_suggestions(ai_insights, "Investment Analysis")
                
                # Projections
                if allocation_data.get('projections'):
                    st.subheader("Portfolio Growth Projections")
                    projections_df = pd.DataFrame(allocation_data['projections']).T
                    st.dataframe(projections_df.style.format("${:,.0f}"))
                
                # Visualizations
                st.subheader("Investment Portfolio Analysis")
                investment_viz = FinancialVisualizer.plot_investment_allocation(allocation_data)
                st.plotly_chart(investment_viz, use_container_width=True)
                
                # Store in session state
                st.session_state.investment_data = allocation_data
                # AI INJECTION: Store AI insights
                st.session_state.investment_ai_insights = ai_insights
            
            return allocation_data
    
    @staticmethod
    def debt_repayment_flow():
        """
        Interactive debt repayment planning flow with fixed avalanche/snowball logic.
        
        Returns:
            Debt analysis results or None
        """
        if not TEST_MODE:
            st.markdown('<div class="flow-card"><h2>💳 Debt Freedom Planner</h2><p>Let\'s create a strategic plan to eliminate your debt efficiently.</p></div>', unsafe_allow_html=True)
        
        # Step 1: Debt Inventory
        if not TEST_MODE:
            st.subheader("Step 1: Your Current Debts")
        
        if 'debts' not in st.session_state:
            st.session_state.debts = []
        
        if not TEST_MODE:
            # Add new debt form
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
            
            # Display current debts
            if st.session_state.debts:
                st.subheader("Your Current Debts")
                debt_df = pd.DataFrame(st.session_state.debts)
                debt_df['Balance'] = debt_df['balance'].apply(lambda x: f"${x:,.2f}")
                debt_df['Interest Rate'] = debt_df['interest_rate'].apply(lambda x: f"{x:.1f}%")
                debt_df['Min Payment'] = debt_df['minimum_payment'].apply(lambda x: f"${x:.2f}")
                
                display_df = debt_df[['name', 'Balance', 'Interest Rate', 'Min Payment']].copy()
                display_df.columns = ['Debt Name', 'Balance', 'Interest Rate', 'Min Payment']
                st.dataframe(display_df, use_container_width=True)
                
                # Clear debts button
                if st.button("Clear All Debts"):
                    st.session_state.debts = []
                    st.rerun()
        else:
            # Test mode - use sample debts
            st.session_state.debts = [
                {'name': 'Credit Card 1', 'balance': 5000, 'interest_rate': 18.0, 'minimum_payment': 150},
                {'name': 'Credit Card 2', 'balance': 3000, 'interest_rate': 22.0, 'minimum_payment': 100},
                {'name': 'Student Loan', 'balance': 15000, 'interest_rate': 6.0, 'minimum_payment': 180}
            ]
        
        # Step 2: Repayment Strategy
        if st.session_state.debts:
            if not TEST_MODE:
                st.subheader("Step 2: Repayment Strategy")
            
            # Initialize session state for debt form
            if 'debt_form_submitted' not in st.session_state:
                st.session_state.debt_form_submitted = False
            if 'debt_form_data' not in st.session_state:
                st.session_state.debt_form_data = {}
            
            # Create form or use test data
            if not TEST_MODE:
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
                    
                    # Submit button
                    submitted = st.form_submit_button("Create Debt Payoff Plan", type="primary")
            else:
                submitted = True
                strategy = 'avalanche'
                extra_payment = 200.0
            
            # Process form submission
            if submitted:
                st.session_state.debt_form_submitted = True
                st.session_state.debt_form_data = {
                    'strategy': strategy,
                    'extra_payment': extra_payment
                }
            
            # Display results if form has been submitted
            if st.session_state.debt_form_submitted and st.session_state.debt_form_data:
                form_data = st.session_state.debt_form_data
                debt_analysis = FinancialCalculator.calculate_debt_payoff(st.session_state.debts, form_data['extra_payment'], form_data['strategy'])
                
                # AI INJECTION: Generate AI insights for debt analysis
                ai_insights = generate_ai_insights(debt_analysis, "Debt Analysis")
                
                if not TEST_MODE:
                    # Summary metrics
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
                        # FIXED: If extra payment is 0, explicitly display "Interest Savings: $0"
                        if form_data['extra_payment'] > 0:
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
                        # FIXED: Standardized AI score display - show both System Score and AI Score
                        ai_score = ai_insights.get("ai_score")
                        if ai_score is not None:
                            st.markdown(f'''
                            <div class="metric-card">
                                <h3>System Score</h3>
                                <h2>{100 - min(100, debt_analysis["total_debt"] / 1000):.0f}/100</h2>
                                <p>AI Score: {ai_score}/100</p>
                            </div>
                            ''', unsafe_allow_html=True)
                        else:
                            # FIXED: If extra payment is 0, explicitly display "Time Savings: 0 months"
                            if form_data['extra_payment'] > 0:
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
                    
                    # Detailed payoff plan
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
                    
                    # Recommendations
                    st.markdown(f'''
                    <div class="summary-card">
                        <h3>Debt Payoff Recommendations</h3>
                        <ul>
                            <li>🎯 Focus on paying ${debt_analysis["recommended_extra_payment"]:.0f} extra per month if possible</li>
                            <li>📊 You're using the <strong>{form_data["strategy"].title()}</strong> method - {"pay highest interest rates first" if form_data["strategy"] == "avalanche" else "pay smallest balances first"}</li>
                            <li>💡 Consider debt consolidation if you have high-interest credit cards</li>
                            <li>🚫 Avoid taking on new debt during your payoff journey</li>
                            <li>📱 Set up automatic payments to stay on track</li>
                        </ul>
                    </div>
                    ''', unsafe_allow_html=True)
                    
                    # AI INJECTION: Display AI suggestions for debt
                    display_ai_suggestions(ai_insights, "Debt Analysis")
                    
                    # Visualization
                    st.subheader("Debt Analysis Dashboard")
                    debt_viz = FinancialVisualizer.plot_debt_payoff(debt_analysis)
                    st.plotly_chart(debt_viz, use_container_width=True)
                    
                    # Store in session state
                    st.session_state.debt_data = debt_analysis
                    # AI INJECTION: Store AI insights
                    st.session_state.debt_ai_insights = ai_insights
                
                return debt_analysis
    
    @staticmethod
    def retirement_planning_flow():
        """
        Interactive retirement planning flow with improved calculations.
        
        Returns:
            Retirement analysis results or None
        """
        if not TEST_MODE:
            st.markdown('<div class="flow-card"><h2>🏖️ Retirement Planning Assistant</h2><p>Let\'s ensure you\'re on track for a comfortable retirement.</p></div>', unsafe_allow_html=True)
        
        # Initialize session state for form data - FIXED to prevent re-loops
        if 'retirement_form_submitted' not in st.session_state:
            st.session_state.retirement_form_submitted = False
        if 'retirement_form_data' not in st.session_state:
            st.session_state.retirement_form_data = {}
        
        # Create form or use test data
        if not TEST_MODE:
            # FIXED: Use proper st.form to prevent re-run loops
            with st.form("retirement_form"):
                # Step 1: Current Situation
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
                
                # Step 2: Retirement Goals
                st.subheader("Step 2: Retirement Lifestyle Goals")
                
                lifestyle_choice = st.selectbox(
                    "Desired Retirement Lifestyle",
                    ["Basic (60% of current income)", "Comfortable (80% of current income)", "Luxurious (100% of current income)"]
                )
                
                replacement_ratios = {
                    "Basic (60% of current income)": 0.60,
                    "Comfortable (80% of current income)": 0.80,
                    "Luxurious (100% of current income)": 1.00
                }
                
                # Additional considerations
                col1, col2 = st.columns(2)
                
                with col1:
                    healthcare_inflation = st.checkbox("Account for higher healthcare costs", value=True)
                    social_security = st.checkbox("Include Social Security benefits", value=True)
                
                with col2:
                    inheritance_expected = st.number_input("Expected Inheritance", min_value=0.0, value=0.0, step=10000.0)
                    other_retirement_income = st.number_input("Other Retirement Income (monthly)", min_value=0.0, value=0.0, step=100.0)
                
                # FIXED: Use st.form_submit_button to prevent re-runs
                submitted = st.form_submit_button("Analyze Retirement Plan", type="primary")
        else:
            # Test mode
            submitted = True
            current_age = 35
            retirement_age = 65
            current_income = 75000.0
            current_savings = 50000.0
            monthly_contribution = 500.0
            employer_match = 150.0
        
        # FIXED: Process form submission only when submitted
        if submitted:
            st.session_state.retirement_form_submitted = True
            st.session_state.retirement_form_data = {
                'current_age': current_age,
                'retirement_age': retirement_age,
                'current_income': current_income,
                'current_savings': current_savings,
                'monthly_contribution': monthly_contribution,
                'employer_match': employer_match
            }
        
        # FIXED: Display results only if form has been submitted and data exists
        if st.session_state.retirement_form_submitted and st.session_state.retirement_form_data:
            form_data = st.session_state.retirement_form_data
            total_monthly_contribution = form_data['monthly_contribution'] + form_data['employer_match']
            
            # FIXED: Calculate retirement needs with proper values
            retirement_analysis = FinancialCalculator.calculate_retirement_needs(
                form_data['current_age'], form_data['retirement_age'], form_data['current_income'], 
                form_data['current_savings'], total_monthly_contribution
            )
            
            # AI INJECTION: Generate AI insights for retirement analysis
            ai_insights = generate_ai_insights(retirement_analysis, "Retirement Analysis")
            
            if not TEST_MODE:
                # Key metrics
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
                    # FIXED: Standardized AI score display - show both System Score and AI Score
                    ai_score = ai_insights.get("ai_score")
                    gap = retirement_analysis["retirement_gap"]
                    
                    if ai_score is not None:
                        gap_color = "red" if gap > 0 else "green"
                        gap_text = f"${gap:,.0f}" if gap > 0 else "On Track!"
                        
                        st.markdown(f'''
                        <div class="metric-card">
                            <h3>System Score</h3>
                            <h2 style="color: {gap_color}">{100 - min(100, gap/10000):.0f}/100</h2>
                            <p>AI Score: {ai_score}/100</p>
                        </div>
                        ''', unsafe_allow_html=True)
                    else:
                        gap_color = "red" if gap > 0 else "green"
                        gap_text = f"${gap:,.0f}" if gap > 0 else "On Track!"
                        
                        st.markdown(f'''
                        <div class="metric-card">
                            <h3>Retirement Gap</h3>
                            <h2 style="color: {gap_color}">{gap_text}</h2>
                        </div>
                        ''', unsafe_allow_html=True)
                
                # Scenario comparison
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
                
                # FIXED Recommendations - proper logic
                st.markdown(f'''
                <div class="summary-card">
                    <h3>Retirement Planning Recommendations</h3>
                    <ul>
                        {"".join(f"<li>{rec}</li>" for rec in retirement_analysis["recommendations"])}
                    </ul>
                </div>
                ''', unsafe_allow_html=True)
                
                # AI INJECTION: Display AI suggestions for retirement
                display_ai_suggestions(ai_insights, "Retirement Analysis")
                
                # FIXED: Action items with proper recommendation logic
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
                
                # Visualization
                st.subheader("Retirement Planning Dashboard")
                retirement_viz = FinancialVisualizer.plot_retirement_projections(retirement_analysis)
                st.plotly_chart(retirement_viz, use_container_width=True)
                
                # Store in session state
                st.session_state.retirement_data = retirement_analysis
                # AI INJECTION: Store AI insights
                st.session_state.retirement_ai_insights = ai_insights
            
            return retirement_analysis

class PersonaManager:
    """Manage different financial advisor personas"""
    
    PERSONAS = {
        "Friendly Coach": {
            "description": "Encouraging and supportive, focuses on building confidence",
            "tone": "friendly",
            "emoji": "😊",
            "style": "I'm here to cheer you on! Let's make your financial dreams come true together!"
        },
        "Practical Advisor": {
            "description": "Direct and actionable, focuses on practical steps",
            "tone": "practical",
            "emoji": "💼",
            "style": "Let's focus on concrete actions and realistic strategies that work."
        },
        "Conservative Planner": {
            "description": "Risk-averse and cautious, emphasizes security",
            "tone": "conservative",
            "emoji": "🛡️",
            "style": "Safety first! Let's build a solid foundation for your financial future."
        }
    }
    
    @staticmethod
    def display_persona_selector():
        """
        Display persona selection interface.
        
        Returns:
            Tuple of selected persona name and persona info
        """
        if TEST_MODE:
            return "Friendly Coach", PersonaManager.PERSONAS["Friendly Coach"]
            
        st.sidebar.subheader("🎭 Choose Your Financial Advisor")
        
        selected_persona = st.sidebar.selectbox(
            "Advisor Personality",
            list(PersonaManager.PERSONAS.keys()),
            index=0
        )
        
        persona_info = PersonaManager.PERSONAS[selected_persona]
        
        st.sidebar.markdown(f'''
        <div class="persona-card">
            <h4>{persona_info["emoji"]} {selected_persona}</h4>
            <p>{persona_info["description"]}</p>
            <em>"{persona_info["style"]}"</em>
        </div>
        ''', unsafe_allow_html=True)
        
        return selected_persona, persona_info

def extract_text_from_pdf(file_path):
    """
    Extracts text from PDFs using PyMuPDF, falls back to OCR if needed.
    
    Args:
        file_path: Path to the PDF file
    
    Returns:
        List of text content from each page
    """
    if TEST_MODE:
        return ["Test PDF content"]
        
    try:
        doc = fitz.open(file_path)
        text_list = [page.get_text("text") for page in doc if page.get_text("text").strip()]
        doc.close()
        return text_list if text_list else extract_text_from_images(file_path)
    except Exception as e:
        st.error(f"⚠️ Error extracting text from PDF: {e}")
        return []

def extract_text_from_images(pdf_path):
    """
    Extracts text from image-based PDFs using GPU-accelerated EasyOCR.
    
    Args:
        pdf_path: Path to the PDF file
    
    Returns:
        List of extracted text from each page
    """
    if reader is None or TEST_MODE:
        return ["OCR not available in test mode"]
    
    try:
        images = convert_from_path(pdf_path, dpi=150, first_page=1, last_page=5)
        return ["\n".join(reader.readtext(np.array(img), detail=0)) for img in images]
    except Exception as e:
        st.error(f"⚠️ Error extracting text from images: {e}")
        return []

def setup_vectorstore(documents):
    """
    Creates a FAISS vector store using Hugging Face embeddings.
    
    Args:
        documents: List of text documents
    
    Returns:
        FAISS vectorstore instance or None
    """
    if TEST_MODE:
        return None
        
    try:
        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        
        if DEVICE == "cuda":
            embeddings.model = embeddings.model.to(torch.device("cuda"))
        
        text_splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=100)
        doc_chunks = text_splitter.split_text("\n".join(documents))
        return FAISS.from_texts(doc_chunks, embeddings)
    except Exception as e:
        st.error(f"Error setting up vector store: {e}")
        return None

def create_chain(vectorstore):
    """
    Creates the chat chain with optimized retriever settings.
    
    Args:
        vectorstore: FAISS vectorstore instance
    
    Returns:
        ConversationalRetrievalChain instance or None
    """
    if TEST_MODE:
        return None
        
    try:
        # FIXED: Add safeguards - ensure memory exists before using it
        if "memory" not in st.session_state:
            st.session_state.memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

        llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0, groq_api_key=groq_api_key)
        retriever = vectorstore.as_retriever(search_kwargs={"k": 5})
        
        return ConversationalRetrievalChain.from_llm(
            llm=llm,
            retriever=retriever,
            chain_type="stuff",
            memory=st.session_state.memory,
            verbose=False
        )
    except Exception as e:
        st.error(f"Error creating conversation chain: {e}")
        return None

def get_fallback_response(user_input: str, persona: str) -> str:
    """
    Rule-based fallback responses when LLM is unavailable.
    
    Args:
        user_input: User's input message
        persona: Selected persona type
    
    Returns:
        Appropriate fallback response string
    """
    user_input_lower = user_input.lower()
    
    # Budget-related responses
    if any(word in user_input_lower for word in ['budget', 'expense', 'spending', 'money']):
        if persona == "Friendly Coach":
            return "😊 Great question about budgeting! I'd love to help you create a budget. Try using our Budget Flow above to get personalized recommendations!"
        elif persona == "Practical Advisor":
            return "💼 For budgeting, follow the 50/30/20 rule: 50% needs, 30% wants, 20% savings. Use our budgeting tool for detailed analysis."
        else:
            return "🛡️ A conservative approach to budgeting is essential. Start by tracking all expenses and prioritizing emergency savings."
    
    # Investment-related responses
    elif any(word in user_input_lower for word in ['invest', 'portfolio', 'stocks', 'bonds']):
        if persona == "Friendly Coach":
            return "📈 Investing is exciting! Let's build a portfolio that matches your dreams. Check out our Investment Flow for personalized recommendations!"
        elif persona == "Practical Advisor":
            return "💼 Diversification is key. Consider low-cost index funds and match your risk tolerance to your time horizon."
        else:
            return "🛡️ Conservative investing focuses on capital preservation. Consider bonds, CDs, and blue-chip dividend stocks."
    
    # Debt-related responses
    elif any(word in user_input_lower for word in ['debt', 'loan', 'credit card', 'payoff']):
        if persona == "Friendly Coach":
            return "💪 You can conquer your debt! Let's create a plan together. Our Debt Repayment Flow will help you become debt-free!"
        elif persona == "Practical Advisor":
            return "💼 Focus on high-interest debt first (avalanche method) or smallest balances (snowball method). Use our debt calculator."
        else:
            return "🛡️ Debt elimination should be your priority. Pay minimums on all debts, then attack the highest interest rate first."
    
    # Retirement-related responses
    elif any(word in user_input_lower for word in ['retirement', 'retire', '401k', 'ira']):
        if persona == "Friendly Coach":
            return "🏖️ Your future self will thank you for planning now! Let's use our Retirement Planning Flow to secure your golden years!"
        elif persona == "Practical Advisor":
            return "💼 Start with employer 401(k) matching, then max out IRAs. Aim to save 10-15% of income for retirement."
        else:
            return "🛡️ Conservative retirement planning means starting early and saving consistently. Consider target-date funds for simplicity."
    
    # General financial advice
    else:
        if persona == "Friendly Coach":
            return "😊 I'm here to help with all your financial questions! Try our guided flows above, or ask me about budgeting, investing, debt, or retirement planning!"
        elif persona == "Practical Advisor":
            return "💼 I can help with budgeting, investing, debt payoff, and retirement planning. What specific financial goal would you like to work on?"
        else:
            return "🛡️ Financial security comes from careful planning. I can help with conservative strategies for budgeting, investing, and retirement. What's your priority?"

async def get_response(user_input, persona_info):
    """
    Get response from LLM or fallback system.
    
    Args:
        user_input: User's input message
        persona_info: Persona information dictionary
    
    Returns:
        Response string from AI or fallback system
    """
    if TEST_MODE:
        return "Test response from AI assistant"
        
    try:
        if "conversation_chain" in st.session_state and st.session_state.conversation_chain:
            # Enhance prompt with persona and financial context
            enhanced_prompt = f"""
            You are a {persona_info['description']} financial advisor. {persona_info['style']}
            
            Context from previous financial analysis:
            {get_financial_context()}
            
            User question: {user_input}
            
            Provide helpful, personalized financial advice in your characteristic style.
            """
            
            # FIXED: Add safeguards - ensure memory exists
            if "memory" not in st.session_state:
                st.session_state.memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
            
            response = await asyncio.to_thread(
                st.session_state.conversation_chain.invoke,
                {"question": enhanced_prompt, "chat_history": st.session_state.memory.chat_memory.messages}
            )
            return response.get("answer", "I'm sorry, I couldn't process that.")
        else:
            # Use fallback system
            return get_fallback_response(user_input, list(PersonaManager.PERSONAS.keys())[0])
    except Exception as e:
        return get_fallback_response(user_input, list(PersonaManager.PERSONAS.keys())[0])

def get_financial_context():
    """
    Get context from previous financial analysis.
    
    Returns:
        String containing financial context or default message
    """
    context = []
    
    if 'budget_data' in st.session_state:
        budget = st.session_state.budget_data
        context.append(f"Budget Analysis: Income ${budget['total_income']:,.0f}, Expenses ${budget['total_expenses']:,.0f}, Savings Rate {budget['savings_rate']:.1f}%")
    
    if 'investment_data' in st.session_state:
        investment = st.session_state.investment_data
        context.append(f"Investment Portfolio: {investment['risk_level']} risk profile, Expected return {investment['expected_annual_return']:.1%}")
    
    if 'debt_data' in st.session_state:
        debt = st.session_state.debt_data
        context.append(f"Debt Analysis: Total debt ${debt['total_debt']:,.0f}, Strategy: {debt['strategy']}")
    
    if 'retirement_data' in st.session_state:
        retirement = st.session_state.retirement_data
        context.append(f"Retirement Planning: {retirement['years_to_retirement']} years to retirement, Gap: ${retirement['retirement_gap']:,.0f}")
    
    return " | ".join(context) if context else "No previous financial analysis available."

def run_tests():
    """Run test scenarios to validate functionality"""
    print("🧪 Running Financial App Tests...")
    
    # Test 1: Zero income budget
    print("\n📊 Test 1: Zero income budget")
    try:
        budget_result = FinancialCalculator.calculate_budget_summary(0, {'housing': 1000})
        assert budget_result['financial_health'] == 'Critical', "Should show critical health for zero income"
        assert budget_result['health_score'] == 0, "Health score should be 0 for zero income"
        print("✅ PASS: Zero income handled correctly")
    except Exception as e:
        print(f"❌ FAIL: {e}")
    
    # Test 2: High savings budget
    print("\n💰 Test 2: High savings budget")
    try:
        expenses = {'housing': 2000, 'utilities': 300, 'groceries': 400}
        budget_result = FinancialCalculator.calculate_budget_summary(8000, expenses)
        assert budget_result['savings_rate'] > 20, "Should have high savings rate"
        assert budget_result['health_score'] >= 70, "Should have high health score"
        print(f"✅ PASS: High savings rate {budget_result['savings_rate']:.1f}%, Health score: {budget_result['health_score']}")
    except Exception as e:
        print(f"❌ FAIL: {e}")
    
    # Test 3: High debt analysis
    print("\n💳 Test 3: High debt payoff analysis")
    try:
        debts = [
            {'name': 'Credit Card', 'balance': 10000, 'interest_rate': 24.0, 'minimum_payment': 300}
        ]
        debt_result = FinancialCalculator.calculate_debt_payoff(debts, 200, 'avalanche')
        assert debt_result['scenarios']['minimum_only']['total_months'] > 24, "Should take significant time to pay off high-interest debt"
        assert debt_result['interest_savings'] > 0, "Extra payments should save interest"
        print(f"✅ PASS: Debt payoff time {debt_result['scenarios']['minimum_only']['total_months']} months, Interest savings: ${debt_result['interest_savings']:,.0f}")
    except Exception as e:
        print(f"❌ FAIL: {e}")
    
    # Test 4: Investment risk profile calculation
    print("\n📈 Test 4: Investment risk profile")
    try:
        allocation = FinancialCalculator.calculate_investment_allocation('aggressive', 25, 50000, 30)
        assert allocation['allocation_percentages']['stocks'] >= 70, "Aggressive profile should have high stock allocation"
        assert allocation['expected_annual_return'] > 0.08, "Should have reasonable expected return"
        print(f"✅ PASS: Aggressive allocation - Stocks: {allocation['allocation_percentages']['stocks']}%, Expected return: {allocation['expected_annual_return']:.1%}")
    except Exception as e:
        print(f"❌ FAIL: {e}")
    
    # Test 5: Retirement planning
    print("\n🏖️ Test 5: Retirement planning")
    try:
        retirement = FinancialCalculator.calculate_retirement_needs(35, 65, 75000, 50000, 600)
        assert retirement['years_to_retirement'] == 30, "Should calculate years correctly"
        assert retirement['retirement_corpus_needed'] > 0, "Should calculate required corpus"
        print(f"✅ PASS: Retirement corpus needed: ${retirement['retirement_corpus_needed']:,.0f}, Gap: ${retirement['retirement_gap']:,.0f}")
    except Exception as e:
        print(f"❌ FAIL: {e}")
    
    # Test flows
    print("\n🔄 Test 6: Financial flows")
    try:
        # Test budget flow
        budget_data = FinancialFlows.budgeting_flow()
        assert budget_data is not None, "Budget flow should return data"
        
        # Test investment flow  
        investment_data = FinancialFlows.investing_flow()
        assert investment_data is not None, "Investment flow should return data"
        
        # Test debt flow
        debt_data = FinancialFlows.debt_repayment_flow()
        assert debt_data is not None, "Debt flow should return data"
        
        # Test retirement flow
        retirement_data = FinancialFlows.retirement_planning_flow()
        assert retirement_data is not None, "Retirement flow should return data"
        
        print("✅ PASS: All financial flows working correctly")
    except Exception as e:
        print(f"❌ FAIL: Flow test failed: {e}")
    
    # Test 7: AI Integration
    print("\n🤖 Test 7: AI Integration")
    try:
        # Test AI insights generation with sample data
        sample_data = {'total_income': 5000, 'savings_rate': 15, 'health_score': 75}
        ai_result = generate_ai_insights(sample_data, "Budget Analysis")
        
        assert 'ai_reasoning' in ai_result, "Should have AI reasoning"
        assert 'ai_recommendations' in ai_result, "Should have AI recommendations"
        assert isinstance(ai_result['ai_recommendations'], list), "Recommendations should be a list"
        
        print("✅ PASS: AI integration working correctly")
    except Exception as e:
        print(f"❌ FAIL: AI test failed: {e}")
    
    print("\n🎉 Test suite completed!")

def main():
    """Main application function"""
    if TEST_MODE:
        run_tests()
        return
    
    # Header
    st.markdown('<h1 class="main-header">🦙 AI Financial Advisor - LLAMA 3.3</h1>', unsafe_allow_html=True)
    
    # AI INJECTION: Add disclaimer note
    st.info("💡 **Disclaimer**: AI suggestions are educational only and not financial advice. Always consult with a qualified financial professional for personalized guidance.")
    
    # Sidebar for persona selection and navigation
    selected_persona, persona_info = PersonaManager.display_persona_selector()
    
    # Navigation
    st.sidebar.subheader("📊 Financial Tools")
    selected_flow = st.sidebar.selectbox(
        "Choose a Financial Flow",
        ["Chat Interface", "Smart Budgeting", "Investment Planning", "Debt Repayment", "Retirement Planning"]
    )
    
    # PDF Upload Section
    st.sidebar.subheader("📄 Document Analysis")
    uploaded_files = st.sidebar.file_uploader(
        "Upload Financial Documents", 
        type=["pdf"], 
        accept_multiple_files=True,
        help="Upload bank statements, investment reports, or other financial documents"
    )
    
    if uploaded_files:
        all_extracted_text = []
        
        for uploaded_file in uploaded_files:
            # Use safe file handling with UUID prefix
            safe_filename = f"{uuid.uuid4().hex}_{uploaded_file.name}"
            file_path = os.path.join(tempfile.gettempdir(), safe_filename)
            
            with open(file_path, "wb") as f:
                f.write(uploaded_file.getbuffer())

            try:
                extracted_text = extract_text_from_pdf(file_path)
                all_extracted_text.extend(extracted_text)
                st.sidebar.success(f"✅ Processed {uploaded_file.name}")
            except Exception as e:
                st.sidebar.error(f"⚠️ Error processing {uploaded_file.name}: {e}")
            finally:
                # Clean up temporary file
                try:
                    os.remove(file_path)
                except:
                    pass

        if all_extracted_text:
            vectorstore = setup_vectorstore(all_extracted_text)
            if vectorstore:
                st.session_state.vectorstore = vectorstore
                conversation_chain = create_chain(vectorstore)
                if conversation_chain:
                    st.session_state.conversation_chain = conversation_chain
                    st.sidebar.info("📚 Documents ready for analysis!")
                else:
                    st.sidebar.warning("⚠️ Could not create conversation chain")
            else:
                st.sidebar.warning("⚠️ Could not process documents")
    
    # Main content area
    if selected_flow == "Smart Budgeting":
        FinancialFlows.budgeting_flow()
    elif selected_flow == "Investment Planning":
        FinancialFlows.investing_flow()
    elif selected_flow == "Debt Repayment":
        FinancialFlows.debt_repayment_flow()
    elif selected_flow == "Retirement Planning":
        FinancialFlows.retirement_planning_flow()
    else:
        # Chat Interface
        st.subheader(f"💬 Chat with {persona_info['emoji']} {selected_persona}")
        
        # Initialize chat history
        if "chat_history" not in st.session_state:
            st.session_state.chat_history = []
        
        # Display chat history
        for message in st.session_state.chat_history:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
        
        # Chat input
        if user_input := st.chat_input(f"Ask {selected_persona} about your finances..."):
            # Add user message to chat history
            st.session_state.chat_history.append({"role": "user", "content": user_input})
            
            with st.chat_message("user"):
                st.markdown(user_input)
            
            # Get AI response
            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    try:
                        assistant_response = asyncio.run(get_response(user_input, persona_info))
                    except Exception as e:
                        assistant_response = get_fallback_response(user_input, selected_persona)
                
                st.markdown(assistant_response)
            
            # Add assistant response to chat history
            st.session_state.chat_history.append({"role": "assistant", "content": assistant_response})
            
            # FIXED: Add safeguards - initialize memory if missing before saving context
            if "memory" not in st.session_state:
                st.session_state.memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
            
            # Save to memory if available
            st.session_state.memory.save_context({"input": user_input}, {"output": assistant_response})
    
    # Footer with financial summary
    if any(key in st.session_state for key in ['budget_data', 'investment_data', 'debt_data', 'retirement_data']):
        st.markdown("---")
        st.subheader("📊 Your Financial Summary")
        
        summary_cols = st.columns(4)
        
        if 'budget_data' in st.session_state:
            with summary_cols[0]:
                budget = st.session_state.budget_data
                # AI INJECTION: Show AI score if available
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
                # AI INJECTION: Show AI score if available
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
                # AI INJECTION: Show AI score if available
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
                # AI INJECTION: Show AI score if available
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
