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

        /* Fix metric card spacing to prevent overlap */
        [data-testid="stMetric"] {
            background-color: #1f2937;
            padding: 1rem;
            border-radius: 8px;
            border: 1px solid #4b5563;
            margin-bottom: 1rem;
        }

        [data-testid="stMetricValue"] {
            font-size: 1.5rem;
            font-weight: bold;
            color: #ffffff;
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

# ===================================================================
# === PDF AUTO ANALYZER - ENHANCEMENT ===
# ===================================================================

def extract_financial_entities_from_text(text: str) -> Dict[str, Any]:
    """
    Extract financial entities from text using pattern matching and NLP.

    Args:
        text: Extracted text from PDF

    Returns:
        Dict containing extracted financial data
    """
    import re

    extracted_data = {
        "income": [],
        "expenses": {},
        "investments": [],
        "debts": [],
        "assets": []
    }

    # Normalize text
    text = text.lower()

    # Extract income patterns
    income_patterns = [
        r'salary[:\s]+\$?([\d,]+\.?\d*)',
        r'income[:\s]+\$?([\d,]+\.?\d*)',
        r'earnings[:\s]+\$?([\d,]+\.?\d*)',
        r'wages[:\s]+\$?([\d,]+\.?\d*)',
        r'gross pay[:\s]+\$?([\d,]+\.?\d*)',
    ]

    for pattern in income_patterns:
        matches = re.findall(pattern, text)
        for match in matches:
            try:
                amount = float(match.replace(',', ''))
                if amount > 0:
                    extracted_data["income"].append(amount)
            except ValueError:
                pass

    # Extract expense categories - EXPANDED PATTERNS FOR ALL MAJOR CATEGORIES
    expense_categories = {
        "housing": [r'rent[:\s]+\$?([\d,]+\.?\d*)', r'mortgage[:\s]+\$?([\d,]+\.?\d*)', r'property tax[:\s]+\$?([\d,]+\.?\d*)', r'hoa[:\s]+\$?([\d,]+\.?\d*)'],
        "utilities": [r'utilit(?:y|ies)[:\s]+\$?([\d,]+\.?\d*)', r'electric(?:ity)?[:\s]+\$?([\d,]+\.?\d*)', r'water[:\s]+\$?([\d,]+\.?\d*)', r'gas[:\s]+\$?([\d,]+\.?\d*)', r'internet[:\s]+\$?([\d,]+\.?\d*)', r'phone[:\s]+\$?([\d,]+\.?\d*)', r'cable[:\s]+\$?([\d,]+\.?\d*)'],
        "food": [r'food[:\s]+\$?([\d,]+\.?\d*)', r'groceries[:\s]+\$?([\d,]+\.?\d*)', r'dining[:\s]+\$?([\d,]+\.?\d*)', r'restaurant[s]?[:\s]+\$?([\d,]+\.?\d*)', r'eating out[:\s]+\$?([\d,]+\.?\d*)'],
        "transportation": [r'transportation[:\s]+\$?([\d,]+\.?\d*)', r'car payment[:\s]+\$?([\d,]+\.?\d*)', r'gas(?:oline)?[:\s]+\$?([\d,]+\.?\d*)', r'fuel[:\s]+\$?([\d,]+\.?\d*)', r'auto[:\s]+\$?([\d,]+\.?\d*)', r'parking[:\s]+\$?([\d,]+\.?\d*)', r'public transit[:\s]+\$?([\d,]+\.?\d*)', r'uber[:\s]+\$?([\d,]+\.?\d*)', r'lyft[:\s]+\$?([\d,]+\.?\d*)'],
        "insurance": [r'insurance[:\s]+\$?([\d,]+\.?\d*)', r'health insurance[:\s]+\$?([\d,]+\.?\d*)', r'auto insurance[:\s]+\$?([\d,]+\.?\d*)', r'life insurance[:\s]+\$?([\d,]+\.?\d*)', r'dental[:\s]+\$?([\d,]+\.?\d*)', r'vision[:\s]+\$?([\d,]+\.?\d*)'],
        "entertainment": [r'entertainment[:\s]+\$?([\d,]+\.?\d*)', r'leisure[:\s]+\$?([\d,]+\.?\d*)', r'movies[:\s]+\$?([\d,]+\.?\d*)', r'streaming[:\s]+\$?([\d,]+\.?\d*)', r'netflix[:\s]+\$?([\d,]+\.?\d*)', r'spotify[:\s]+\$?([\d,]+\.?\d*)', r'subscription[s]?[:\s]+\$?([\d,]+\.?\d*)'],
        "shopping": [r'shopping[:\s]+\$?([\d,]+\.?\d*)', r'clothing[:\s]+\$?([\d,]+\.?\d*)', r'retail[:\s]+\$?([\d,]+\.?\d*)', r'amazon[:\s]+\$?([\d,]+\.?\d*)'],
        "healthcare": [r'healthcare[:\s]+\$?([\d,]+\.?\d*)', r'medical[:\s]+\$?([\d,]+\.?\d*)', r'doctor[:\s]+\$?([\d,]+\.?\d*)', r'pharmacy[:\s]+\$?([\d,]+\.?\d*)', r'prescription[s]?[:\s]+\$?([\d,]+\.?\d*)'],
        "bills": [r'bills?[:\s]+\$?([\d,]+\.?\d*)', r'payment[s]?[:\s]+\$?([\d,]+\.?\d*)'],
        "loans": [r'loan[s]?[:\s]+\$?([\d,]+\.?\d*)', r'student loan[:\s]+\$?([\d,]+\.?\d*)', r'personal loan[:\s]+\$?([\d,]+\.?\d*)'],
        "emi": [r'emi[:\s]+\$?([\d,]+\.?\d*)', r'installment[:\s]+\$?([\d,]+\.?\d*)'],
        "debt_payments": [r'debt payment[s]?[:\s]+\$?([\d,]+\.?\d*)', r'minimum payment[:\s]+\$?([\d,]+\.?\d*)'],
        "savings": [r'savings[:\s]+\$?([\d,]+\.?\d*)', r'emergency fund[:\s]+\$?([\d,]+\.?\d*)'],
        "other": [r'misc(?:ellaneous)?[:\s]+\$?([\d,]+\.?\d*)', r'other[:\s]+\$?([\d,]+\.?\d*)'],
    }

    for category, patterns in expense_categories.items():
        for pattern in patterns:
            matches = re.findall(pattern, text)
            for match in matches:
                try:
                    amount = float(match.replace(',', ''))
                    if amount > 0:
                        if category not in extracted_data["expenses"]:
                            extracted_data["expenses"][category] = 0
                        extracted_data["expenses"][category] += amount
                except ValueError:
                    pass

    # Extract debt information
    debt_patterns = [
        r'credit card.*\$?([\d,]+\.?\d*)',
        r'loan.*\$?([\d,]+\.?\d*)',
        r'debt.*\$?([\d,]+\.?\d*)',
        r'balance.*\$?([\d,]+\.?\d*)',
    ]

    for pattern in debt_patterns:
        matches = re.findall(pattern, text)
        for match in matches:
            try:
                amount = float(match.replace(',', ''))
                if amount > 0:
                    extracted_data["debts"].append(amount)
            except ValueError:
                pass

    # Extract investment information
    investment_patterns = [
        r'401\(?k\)?[:\s]+\$?([\d,]+\.?\d*)',
        r'ira[:\s]+\$?([\d,]+\.?\d*)',
        r'stocks?[:\s]+\$?([\d,]+\.?\d*)',
        r'bonds?[:\s]+\$?([\d,]+\.?\d*)',
        r'investment[s]?[:\s]+\$?([\d,]+\.?\d*)',
        r'portfolio[:\s]+\$?([\d,]+\.?\d*)',
    ]

    for pattern in investment_patterns:
        matches = re.findall(pattern, text)
        for match in matches:
            try:
                amount = float(match.replace(',', ''))
                if amount > 0:
                    extracted_data["investments"].append(amount)
            except ValueError:
                pass

    return extracted_data

def process_pdf_and_extract_financials(uploaded_file) -> Tuple[str, Dict[str, Any]]:
    """
    Process uploaded PDF and extract financial data.

    Args:
        uploaded_file: Streamlit uploaded file object

    Returns:
        Tuple of (extracted_text, financial_data)
    """
    # Save uploaded file temporarily
    safe_filename = f"{uuid.uuid4().hex}_{uploaded_file.name}"
    file_path = os.path.join(tempfile.gettempdir(), safe_filename)

    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    try:
        # Extract text from PDF
        extracted_text = extract_text_from_pdf(file_path)
        full_text = "\n".join(extracted_text)

        # Extract financial entities
        financial_data = extract_financial_entities_from_text(full_text)

        return full_text, financial_data
    finally:
        # Clean up temporary file
        try:
            os.remove(file_path)
        except:
            pass

# ===================================================================
# === STRUCTURED AI OUTPUT & RECOMMENDATIONS - ENHANCEMENT ===
# ===================================================================

def generate_comprehensive_ai_analysis(financial_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Master AI analysis function using structured prompts.

    Args:
        financial_data: Complete financial data dictionary

    Returns:
        Structured JSON output with summary, metrics, visualizations, and recommendations
    """
    if not groq_api_key or TEST_MODE:
        return generate_fallback_analysis(financial_data)

    try:
        # Initialize LLM
        llm = ChatGroq(
            model="llama-3.3-70b-versatile",
            temperature=0.2,
            groq_api_key=groq_api_key
        )

        # Create master prompt
        system_prompt = """You are an advanced autonomous AI Financial Analyst.
Analyze the provided financial data, generate visual and narrative insights, compute key metrics, and propose 3-5 actionable improvements.
Always output valid JSON with keys: ai_summary, visual_plan, recommendations, qa_context.

The visual_plan should include:
- charts: array of chart types to display
- key_metrics: object with calculated financial ratios

The recommendations should be specific, quantified, and actionable."""

        user_prompt = f"""
Analyze this financial data and provide comprehensive insights:

Financial Data (JSON):
{json.dumps(financial_data, indent=2, default=str)}

Calculate and provide:
1. Savings Rate = (Income - Total Expenses) / Income * 100
2. Debt-to-Income Ratio = Total Debt / Annual Income
3. Emergency Fund Coverage = Savings / Monthly Expenses (in months)
4. Financial Health Score (0-100)

Output Format (MUST be valid JSON):
{{
    "ai_summary": "2-3 paragraph narrative summary of financial health",
    "visual_plan": {{
        "charts": ["income_vs_expense", "expense_pie", "savings_gauge", "debt_ratio"],
        "key_metrics": {{
            "savings_rate": 0.18,
            "dti": 0.35,
            "emergency_fund_months": 3.5,
            "financial_health_score": 75
        }}
    }},
    "recommendations": [
        "Specific recommendation 1 with numbers",
        "Specific recommendation 2 with numbers",
        "Specific recommendation 3 with numbers"
    ],
    "qa_context": "Full detailed context for conversational follow-up questions"
}}
"""

        # Call LLM
        response = llm.invoke([
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ])

        response_text = response.content.strip()

        # Parse JSON response
        import re
        try:
            # Extract JSON from response
            if response_text.startswith("{") and response_text.endswith("}"):
                ai_result = json.loads(response_text)
            else:
                # Try to find JSON block
                json_match = re.search(r'\{[\s\S]*\}', response_text)
                if json_match:
                    ai_result = json.loads(json_match.group())
                else:
                    return generate_fallback_analysis(financial_data)

            # Validate structure
            required_keys = ["ai_summary", "visual_plan", "recommendations", "qa_context"]
            if not all(key in ai_result for key in required_keys):
                return generate_fallback_analysis(financial_data)

            return ai_result

        except json.JSONDecodeError:
            return generate_fallback_analysis(financial_data)

    except Exception as e:
        if not TEST_MODE:
            st.warning(f"AI analysis temporarily unavailable: {str(e)}")
        return generate_fallback_analysis(financial_data)

def generate_fallback_analysis(financial_data: Dict[str, Any]) -> Dict[str, Any]:
    """Generate deterministic analysis when AI is unavailable"""
    income = financial_data.get("monthly_income", 0)
    expenses = financial_data.get("expenses", {})
    total_expenses = sum(expenses.values()) if isinstance(expenses, dict) else 0
    debts = financial_data.get("debts", [])
    total_debt = sum(debts) if isinstance(debts, list) else 0

    savings = income - total_expenses
    savings_rate = (savings / income * 100) if income > 0 else 0
    dti = (total_debt / (income * 12)) if income > 0 else 0
    emergency_fund_months = (financial_data.get("savings", 0) / total_expenses) if total_expenses > 0 else 0

    # Calculate financial health score
    score = 50
    if savings_rate > 20:
        score += 20
    elif savings_rate > 10:
        score += 10
    if dti < 0.36:
        score += 15
    if emergency_fund_months >= 3:
        score += 15

    return {
        "ai_summary": f"Your current financial health shows a savings rate of {savings_rate:.1f}% and a debt-to-income ratio of {dti:.1%}. You have approximately {emergency_fund_months:.1f} months of emergency fund coverage. {'You are on a good financial path.' if score >= 70 else 'There is room for improvement in your financial planning.'}",
        "visual_plan": {
            "charts": ["income_vs_expense", "expense_pie", "savings_gauge", "debt_ratio"],
            "key_metrics": {
                "savings_rate": round(savings_rate, 2),
                "dti": round(dti, 2),
                "emergency_fund_months": round(emergency_fund_months, 2),
                "financial_health_score": score
            }
        },
        "recommendations": [
            f"Increase savings rate to 20% by reducing discretionary spending by ${(0.20 * income - savings):.0f}/month" if savings_rate < 20 else "Maintain your excellent savings rate",
            f"Build emergency fund to 6 months of expenses (${total_expenses * 6:.0f})" if emergency_fund_months < 6 else "Your emergency fund is well-established",
            f"Focus on paying down debt to reduce DTI below 36%" if dti > 0.36 else "Your debt levels are manageable"
        ],
        "qa_context": f"User has monthly income of ${income:.0f}, total expenses of ${total_expenses:.0f}, savings of ${savings:.0f}, and total debt of ${total_debt:.0f}. Savings rate is {savings_rate:.1f}% and DTI is {dti:.1%}."
    }

# ===================================================================
# === DIGITAL REPORT & VISUALIZATION - ENHANCEMENT ===
# ===================================================================

def create_enhanced_financial_visualizations(financial_data: Dict[str, Any], ai_analysis: Dict[str, Any]):
    """
    Display comprehensive digital financial report with visualizations.

    Args:
        financial_data: Raw financial data
        ai_analysis: AI-generated analysis with metrics
    """
    st.markdown("---")
    st.markdown("## 📊 Your Enhanced Financial Report")

    # Display AI Summary
    st.markdown('<div class="ai-suggestions-card">', unsafe_allow_html=True)
    st.markdown("### 🤖 AI Financial Analysis")
    st.write(ai_analysis.get("ai_summary", "Analysis not available"))
    st.markdown('</div>', unsafe_allow_html=True)

    # Display Key Metrics
    st.markdown("### 📈 Key Financial Metrics")
    metrics = ai_analysis.get("visual_plan", {}).get("key_metrics", {})

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        savings_rate_val = float(metrics.get('savings_rate', 0))
        st.metric(
            "Savings Rate",
            f"{savings_rate_val:.1f}%",
            delta=f"{savings_rate_val - 20:.1f}% vs 20% target"
        )

    with col2:
        health_score_val = float(metrics.get('financial_health_score', 0))
        st.metric(
            "Financial Health",
            f"{health_score_val:.0f}/100",
            delta="Good" if health_score_val >= 70 else "Needs Work"
        )

    with col3:
        dti_val = float(metrics.get('dti', 0))
        st.metric(
            "DTI Ratio",
            f"{dti_val:.1%}",
            delta="Healthy" if dti_val < 0.36 else "High",
            delta_color="inverse"
        )

    with col4:
        emergency_months_val = float(metrics.get('emergency_fund_months', 0))
        st.metric(
            "Emergency Fund",
            f"{emergency_months_val:.1f} mo",
            delta="Ready" if emergency_months_val >= 3 else "Build Up"
        )

    # Display Charts
    st.markdown("### 📊 Visual Analysis")

    chart_col1, chart_col2 = st.columns(2)

    with chart_col1:
        # Income vs Expenses Chart
        if financial_data.get("monthly_income") and financial_data.get("expenses"):
            income = float(financial_data["monthly_income"])
            expenses = financial_data["expenses"]
            total_expenses = float(sum(expenses.values())) if isinstance(expenses, dict) else 0

            fig1 = go.Figure(data=[
                go.Bar(name='Income', x=['Monthly Cash Flow'], y=[income], marker_color='#10b981', text=[f'${income:,.0f}'], textposition='outside', textfont=dict(size=14)),
                go.Bar(name='Expenses', x=['Monthly Cash Flow'], y=[total_expenses], marker_color='#ef4444', text=[f'${total_expenses:,.0f}'], textposition='outside', textfont=dict(size=14))
            ])
            fig1.update_layout(
                title='Income vs Expenses',
                barmode='group',
                plot_bgcolor='#1f2937',
                paper_bgcolor='#1f2937',
                font=dict(color='#ffffff'),
                height=450,
                margin=dict(t=50, b=50, l=50, r=50),
                yaxis=dict(gridcolor='#374151')
            )
            st.plotly_chart(fig1, use_container_width=True)

        # Savings Rate Gauge
        savings_rate_value = float(metrics.get('savings_rate', 0))
        fig3 = go.Figure(go.Indicator(
            mode="gauge+number+delta",
            value=savings_rate_value,
            domain={'x': [0, 1], 'y': [0, 1]},
            title={'text': "Savings Rate (%)", 'font': {'size': 16}},
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
            number={'font': {'size': 32}}
        ))
        fig3.update_layout(
            plot_bgcolor='#1f2937',
            paper_bgcolor='#1f2937',
            font=dict(color='#ffffff', size=12),
            height=350,
            margin=dict(t=80, b=20, l=20, r=20)
        )
        st.plotly_chart(fig3, use_container_width=True)

    with chart_col2:
        # Expense Breakdown Pie
        if financial_data.get("expenses"):
            expenses = financial_data["expenses"]
            if isinstance(expenses, dict) and expenses:
                labels = list(expenses.keys())
                values = list(expenses.values())
                fig2 = go.Figure(data=[go.Pie(labels=labels, values=values, hole=.3)])
                fig2.update_layout(title='Expense Breakdown', plot_bgcolor='#1f2937', paper_bgcolor='#1f2937', font=dict(color='#ffffff'), height=400)
                st.plotly_chart(fig2, use_container_width=True)

        # Debt Ratio Gauge
        if financial_data.get("debts") and financial_data.get("monthly_income"):
            total_debt = sum(financial_data["debts"]) if isinstance(financial_data["debts"], list) else float(financial_data["debts"])
            dti_ratio = float(metrics.get('dti', 0)) * 100
            fig4 = go.Figure(go.Indicator(
                mode="gauge+number",
                value=dti_ratio,
                domain={'x': [0, 1], 'y': [0, 1]},
                title={'text': "Debt-to-Income Ratio (%)", 'font': {'size': 16}},
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
                number={'font': {'size': 32}}
            ))
            fig4.update_layout(
                plot_bgcolor='#1f2937',
                paper_bgcolor='#1f2937',
                font=dict(color='#ffffff', size=12),
                height=350,
                margin=dict(t=80, b=20, l=20, r=20)
            )
            st.plotly_chart(fig4, use_container_width=True)

    # Display Recommendations
    st.markdown("### 💡 AI Recommendations")
    recommendations = ai_analysis.get("recommendations", [])

    for i, rec in enumerate(recommendations, 1):
        st.markdown(f"""
        <div class="metric-card">
            <h4>Recommendation {i}</h4>
            <p>{rec}</p>
        </div>
        """, unsafe_allow_html=True)

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
        return text_list if text_list else [""]
    except Exception as e:
        st.error(f"⚠️ Error extracting text from PDF: {e}")
        return [""]

def main():
    """Main application function"""
    if TEST_MODE:
        return

    # Header
    st.markdown('<h1 class="main-header">🦙 AI Financial Advisor - LLAMA 3.3</h1>', unsafe_allow_html=True)

    st.info("💡 **Disclaimer**: AI suggestions are educational only and not financial advice. Always consult with a qualified financial professional for personalized guidance.")

    # Navigation
    st.sidebar.subheader("📊 Financial Tools")
    selected_flow = st.sidebar.selectbox(
        "Choose a Financial Flow",
        ["Chat Interface", "Smart Budgeting", "Investment Planning", "Debt Repayment", "Retirement Planning"]
    )

    # Main content area - REFACTORED CHAT INTERFACE
    if selected_flow == "Chat Interface":
        st.subheader("📄 Upload Financial Report / PDF")
        uploaded_file = st.file_uploader("Upload your financial PDF", type=["pdf"])

        if "financial_data" not in st.session_state:
            st.session_state.financial_data = None
        if "ai_analysis" not in st.session_state:
            st.session_state.ai_analysis = None

        if uploaded_file is not None:
            if uploaded_file != st.session_state.get("uploaded_pdf"):
                # Reset chat & analysis memory
                st.session_state.pop("enhanced_chat_messages", None)
                st.session_state.pop("ai_analysis", None)

                with st.spinner("Analyzing PDF..."):
                    text, financial_data = process_pdf_and_extract_financials(uploaded_file)
                    ai_analysis = generate_comprehensive_ai_analysis(financial_data)
                    st.session_state.financial_data = financial_data
                    st.session_state.ai_analysis = ai_analysis
                    st.session_state.uploaded_pdf = uploaded_file
                    st.success("✅ Analysis completed successfully!")

        if st.session_state.ai_analysis:
            create_enhanced_financial_visualizations(
                st.session_state.financial_data,
                st.session_state.ai_analysis
            )
    else:
        st.info(f"Selected flow: {selected_flow} - Implementation would go here")

if __name__ == "__main__":
    main()
