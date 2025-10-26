"""
AI Financial Advisor Application - LLAMA 3.3
A comprehensive financial planning tool with AI-powered PDF analysis

Required pip packages:
pip install streamlit plotly pandas numpy easyocr torch torchvision torchaudio opencv-python pdf2image pymupdf python-dotenv sentence-transformers langchain-groq
"""

import streamlit as st
import os
import json
import torch
import tempfile
import uuid
import sys
import re
from dotenv import load_dotenv
import fitz  # PyMuPDF for text extraction
import easyocr
from pdf2image import convert_from_path
from langchain_groq import ChatGroq
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from typing import Dict, List, Tuple, Optional, Any

# Test mode check
TEST_MODE = "--test" in sys.argv

if not TEST_MODE:
    st.set_page_config(
        page_title="AI Financial Advisor - LLAMA 3.3",
        page_icon="💰",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    # Custom CSS for dark theme
    st.markdown("""
    <style>
        .stApp { background-color: #0e1117; color: #ffffff; }
        .main-header {
            font-size: 2.5rem; font-weight: bold; color: #ffffff;
            text-align: center; margin-bottom: 2rem;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            -webkit-background-clip: text; -webkit-text-fill-color: transparent;
        }
        .metric-card {
            background: linear-gradient(135deg, #1f2937 0%, #374151 100%);
            padding: 1.5rem; border-radius: 12px; border-left: 5px solid #3b82f6;
            margin: 1rem 0; box-shadow: 0 4px 16px rgba(0,0,0,0.2);
            color: #ffffff; border: 1px solid #4b5563;
        }
        .metric-card h2, .metric-card h3, .metric-card h4 { color: #ffffff !important; }
        .metric-card p { color: #d1d5db !important; }
        .ai-suggestions-card {
            background: linear-gradient(135deg, #581c87 0%, #7c3aed 100%);
            padding: 1.5rem; border-radius: 12px; border-left: 5px solid #a78bfa;
            margin: 1rem 0; box-shadow: 0 4px 16px rgba(0,0,0,0.2);
            color: #ffffff; border: 1px solid #7c3aed;
        }
        .ai-suggestions-card h3, .ai-suggestions-card h4 { color: #ffffff !important; }
        .ai-suggestions-card p, .ai-suggestions-card ul li { color: #e9d5ff !important; margin-bottom: 0.5rem; }
        .flow-card {
            background: linear-gradient(135deg, #1e293b 0%, #334155 100%);
            padding: 1.5rem; border-radius: 15px; color: #ffffff; margin: 1rem 0;
            box-shadow: 0 8px 32px rgba(0,0,0,0.3); border: 1px solid #374151;
        }
        .stButton > button {
            background-color: #3b82f6 !important; color: #ffffff !important;
            border: none !important; border-radius: 8px !important;
        }
        .stButton > button:hover {
            background-color: #2563eb !important;
            box-shadow: 0 4px 12px rgba(59, 130, 246, 0.3) !important;
        }
    </style>
    """, unsafe_allow_html=True)

# Load environment variables
load_dotenv()
working_dir = os.path.dirname(os.path.abspath(__file__))

# GPU availability
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
if torch.cuda.is_available():
    torch.backends.cudnn.benchmark = True

def load_groq_api_key():
    """Load GROQ API key from config.json or environment variables"""
    try:
        with open(os.path.join(working_dir, "config.json"), "r") as f:
            return json.load(f).get("GROQ_API_KEY")
    except FileNotFoundError:
        return os.getenv("GROQ_API_KEY")

groq_api_key = load_groq_api_key() if not TEST_MODE else "test_key"

# Initialize EasyOCR
reader = None
if not TEST_MODE:
    try:
        reader = easyocr.Reader(["en"], gpu=torch.cuda.is_available())
    except Exception as e:
        st.warning(f"EasyOCR initialization failed. OCR features limited.")
        reader = None

# ===================================================================
# PDF EXTRACTION FUNCTIONS
# ===================================================================

def extract_text_from_pdf(file_path):
    """Extract text from PDF using PyMuPDF"""
    if TEST_MODE:
        return ["Test PDF content"]
    try:
        doc = fitz.open(file_path)
        text_list = [page.get_text("text") for page in doc if page.get_text("text").strip()]
        doc.close()
        return text_list if text_list else extract_text_from_images(file_path)
    except Exception as e:
        st.error(f"Error extracting text from PDF: {e}")
        return []

def extract_text_from_images(pdf_path):
    """Extract text from image-based PDFs using OCR"""
    if reader is None or TEST_MODE:
        return ["OCR not available"]
    try:
        images = convert_from_path(pdf_path, dpi=150, first_page=1, last_page=5)
        return ["\n".join(reader.readtext(np.array(img), detail=0)) for img in images]
    except Exception as e:
        st.error(f"Error extracting text from images: {e}")
        return []

def extract_financial_entities_from_text(text: str) -> Dict[str, Any]:
    """Extract financial entities from text using pattern matching"""
    extracted_data = {
        "income": [],
        "expenses": {},
        "investments": [],
        "debts": [],
        "assets": []
    }

    text = text.lower()

    # Income patterns
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

    # Expense categories
    expense_categories = {
        "housing": [r'rent[:\s]+\$?([\d,]+\.?\d*)', r'mortgage[:\s]+\$?([\d,]+\.?\d*)'],
        "utilities": [r'utilit(?:y|ies)[:\s]+\$?([\d,]+\.?\d*)'],
        "food": [r'food[:\s]+\$?([\d,]+\.?\d*)', r'groceries[:\s]+\$?([\d,]+\.?\d*)'],
        "transportation": [r'transportation[:\s]+\$?([\d,]+\.?\d*)', r'car payment[:\s]+\$?([\d,]+\.?\d*)'],
        "insurance": [r'insurance[:\s]+\$?([\d,]+\.?\d*)'],
        "entertainment": [r'entertainment[:\s]+\$?([\d,]+\.?\d*)'],
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

    # Debt patterns
    debt_patterns = [
        r'credit card.*\$?([\d,]+\.?\d*)',
        r'loan.*\$?([\d,]+\.?\d*)',
        r'debt.*\$?([\d,]+\.?\d*)',
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

    # Investment patterns
    investment_patterns = [
        r'401\(?k\)?[:\s]+\$?([\d,]+\.?\d*)',
        r'ira[:\s]+\$?([\d,]+\.?\d*)',
        r'stocks?[:\s]+\$?([\d,]+\.?\d*)',
        r'investment[s]?[:\s]+\$?([\d,]+\.?\d*)',
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

# ===================================================================
# AI ANALYSIS FUNCTIONS
# ===================================================================

def generate_comprehensive_ai_analysis(financial_data: Dict[str, Any]) -> Dict[str, Any]:
    """Generate comprehensive AI analysis with metrics and recommendations"""
    if not groq_api_key or TEST_MODE:
        return generate_fallback_analysis(financial_data)

    try:
        llm = ChatGroq(
            model="llama-3.3-70b-versatile",
            temperature=0.2,
            groq_api_key=groq_api_key
        )

        system_prompt = """You are an advanced AI Financial Analyst.
Analyze the provided financial data, compute key metrics, and propose 3-5 actionable recommendations.
Output valid JSON with keys: ai_summary, visual_plan, recommendations.

Calculate:
1. Savings Rate = (Income - Total Expenses) / Income * 100
2. Debt-to-Income Ratio = Total Debt / Annual Income
3. Emergency Fund Coverage = Savings / Monthly Expenses (in months)
4. Financial Health Score (0-100)"""

        user_prompt = f"""
Financial Data (JSON):
{json.dumps(financial_data, indent=2, default=str)}

Output Format (MUST be valid JSON):
{{
    "ai_summary": "2-3 paragraph narrative summary",
    "visual_plan": {{
        "charts": ["income_vs_expense", "expense_pie", "savings_gauge", "debt_ratio"],
        "key_metrics": {{
            "savings_rate": 18.5,
            "dti": 0.35,
            "emergency_fund_months": 3.5,
            "financial_health_score": 75
        }}
    }},
    "recommendations": [
        "Specific recommendation 1",
        "Specific recommendation 2",
        "Specific recommendation 3"
    ]
}}
"""

        response = llm.invoke([
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ])

        response_text = response.content.strip()

        try:
            if response_text.startswith("{") and response_text.endswith("}"):
                ai_result = json.loads(response_text)
            else:
                json_match = re.search(r'\{[\s\S]*\}', response_text)
                if json_match:
                    ai_result = json.loads(json_match.group())
                else:
                    return generate_fallback_analysis(financial_data)

            required_keys = ["ai_summary", "visual_plan", "recommendations"]
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
    savings_amt = financial_data.get("savings", income - total_expenses)

    savings_rate = (savings_amt / income * 100) if income > 0 else 0
    dti = (total_debt / (income * 12)) if income > 0 else 0
    emergency_fund_months = (savings_amt / total_expenses) if total_expenses > 0 else 0

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
            f"Increase savings rate to 20% by reducing discretionary spending by ${max(0, 0.20 * income - savings_amt):.0f}/month" if savings_rate < 20 else "Maintain your excellent savings rate",
            f"Build emergency fund to 6 months of expenses (${total_expenses * 6:.0f})" if emergency_fund_months < 6 else "Your emergency fund is well-established",
            f"Focus on paying down debt to reduce DTI below 36%" if dti > 0.36 else "Your debt levels are manageable"
        ]
    }

# ===================================================================
# VISUALIZATION FUNCTIONS
# ===================================================================

def create_enhanced_financial_visualizations(financial_data: Dict[str, Any], ai_analysis: Dict[str, Any]):
    """Display comprehensive financial report with visualizations"""
    st.markdown("---")
    st.markdown("## 📊 Your Enhanced Financial Report")

    # AI Summary
    st.markdown('<div class="ai-suggestions-card">', unsafe_allow_html=True)
    st.markdown("### 🤖 AI Financial Analysis")
    st.write(ai_analysis.get("ai_summary", "Analysis not available"))
    st.markdown('</div>', unsafe_allow_html=True)

    # Key Metrics
    st.markdown("### 📈 Key Financial Metrics")
    metrics = ai_analysis.get("visual_plan", {}).get("key_metrics", {})

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        savings_rate = metrics.get('savings_rate', 0)
        delta_vs_target = savings_rate - 20
        st.metric(
            "Savings Rate",
            f"{savings_rate:.1f}%",
            delta=f"{delta_vs_target:.1f}% vs target"
        )

    with col2:
        health_score = metrics.get('financial_health_score', 0)
        health_status = "Good" if health_score >= 70 else "Needs Work"
        st.metric(
            "Financial Health",
            f"{health_score:.0f}/100",
            delta=health_status
        )

    with col3:
        dti = metrics.get('dti', 0)
        dti_status = "Healthy" if dti < 0.36 else "High"
        st.metric(
            "DTI Ratio",
            f"{dti:.1%}",
            delta=dti_status,
            delta_color="inverse"
        )

    with col4:
        emergency_fund = metrics.get('emergency_fund_months', 0)
        ef_status = "Ready" if emergency_fund >= 3 else "Build Up"
        st.metric(
            "Emergency Fund",
            f"{emergency_fund:.1f} mo",
            delta=ef_status
        )

    # Charts
    st.markdown("### 📊 Visual Analysis")
    chart_col1, chart_col2 = st.columns(2)

    with chart_col1:
        # Income vs Expenses
        if financial_data.get("monthly_income") and financial_data.get("expenses"):
            income = financial_data["monthly_income"]
            expenses = financial_data["expenses"]
            total_expenses = sum(expenses.values()) if isinstance(expenses, dict) else 0

            fig1 = go.Figure(data=[
                go.Bar(name='Income', x=['Monthly Cash Flow'], y=[income],
                       marker_color='#10b981', text=[f'${income:,.0f}'], textposition='auto'),
                go.Bar(name='Expenses', x=['Monthly Cash Flow'], y=[total_expenses],
                       marker_color='#ef4444', text=[f'${total_expenses:,.0f}'], textposition='auto')
            ])
            fig1.update_layout(
                title='Income vs Expenses',
                barmode='group',
                plot_bgcolor='#1f2937',
                paper_bgcolor='#1f2937',
                font=dict(color='#ffffff'),
                height=400
            )
            st.plotly_chart(fig1, use_container_width=True)

        # Savings Rate Gauge
        fig3 = go.Figure(go.Indicator(
            mode="gauge+number+delta",
            value=savings_rate,
            domain={'x': [0, 1], 'y': [0, 1]},
            title={'text': "Savings Rate (%)"},
            delta={'reference': 20, 'increasing': {'color': "#10b981"}},
            gauge={
                'axis': {'range': [None, 100]},
                'bar': {'color': "#3b82f6"},
                'steps': [
                    {'range': [0, 10], 'color': "#ef4444"},
                    {'range': [10, 20], 'color': "#f59e0b"},
                    {'range': [20, 100], 'color': "#10b981"}
                ]
            }
        ))
        fig3.update_layout(
            plot_bgcolor='#1f2937',
            paper_bgcolor='#1f2937',
            font=dict(color='#ffffff'),
            height=300
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
                fig2.update_layout(
                    title='Expense Breakdown',
                    plot_bgcolor='#1f2937',
                    paper_bgcolor='#1f2937',
                    font=dict(color='#ffffff'),
                    height=400
                )
                st.plotly_chart(fig2, use_container_width=True)

        # Debt Ratio Gauge
        if financial_data.get("debts") and financial_data.get("monthly_income"):
            dti_ratio = dti * 100
            fig4 = go.Figure(go.Indicator(
                mode="gauge+number",
                value=dti_ratio,
                domain={'x': [0, 1], 'y': [0, 1]},
                title={'text': "Debt-to-Income Ratio (%)"},
                gauge={
                    'axis': {'range': [None, 50]},
                    'bar': {'color': "#3b82f6"},
                    'steps': [
                        {'range': [0, 20], 'color': "#10b981"},
                        {'range': [20, 36], 'color': "#f59e0b"},
                        {'range': [36, 50], 'color': "#ef4444"}
                    ]
                }
            ))
            fig4.update_layout(
                plot_bgcolor='#1f2937',
                paper_bgcolor='#1f2937',
                font=dict(color='#ffffff'),
                height=300
            )
            st.plotly_chart(fig4, use_container_width=True)

    # Recommendations
    st.markdown("### 💡 AI Recommendations")
    recommendations = ai_analysis.get("recommendations", [])

    for i, rec in enumerate(recommendations, 1):
        st.markdown(f"""
        <div class="metric-card">
            <h4>Recommendation {i}</h4>
            <p>{rec}</p>
        </div>
        """, unsafe_allow_html=True)

# ===================================================================
# FINANCIAL CALCULATOR CLASS (kept from original for other tabs)
# ===================================================================

class FinancialCalculator:
    """Core financial calculation functions"""

    @staticmethod
    def calculate_budget_summary(income: float, expenses: Dict[str, float]) -> Dict[str, Any]:
        """Calculate comprehensive budget summary"""
        if income <= 0:
            return {
                'total_income': 0, 'total_expenses': 0, 'savings': 0,
                'savings_rate': 0, 'financial_health': 'Critical',
                'health_color': '#f44336', 'health_score': 0,
                'recommendations': ['Please enter valid income and expense data.'],
                'expense_breakdown': {}
            }

        total_expenses = sum(expenses.values())
        savings = income - total_expenses
        savings_rate = (savings / income * 100) if income > 0 else 0

        # Health scoring
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
        if savings_rate >= 20:
            recommendations.append("🌟 Excellent savings rate! Consider investing surplus funds")

        return {
            'total_income': income,
            'total_expenses': total_expenses,
            'savings': savings,
            'savings_rate': savings_rate,
            'expense_breakdown': expenses,
            'financial_health': health_status,
            'health_color': health_color,
            'health_score': health_score,
            'recommendations': recommendations
        }

# ===================================================================
# FINANCIAL FLOWS (simplified versions for other tabs)
# ===================================================================

class FinancialFlows:
    """Financial advisory flows"""

    @staticmethod
    def budgeting_flow():
        """Interactive budgeting flow"""
        st.markdown('<div class="flow-card"><h2>💰 Smart Budgeting Assistant</h2><p>Create a comprehensive budget plan.</p></div>', unsafe_allow_html=True)

        with st.form("budget_form"):
            st.subheader("Monthly Income")
            col1, col2 = st.columns(2)

            with col1:
                primary_income = st.number_input("Primary Income (after taxes)", min_value=0.0, value=5000.0, step=100.0)
                secondary_income = st.number_input("Secondary Income", min_value=0.0, value=0.0, step=100.0)

            with col2:
                other_income = st.number_input("Other Income", min_value=0.0, value=0.0, step=100.0)
                total_income = primary_income + secondary_income + other_income
                st.metric("Total Monthly Income", f"${total_income:,.2f}")

            st.subheader("Monthly Expenses")
            expense_categories = {
                'housing': 'Housing', 'utilities': 'Utilities', 'groceries': 'Groceries',
                'transportation': 'Transportation', 'insurance': 'Insurance', 'healthcare': 'Healthcare'
            }

            expenses = {}
            col1, col2 = st.columns(2)

            for i, (key, label) in enumerate(expense_categories.items()):
                with col1 if i % 2 == 0 else col2:
                    expenses[key] = st.number_input(label, min_value=0.0, value=0.0, step=50.0, key=f"expense_{key}")

            submitted = st.form_submit_button("Analyze My Budget", type="primary")

        if submitted:
            budget_summary = FinancialCalculator.calculate_budget_summary(total_income, expenses)

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
                st.markdown(f'''
                <div class="metric-card">
                    <h3>Health Score</h3>
                    <h2 style="color: {budget_summary["health_color"]}">{budget_summary["health_score"]}/100</h2>
                </div>
                ''', unsafe_allow_html=True)

# ===================================================================
# MAIN APPLICATION
# ===================================================================

def main():
    """Main application function"""
    if TEST_MODE:
        return

    # Header
    st.markdown('<h1 class="main-header">🦙 AI Financial Advisor - LLAMA 3.3</h1>', unsafe_allow_html=True)

    # Navigation
    st.sidebar.subheader("📊 Financial Tools")
    selected_flow = st.sidebar.selectbox(
        "Choose a Tool",
        ["📘 AI PDF Analyzer", "💰 Budget", "📈 Investment", "💳 Debt", "🏖️ Retirement"]
    )

    # PDF Analyzer Section
    if selected_flow == "📘 AI PDF Analyzer":
        st.markdown('<div class="flow-card"><h2>📘 AI PDF Analyzer</h2><p>Upload financial documents for instant AI-powered analysis</p></div>', unsafe_allow_html=True)

        uploaded_files = st.file_uploader(
            "Upload Financial Documents (PDF)",
            type=["pdf"],
            accept_multiple_files=True,
            help="Upload bank statements, investment reports, or financial documents"
        )

        if uploaded_files:
            all_extracted_text = []
            all_financial_data = []

            with st.spinner("📄 Processing PDFs..."):
                for uploaded_file in uploaded_files:
                    safe_filename = f"{uuid.uuid4().hex}_{uploaded_file.name}"
                    file_path = os.path.join(tempfile.gettempdir(), safe_filename)

                    with open(file_path, "wb") as f:
                        f.write(uploaded_file.getbuffer())

                    try:
                        extracted_text = extract_text_from_pdf(file_path)
                        all_extracted_text.extend(extracted_text)

                        full_text = "\n".join(extracted_text)
                        financial_entities = extract_financial_entities_from_text(full_text)
                        all_financial_data.append(financial_entities)

                        st.success(f"✅ Processed {uploaded_file.name}")
                    except Exception as e:
                        st.error(f"⚠️ Error processing {uploaded_file.name}: {e}")
                    finally:
                        try:
                            os.remove(file_path)
                        except:
                            pass

            # Auto-generate report
            if all_financial_data and any(fd['income'] or fd['expenses'] or fd['debts'] or fd['investments'] for fd in all_financial_data):
                aggregated_income = []
                aggregated_expenses = {}
                aggregated_debts = []
                aggregated_investments = []

                for fd in all_financial_data:
                    aggregated_income.extend(fd.get('income', []))
                    for category, amount in fd.get('expenses', {}).items():
                        aggregated_expenses[category] = aggregated_expenses.get(category, 0) + amount
                    aggregated_debts.extend(fd.get('debts', []))
                    aggregated_investments.extend(fd.get('investments', []))

                total_income = sum(aggregated_income) if aggregated_income else 0
                total_debt = sum(aggregated_debts) if aggregated_debts else 0
                total_investments = sum(aggregated_investments) if aggregated_investments else 0
                total_expenses = sum(aggregated_expenses.values())

                if total_income > 0 or total_expenses > 0:
                    financial_data = {
                        "monthly_income": total_income,
                        "expenses": aggregated_expenses,
                        "debts": aggregated_debts,
                        "investments": aggregated_investments,
                        "savings": total_income - total_expenses,
                        "extracted_from_pdf": True
                    }

                    # Show extracted data
                    with st.expander("🔍 View Extracted Financial Data", expanded=False):
                        col1, col2 = st.columns(2)
                        with col1:
                            st.markdown("### Income")
                            st.write(f"**Total Monthly Income:** ${financial_data['monthly_income']:,.2f}")

                            st.markdown("### Expenses")
                            for category, amount in financial_data['expenses'].items():
                                st.write(f"- {category.title()}: ${amount:,.2f}")
                            st.write(f"**Total Expenses:** ${sum(financial_data['expenses'].values()):,.2f}")

                        with col2:
                            st.markdown("### Debts")
                            if financial_data['debts']:
                                for i, debt in enumerate(financial_data['debts'], 1):
                                    st.write(f"- Debt {i}: ${debt:,.2f}")
                                st.write(f"**Total Debt:** ${sum(financial_data['debts']):,.2f}")
                            else:
                                st.write("No debts detected")

                            st.markdown("### Investments")
                            if financial_data['investments']:
                                for i, inv in enumerate(financial_data['investments'], 1):
                                    st.write(f"- Investment {i}: ${inv:,.2f}")
                                st.write(f"**Total Investments:** ${sum(financial_data['investments']):,.2f}")
                            else:
                                st.write("No investments detected")

                    # Generate AI analysis
                    with st.spinner("🤖 Generating AI analysis..."):
                        ai_analysis = generate_comprehensive_ai_analysis(financial_data)

                    # Display visualizations
                    create_enhanced_financial_visualizations(financial_data, ai_analysis)
                else:
                    st.warning("No financial data detected in uploaded PDFs. Please ensure documents contain financial information.")
            else:
                st.warning("No financial data detected in uploaded PDFs.")

    # Budget Tab
    elif selected_flow == "💰 Budget":
        FinancialFlows.budgeting_flow()

    # Other tabs
    elif selected_flow == "📈 Investment":
        st.markdown('<div class="flow-card"><h2>📈 Investment Planning</h2><p>Coming soon - Investment portfolio analysis</p></div>', unsafe_allow_html=True)

    elif selected_flow == "💳 Debt":
        st.markdown('<div class="flow-card"><h2>💳 Debt Repayment</h2><p>Coming soon - Debt payoff strategies</p></div>', unsafe_allow_html=True)

    elif selected_flow == "🏖️ Retirement":
        st.markdown('<div class="flow-card"><h2>🏖️ Retirement Planning</h2><p>Coming soon - Retirement analysis</p></div>', unsafe_allow_html=True)

if __name__ == "__main__":
    main()
