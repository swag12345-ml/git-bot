"""
AI Financial Advisor Application - LLAMA 3.3 (ENHANCED VERSION)
A comprehensive financial planning tool with AI-powered insights, PDF analysis, and interactive dashboards

NEW FEATURES:
- PDF-First Flow: Automatic extraction of financial data from uploaded documents
- Digital Financial Report & Dashboard: Interactive visualizations with Plotly
- Structured AI Output: JSON-formatted recommendations with metrics
- Auto Conversational Q&A: Seamless chat interface with financial context
- Enhanced State Management: Persistent session data

Required pip packages:
pip install streamlit plotly pandas numpy easyocr torch torchvision torchaudio opencv-python pdf2image pymupdf python-dotenv faiss-cpu sentence-transformers langchain langchain-community langchain-groq langchain-huggingface langchain-text-splitters pillow
"""

import streamlit as st  # Streamlit must be imported first
import os
import json
import torch
import asyncio
import tempfile
import uuid
import sys
import re
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
from PIL import Image

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
            background: linear-gradient(135deg, #10b981 0%, #3b82f6 100%);
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
            background: linear-gradient(135deg, #065f46 0%, #10b981 100%);
            padding: 1.5rem;
            border-radius: 12px;
            border-left: 5px solid #34d399;
            margin: 1rem 0;
            box-shadow: 0 4px 16px rgba(0,0,0,0.2);
            color: #ffffff;
            border: 1px solid #10b981;
        }
        .ai-suggestions-card h3, .ai-suggestions-card h4 {
            color: #ffffff !important;
        }
        .ai-suggestions-card p, .ai-suggestions-card ul li {
            color: #d1fae5 !important;
            margin-bottom: 0.5rem;
        }

        /* PDF Extraction Card */
        .pdf-card {
            background: linear-gradient(135deg, #1e40af 0%, #3b82f6 100%);
            padding: 1.5rem;
            border-radius: 12px;
            border-left: 5px solid #60a5fa;
            margin: 1rem 0;
            box-shadow: 0 4px 16px rgba(0,0,0,0.2);
            color: #ffffff;
            border: 1px solid #3b82f6;
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
            background: linear-gradient(135deg, #065f46 0%, #10b981 100%);
            border-left: 4px solid #34d399;
            color: #ffffff;
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

        /* Button styling */
        .stButton > button {
            background-color: #10b981 !important;
            color: #ffffff !important;
            border: none !important;
            border-radius: 8px !important;
        }

        .stButton > button:hover {
            background-color: #059669 !important;
            box-shadow: 0 4px 12px rgba(16, 185, 129, 0.3) !important;
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
    </style>
    """, unsafe_allow_html=True)

# Load environment variables
load_dotenv()
working_dir = os.path.dirname(os.path.abspath(__file__))

# Ensure GPU availability
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
if torch.cuda.is_available():
    torch.backends.cudnn.benchmark = True  # Optimize GPU performance

# Initialize EasyOCR reader (global, lazy-loaded)
reader = None

def get_ocr_reader():
    """Lazy-load OCR reader to save memory"""
    global reader
    if reader is None and not TEST_MODE:
        try:
            reader = easyocr.Reader(['en'], gpu=DEVICE == 'cuda')
        except Exception as e:
            st.warning(f"OCR initialization failed: {e}")
    return reader

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
# === PDF AUTO ANALYZER ===
# ===================================================================

def extract_financial_entities_from_text(text: str) -> Dict[str, Any]:
    """
    Extract financial entities from text using pattern matching and NLP.

    Args:
        text: Extracted text from PDF

    Returns:
        Dict containing extracted financial data
    """
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

    # Extract expense categories
    expense_categories = {
        "housing": [r'rent[:\s]+\$?([\d,]+\.?\d*)', r'mortgage[:\s]+\$?([\d,]+\.?\d*)'],
        "utilities": [r'utilit(?:y|ies)[:\s]+\$?([\d,]+\.?\d*)', r'electric[:\s]+\$?([\d,]+\.?\d*)', r'water[:\s]+\$?([\d,]+\.?\d*)'],
        "food": [r'food[:\s]+\$?([\d,]+\.?\d*)', r'groceries[:\s]+\$?([\d,]+\.?\d*)', r'dining[:\s]+\$?([\d,]+\.?\d*)'],
        "transportation": [r'transportation[:\s]+\$?([\d,]+\.?\d*)', r'car payment[:\s]+\$?([\d,]+\.?\d*)', r'gas[:\s]+\$?([\d,]+\.?\d*)'],
        "insurance": [r'insurance[:\s]+\$?([\d,]+\.?\d*)', r'health insurance[:\s]+\$?([\d,]+\.?\d*)'],
        "entertainment": [r'entertainment[:\s]+\$?([\d,]+\.?\d*)', r'leisure[:\s]+\$?([\d,]+\.?\d*)'],
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

def extract_text_from_pdf(file_path):
    """
    Extracts text from PDFs using PyMuPDF, falls back to OCR if needed.

    Args:
        file_path: Path to the PDF file

    Returns:
        List of text content from each page
    """
    if TEST_MODE:
        return ["Test PDF content with salary: $5000, rent: $1500, groceries: $400"]

    try:
        doc = fitz.open(file_path)
        text_list = [page.get_text("text") for page in doc if page.get_text("text").strip()]
        doc.close()
        return text_list if text_list else extract_text_from_images(file_path)
    except Exception as e:
        st.error(f"Error extracting text from PDF: {e}")
        return []

def extract_text_from_images(pdf_path):
    """
    Extracts text from image-based PDFs using GPU-accelerated EasyOCR.

    Args:
        pdf_path: Path to the PDF file

    Returns:
        List of extracted text from each page
    """
    ocr_reader = get_ocr_reader()
    if ocr_reader is None or TEST_MODE:
        return ["OCR not available"]

    try:
        images = convert_from_path(pdf_path, dpi=150, first_page=1, last_page=5)
        return ["\n".join(ocr_reader.readtext(np.array(img), detail=0)) for img in images]
    except Exception as e:
        st.error(f"Error extracting text from images: {e}")
        return []

# ===================================================================
# === DIGITAL REPORT & VISUALIZATION ===
# ===================================================================

class FinancialVisualizer:
    """Creates interactive financial visualizations using Plotly"""

    @staticmethod
    def create_income_vs_expenses_chart(income: float, expenses: Dict[str, float]) -> go.Figure:
        """Create income vs expenses comparison chart"""
        total_expenses = sum(expenses.values())

        fig = go.Figure(data=[
            go.Bar(
                name='Income',
                x=['Monthly Cash Flow'],
                y=[income],
                marker_color='#10b981',
                text=[f'${income:,.0f}'],
                textposition='auto',
            ),
            go.Bar(
                name='Expenses',
                x=['Monthly Cash Flow'],
                y=[total_expenses],
                marker_color='#ef4444',
                text=[f'${total_expenses:,.0f}'],
                textposition='auto',
            )
        ])

        fig.update_layout(
            title='Income vs Expenses',
            barmode='group',
            plot_bgcolor='#1f2937',
            paper_bgcolor='#1f2937',
            font=dict(color='#ffffff'),
            xaxis=dict(showgrid=False),
            yaxis=dict(showgrid=True, gridcolor='#374151'),
            height=400
        )

        return fig

    @staticmethod
    def create_expense_breakdown_pie(expenses: Dict[str, float]) -> go.Figure:
        """Create expense category breakdown pie chart"""
        if not expenses:
            return None

        labels = list(expenses.keys())
        values = list(expenses.values())

        fig = go.Figure(data=[go.Pie(
            labels=labels,
            values=values,
            hole=.3,
            marker=dict(colors=['#3b82f6', '#10b981', '#f59e0b', '#ef4444', '#8b5cf6', '#ec4899'])
        )])

        fig.update_layout(
            title='Expense Breakdown',
            plot_bgcolor='#1f2937',
            paper_bgcolor='#1f2937',
            font=dict(color='#ffffff'),
            height=400
        )

        return fig

    @staticmethod
    def create_savings_rate_gauge(savings_rate: float) -> go.Figure:
        """Create savings rate gauge chart"""
        fig = go.Figure(go.Indicator(
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
                ],
                'threshold': {
                    'line': {'color': "white", 'width': 4},
                    'thickness': 0.75,
                    'value': 20
                }
            }
        ))

        fig.update_layout(
            plot_bgcolor='#1f2937',
            paper_bgcolor='#1f2937',
            font=dict(color='#ffffff'),
            height=300
        )

        return fig

    @staticmethod
    def create_debt_ratio_chart(debt: float, income: float) -> go.Figure:
        """Create debt-to-income ratio chart"""
        dti_ratio = (debt / (income * 12) * 100) if income > 0 else 0

        fig = go.Figure(go.Indicator(
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
                ],
                'threshold': {
                    'line': {'color': "white", 'width': 4},
                    'thickness': 0.75,
                    'value': 36
                }
            }
        ))

        fig.update_layout(
            plot_bgcolor='#1f2937',
            paper_bgcolor='#1f2937',
            font=dict(color='#ffffff'),
            height=300
        )

        return fig

    @staticmethod
    def create_investment_growth_projection(
        current_amount: float,
        monthly_contribution: float,
        years: int,
        annual_return: float
    ) -> go.Figure:
        """Create investment growth projection chart"""
        months = years * 12
        monthly_rate = annual_return / 12

        balances = []
        contributions_only = []

        balance = current_amount
        contributed = current_amount

        for month in range(months + 1):
            balances.append(balance)
            contributions_only.append(contributed)

            if month < months:
                balance = balance * (1 + monthly_rate) + monthly_contribution
                contributed += monthly_contribution

        time_periods = list(range(0, months + 1))

        fig = go.Figure()

        fig.add_trace(go.Scatter(
            x=time_periods,
            y=contributions_only,
            mode='lines',
            name='Contributions Only',
            line=dict(color='#f59e0b', width=2, dash='dash')
        ))

        fig.add_trace(go.Scatter(
            x=time_periods,
            y=balances,
            mode='lines',
            name='With Investment Growth',
            line=dict(color='#10b981', width=3),
            fill='tonexty'
        ))

        fig.update_layout(
            title=f'Investment Growth Projection ({years} Years)',
            xaxis_title='Months',
            yaxis_title='Portfolio Value ($)',
            plot_bgcolor='#1f2937',
            paper_bgcolor='#1f2937',
            font=dict(color='#ffffff'),
            xaxis=dict(showgrid=True, gridcolor='#374151'),
            yaxis=dict(showgrid=True, gridcolor='#374151'),
            height=400
        )

        return fig

# ===================================================================
# === STRUCTURED AI OUTPUT & RECOMMENDATIONS ===
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
# === AI CONVERSATION ENGINE ===
# ===================================================================

def setup_vectorstore(documents: List[str]):
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

def create_conversation_chain(vectorstore):
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
        if "memory" not in st.session_state:
            st.session_state.memory = ConversationBufferMemory(
                memory_key="chat_history",
                return_messages=True,
                output_key="answer"
            )

        llm = ChatGroq(
            model="llama-3.3-70b-versatile",
            temperature=0.3,
            groq_api_key=groq_api_key
        )
        retriever = vectorstore.as_retriever(search_kwargs={"k": 5})

        return ConversationalRetrievalChain.from_llm(
            llm=llm,
            retriever=retriever,
            chain_type="stuff",
            memory=st.session_state.memory,
            return_source_documents=True,
            verbose=False
        )
    except Exception as e:
        st.error(f"Error creating conversation chain: {e}")
        return None

async def get_ai_response(user_input: str, financial_context: str = "") -> str:
    """
    Get AI response for user query with financial context.

    Args:
        user_input: User's question
        financial_context: Additional financial context

    Returns:
        AI response string
    """
    if TEST_MODE:
        return "This is a test response from the AI assistant."

    try:
        if "conversation_chain" in st.session_state and st.session_state.conversation_chain:
            # Enhanced prompt with financial context
            enhanced_prompt = f"""
Financial Context: {financial_context}

User Question: {user_input}

Provide specific, actionable financial advice based on the context above.
"""

            response = await asyncio.to_thread(
                st.session_state.conversation_chain.invoke,
                {"question": enhanced_prompt}
            )
            return response.get("answer", "I couldn't process that question.")
        else:
            # Use direct LLM call without retrieval
            llm = ChatGroq(
                model="llama-3.3-70b-versatile",
                temperature=0.3,
                groq_api_key=groq_api_key
            )

            response = llm.invoke(f"""
You are a helpful financial advisor.

Financial Context: {financial_context}

User Question: {user_input}

Provide clear, actionable advice.
""")
            return response.content

    except Exception as e:
        return f"I apologize, but I encountered an error: {str(e)}. Please try rephrasing your question."

# ===================================================================
# === DISPLAY FUNCTIONS ===
# ===================================================================

def display_digital_report(
    financial_data: Dict[str, Any],
    ai_analysis: Dict[str, Any]
):
    """
    Display comprehensive digital financial report with visualizations.

    Args:
        financial_data: Raw financial data
        ai_analysis: AI-generated analysis with metrics
    """
    st.markdown("---")
    st.markdown("## 📊 Your Financial Report")

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
        st.metric(
            "Savings Rate",
            f"{metrics.get('savings_rate', 0):.1f}%",
            delta=f"{metrics.get('savings_rate', 0) - 20:.1f}% vs target"
        )

    with col2:
        st.metric(
            "Financial Health",
            f"{metrics.get('financial_health_score', 0):.0f}/100",
            delta="Good" if metrics.get('financial_health_score', 0) >= 70 else "Needs Work"
        )

    with col3:
        st.metric(
            "DTI Ratio",
            f"{metrics.get('dti', 0):.1%}",
            delta="Healthy" if metrics.get('dti', 0) < 0.36 else "High",
            delta_color="inverse"
        )

    with col4:
        st.metric(
            "Emergency Fund",
            f"{metrics.get('emergency_fund_months', 0):.1f} mo",
            delta="Ready" if metrics.get('emergency_fund_months', 0) >= 3 else "Build Up"
        )

    # Display Charts
    st.markdown("### 📊 Visual Analysis")

    chart_col1, chart_col2 = st.columns(2)

    with chart_col1:
        # Income vs Expenses
        if financial_data.get("monthly_income") and financial_data.get("expenses"):
            fig1 = FinancialVisualizer.create_income_vs_expenses_chart(
                financial_data["monthly_income"],
                financial_data["expenses"]
            )
            st.plotly_chart(fig1, use_container_width=True)

        # Savings Rate Gauge
        fig3 = FinancialVisualizer.create_savings_rate_gauge(
            metrics.get('savings_rate', 0)
        )
        st.plotly_chart(fig3, use_container_width=True)

    with chart_col2:
        # Expense Breakdown
        if financial_data.get("expenses"):
            fig2 = FinancialVisualizer.create_expense_breakdown_pie(
                financial_data["expenses"]
            )
            if fig2:
                st.plotly_chart(fig2, use_container_width=True)

        # Debt Ratio
        if financial_data.get("debts") and financial_data.get("monthly_income"):
            total_debt = sum(financial_data["debts"]) if isinstance(financial_data["debts"], list) else financial_data["debts"]
            fig4 = FinancialVisualizer.create_debt_ratio_chart(
                total_debt,
                financial_data["monthly_income"]
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

def display_chat_interface(financial_context: str):
    """
    Display auto-start chat interface with financial context.

    Args:
        financial_context: QA context from AI analysis
    """
    st.markdown("---")
    st.markdown("## 💬 Ask Questions About Your Finances")
    st.info("Your financial data has been analyzed. Ask me anything about your financial situation!")

    # Initialize chat history
    if "chat_messages" not in st.session_state:
        st.session_state.chat_messages = []

    # Display chat history
    for message in st.session_state.chat_messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Chat input
    if user_question := st.chat_input("Ask me anything about your finances..."):
        # Add user message
        st.session_state.chat_messages.append({"role": "user", "content": user_question})

        with st.chat_message("user"):
            st.markdown(user_question)

        # Get AI response
        with st.chat_message("assistant"):
            with st.spinner("Analyzing..."):
                try:
                    response = asyncio.run(get_ai_response(user_question, financial_context))
                except Exception as e:
                    response = f"I apologize, but I encountered an error. Please try again. Error: {str(e)}"

            st.markdown(response)

        # Add assistant response
        st.session_state.chat_messages.append({"role": "assistant", "content": response})

# ===================================================================
# === MAIN APPLICATION ===
# ===================================================================

def main():
    """Main application entry point"""
    if TEST_MODE:
        st.write("Running in TEST MODE")
        return

    # Header
    st.markdown('<h1 class="main-header">🦙 AI Financial Advisor - LLAMA 3.3 (Enhanced)</h1>', unsafe_allow_html=True)

    st.info("💡 Upload a PDF financial document to automatically extract and analyze your financial data, or enter data manually below.")

    # ===== PDF-FIRST FLOW =====
    st.markdown("## 📄 Upload Financial Document")

    uploaded_pdf = st.file_uploader(
        "Upload Bank Statement, Budget, or Financial Report (PDF)",
        type=["pdf"],
        help="Upload a PDF containing your financial information. The AI will automatically extract income, expenses, debts, and investments."
    )

    if uploaded_pdf:
        with st.spinner("📄 Processing PDF and extracting financial data..."):
            try:
                full_text, extracted_data = process_pdf_and_extract_financials(uploaded_pdf)

                st.success("✅ PDF processed successfully!")

                # Display extracted data
                with st.expander("🔍 View Extracted Financial Data", expanded=False):
                    st.json(extracted_data)

                # Build structured financial data
                financial_data = {
                    "monthly_income": sum(extracted_data["income"]) if extracted_data["income"] else 0,
                    "expenses": extracted_data["expenses"],
                    "debts": extracted_data["debts"],
                    "investments": extracted_data["investments"],
                    "savings": 0,  # Will be calculated
                    "extracted_from_pdf": True
                }

                # Calculate derived values
                total_expenses = sum(financial_data["expenses"].values())
                financial_data["savings"] = financial_data["monthly_income"] - total_expenses

                # Store in session state
                st.session_state.financial_data = financial_data
                st.session_state.pdf_text = full_text

                # Generate AI Analysis
                st.markdown("### 🤖 Generating AI Analysis...")
                with st.spinner("Analyzing your financial situation..."):
                    ai_analysis = generate_comprehensive_ai_analysis(financial_data)
                    st.session_state.ai_analysis = ai_analysis

                # Display Digital Report
                display_digital_report(financial_data, ai_analysis)

                # Setup vector store for Q&A
                if full_text:
                    qa_context = ai_analysis.get("qa_context", "")
                    combined_docs = [full_text, qa_context]

                    vectorstore = setup_vectorstore(combined_docs)
                    if vectorstore:
                        st.session_state.vectorstore = vectorstore
                        conversation_chain = create_conversation_chain(vectorstore)
                        if conversation_chain:
                            st.session_state.conversation_chain = conversation_chain

                # Display Chat Interface
                display_chat_interface(ai_analysis.get("qa_context", ""))

            except Exception as e:
                st.error(f"Error processing PDF: {str(e)}")

    # ===== MANUAL INPUT FLOW =====
    st.markdown("---")
    st.markdown("## ✍️ Or Enter Financial Data Manually")

    with st.expander("📝 Manual Financial Data Entry", expanded=not uploaded_pdf):
        col1, col2 = st.columns(2)

        with col1:
            monthly_income = st.number_input("Monthly Income ($)", min_value=0.0, value=5000.0, step=100.0)

            st.markdown("### Expenses")
            housing = st.number_input("Housing/Rent ($)", min_value=0.0, value=1200.0, step=50.0)
            utilities = st.number_input("Utilities ($)", min_value=0.0, value=150.0, step=10.0)
            food = st.number_input("Food/Groceries ($)", min_value=0.0, value=400.0, step=50.0)
            transportation = st.number_input("Transportation ($)", min_value=0.0, value=300.0, step=50.0)

        with col2:
            insurance = st.number_input("Insurance ($)", min_value=0.0, value=200.0, step=50.0)
            entertainment = st.number_input("Entertainment ($)", min_value=0.0, value=200.0, step=50.0)
            other_expenses = st.number_input("Other Expenses ($)", min_value=0.0, value=250.0, step=50.0)

            st.markdown("### Assets & Debts")
            current_savings = st.number_input("Current Savings ($)", min_value=0.0, value=5000.0, step=500.0)
            total_debt = st.number_input("Total Debt ($)", min_value=0.0, value=10000.0, step=500.0)
            investments = st.number_input("Investments ($)", min_value=0.0, value=15000.0, step=1000.0)

        if st.button("🚀 Analyze My Finances", type="primary"):
            # Build financial data
            financial_data = {
                "monthly_income": monthly_income,
                "expenses": {
                    "housing": housing,
                    "utilities": utilities,
                    "food": food,
                    "transportation": transportation,
                    "insurance": insurance,
                    "entertainment": entertainment,
                    "other": other_expenses
                },
                "debts": [total_debt],
                "investments": [investments],
                "savings": current_savings,
                "extracted_from_pdf": False
            }

            # Store in session state
            st.session_state.financial_data = financial_data

            # Generate AI Analysis
            with st.spinner("🤖 Generating AI Analysis..."):
                ai_analysis = generate_comprehensive_ai_analysis(financial_data)
                st.session_state.ai_analysis = ai_analysis

            # Display Digital Report
            display_digital_report(financial_data, ai_analysis)

            # Setup Q&A context
            qa_context = ai_analysis.get("qa_context", "")
            vectorstore = setup_vectorstore([qa_context, json.dumps(financial_data)])
            if vectorstore:
                st.session_state.vectorstore = vectorstore
                conversation_chain = create_conversation_chain(vectorstore)
                if conversation_chain:
                    st.session_state.conversation_chain = conversation_chain

            # Display Chat Interface
            display_chat_interface(qa_context)

    # ===== DISPLAY EXISTING ANALYSIS =====
    if "ai_analysis" in st.session_state and not uploaded_pdf:
        st.markdown("---")
        st.markdown("## 📊 Your Current Financial Analysis")
        display_digital_report(
            st.session_state.get("financial_data", {}),
            st.session_state["ai_analysis"]
        )

        # Display chat if available
        if st.session_state.get("conversation_chain") or groq_api_key:
            display_chat_interface(
                st.session_state["ai_analysis"].get("qa_context", "")
            )

if __name__ == "__main__":
    main()
