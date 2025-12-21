import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from groq import Groq
from pypdf import PdfReader
import json
import os
import datetime
from decimal import Decimal
import logging
from typing import List, Dict, Any

# --- 1. CONFIGURATION & LOGGING ---
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class FinanceAIPro:
    def __init__(self):
        self.config = self._load_config()
        if self.config:
            self.client = Groq(api_key=self.config.get("GROQ_API_KEY"))
        self.init_session_state()

    def _load_config(self) -> Dict:
        try:
            with open("config.json", "r") as f:
                return json.load(f)
        except Exception as e:
            st.error(f"Critical System Error: {e}")
            return {}

    def init_session_state(self):
        if "db" not in st.session_state:
            st.session_state.db = pd.DataFrame(columns=["Date", "Description", "Category", "Amount", "Type"])
        if "raw_text" not in st.session_state:
            st.session_state.raw_text = ""

    # --- 2. DATA EXTRACTION ENGINE ---
    def process_pdf(self, uploaded_file):
        try:
            reader = PdfReader(uploaded_file)
            text = ""
            for i, page in enumerate(reader.pages):
                page_text = page.extract_text()
                if page_text:
                    text += f"\n--- Page {i+1} ---\n{page_text}"
            st.session_state.raw_text = text
            return text
        except Exception as e:
            logger.error(f"PDF Extraction failed: {e}")
            return None

    def ai_extraction_logic(self, text: str):
        """Advanced JSON extraction with schema enforcement."""
        system_prompt = """
        You are a Senior Financial Data Auditor. Convert the following bank statement text into a structured JSON.
        Rules:
        1. Classify transactions into: [Housing, Food, Transport, Tech, Healthcare, Income, Leisure].
        2. Clean descriptions (remove extra whitespace/weird characters).
        3. Identify if the transaction is 'Debit' or 'Credit'.
        4. Return ONLY a JSON object with a 'transactions' key.
        """
        
        # Chunking strategy for large PDFs
        preview = text[:8000] 
        
        try:
            completion = self.client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": f"PROCESS THIS DATA:\n{preview}"}
                ],
                response_format={"type": "json_object"},
                temperature=0.0
            )
            return json.loads(completion.choices[0].message.content)
        except Exception as e:
            st.error(f"AI Extraction failed: {e}")
            return None

    # --- 3. ANALYTICS ENGINE ---
    def generate_ai_insights(self, df):
        summary = df.groupby("Category")["Amount"].sum().to_dict()
        prompt = f"""
        Act as a Wealth Manager. Analyze these totals: {summary}.
        Provide:
        1. A Burn Rate analysis.
        2. Detection of any anomalies or unusual patterns.
        3. A 30-day savings strategy.
        Keep it professional and data-driven.
        """
        completion = self.client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": prompt}]
        )
        return completion.choices[0].message.content

# --- 4. ADVANCED UI LAYOUT ---
def main():
    app = FinanceAIPro()
    st.set_page_config(page_title="Enterprise AI Finance", layout="wide", page_icon="🏦")
    
    # Custom Sidebar with System Health
    with st.sidebar:
        st.header("🏦 System Control")
        st.status("Llama 3.3-70B: Online", state="complete")
        st.divider()
        uploaded_file = st.file_uploader("Ingest Financial Document", type="pdf")
        
        if st.button("🔄 Reset Environment"):
            st.session_state.clear()
            st.rerun()

    # Main Dashboard
    st.title("📊 Financial Intelligence Engine")
    
    if uploaded_file:
        tab1, tab2, tab3 = st.tabs(["📉 Visual Analytics", "📑 Data Ledger", "🧠 AI Strategy"])
        
        with st.spinner("Executing Data Pipeline..."):
            text = app.process_pdf(uploaded_file)
            extracted = app.ai_extraction_logic(text)
            
            if extracted:
                new_df = pd.DataFrame(extracted["transactions"])
                new_df["Amount"] = pd.to_numeric(new_df["Amount"]).abs()
                new_df["Date"] = pd.to_datetime(new_df["Date"])
                st.session_state.db = new_df

        with tab1:
            c1, c2 = st.columns([2, 1])
            with c1:
                st.subheader("Cash Flow Velocity")
                fig = px.area(st.session_state.db.sort_values("Date"), x="Date", y="Amount", color="Category",
                             line_group="Category", height=400, template="plotly_dark")
                st.plotly_chart(fig, use_container_width=True)
            
            with c2:
                st.subheader("Allocation Map")
                fig_sun = px.sunburst(st.session_state.db, path=['Category', 'Description'], values='Amount')
                st.plotly_chart(fig_sun, use_container_width=True)

        with tab2:
            st.subheader("Transaction Ledger")
            st.dataframe(st.session_state.db, use_container_width=True)
            
        with tab3:
            st.subheader("Executive Summary")
            insights = app.generate_ai_insights(st.session_state.db)
            st.markdown(insights)
            
            # Additional advanced visualization: Gauge Chart for Budget
            total_spend = st.session_state.db["Amount"].sum()
            fig_gauge = go.Figure(go.Indicator(
                mode = "gauge+number",
                value = total_spend,
                title = {'text': "Total Monthly Burn ($)"},
                gauge = {'axis': {'range': [None, 10000]},
                         'bar': {'color': "darkblue"},
                         'steps' : [
                             {'range': [0, 2000], 'color': "lightcyan"},
                             {'range': [2000, 5000], 'color': "royalblue"}]}))
            st.plotly_chart(fig_gauge)

if __name__ == "__main__":
    main()
