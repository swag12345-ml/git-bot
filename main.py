# main.py
import streamlit as st  # Streamlit must be imported first
import os
import json
import torch
import asyncio
from dotenv import load_dotenv
import fitz  # PyMuPDF for text extraction
import easyocr  # GPU-accelerated OCR
from pdf2image import convert_from_path  # Convert PDFs to images
from langchain_text_splitters import CharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain, LLMChain
from langchain import PromptTemplate
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import re
import io
from typing import List, Dict

# ------------------------ Basic config ------------------------
st.set_page_config(page_title="Chat with Swag AI", page_icon="📝", layout="centered")
load_dotenv()
working_dir = os.path.dirname(os.path.abspath(__file__))

# Ensure GPU availability
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
torch.backends.cudnn.benchmark = True  # Optimize GPU performance

# ------------------------ Load GROQ API Key ------------------------
def load_groq_api_key():
    """Loads the GROQ API key from config.json"""
    try:
        with open(os.path.join(working_dir, "config.json"), "r") as f:
            return json.load(f).get("GROQ_API_KEY")
    except FileNotFoundError:
        st.error("🚨 config.json not found. Please add your GROQ API key.")
        st.stop()

groq_api_key = load_groq_api_key()
if not groq_api_key:
    st.error("🚨 GROQ_API_KEY is missing. Check your config.json file.")
    st.stop()

# ------------------------ OCR Reader ------------------------
reader = easyocr.Reader(["en"], gpu=torch.cuda.is_available())

# ------------------------ Utilities: PDF text extraction ------------------------
def extract_text_from_pdf(file_path: str) -> List[str]:
    """Extracts text from PDFs using PyMuPDF, falls back to OCR if needed."""
    try:
        doc = fitz.open(file_path)
        text_list = []
        for page in doc:
            page_text = page.get_text("text")
            if page_text and page_text.strip():
                text_list.append(page_text)
        doc.close()
        if text_list:
            return text_list
        # fallback: images (if scanned)
        return extract_text_from_images(file_path)
    except Exception as e:
        st.warning(f"⚠️ Error extracting text from PDF: {e}")
        return []

def extract_text_from_images(pdf_path: str) -> List[str]:
    """Extracts text from image-based PDFs using EasyOCR."""
    try:
        images = convert_from_path(pdf_path, dpi=150)  # load all pages if necessary
        pages_text = []
        for img in images:
            arr = np.array(img)
            harvested = reader.readtext(arr, detail=0)
            pages_text.append("\n".join(harvested))
        return pages_text
    except Exception as e:
        st.warning(f"⚠️ Error extracting text from images: {e}")
        return []

# ------------------------ Vector store & chains ------------------------
def setup_vectorstore(documents: List[str]):
    """Creates a FAISS vector store using Hugging Face embeddings."""
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    # move to GPU if available
    try:
        if DEVICE == "cuda" and hasattr(embeddings, "model"):
            embeddings.model = embeddings.model.to(torch.device("cuda"))
    except Exception:
        pass

    text_splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    # split input documents into text chunks
    doc_chunks = []
    for d in documents:
        doc_chunks.extend(text_splitter.split_text(d))
    return FAISS.from_texts(doc_chunks, embeddings)

def create_pdf_chain(vectorstore: FAISS):
    """Conversational retrieval chain for PDF-based QA"""
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

# Simple general finance LLM chain (no retriever) with a finance-focused system prompt
def create_general_chain():
    """Creates an LLMChain that responds as a finance assistant (no PDF retrieval)."""
    # system-style template. adapt/expand as needed.
    system_prompt = """You are FinBot, a friendly, practical personal finance assistant.
When asked, provide budgeting help, expense categorization, savings suggestions, and simple computations.
If the user asks for legal, tax, or guaranteed investment returns, say: "This is educational and not financial advice." 
Be concise and, when possible, ask a single clarifying question if critical info is missing."""
    # Use PromptTemplate with an input variable 'user_input' that will be appended to the system-like instructions
    prompt = PromptTemplate(
        input_variables=["user_input"],
        template=system_prompt + "\n\nUser: {user_input}\n\nAssistant:"
    )
    llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0.2, groq_api_key=groq_api_key)
    return LLMChain(llm=llm, prompt=prompt)

# ------------------------ Routing logic ------------------------
def route_query(user_message: str, similarity_threshold: float = 0.55):
    """
    Decide whether to use PDF-QA (if vectorstore exists and retrieval yields results)
    or fallback to general finance conversation.
    """
    # If we have a vectorstore and pdf_chain, try retrieval-based answer
    try:
        if "vectorstore" in st.session_state and st.session_state.vectorstore is not None:
            # create a retriever on the fly to check if there are relevant docs
            retriever = st.session_state.vectorstore.as_retriever(search_kwargs={"k": 4})
            # attempt to get relevant documents (some retrievers return docs only)
            try:
                docs = retriever.get_relevant_documents(user_message)
            except Exception:
                # Some retriever APIs differ; fallback to generic method - attempt calling pdf chain directly
                docs = []
            if docs:
                # We consider any retrieved docs as "relevant" for this simpler implementation.
                # (If your retriever can return similarity scores, check them here and compare to threshold.)
                try:
                    result = st.session_state.conversation_chain({"question": user_message, "chat_history": st.session_state.memory.chat_memory.messages})
                    return result.get("answer", "Sorry — I couldn't produce an answer from the document.")
                except Exception as e:
                    # If pdf chain errors, fallback to general chain with an explanation
                    st.warning("⚠️ PDF chain error; falling back to general finance assistant.")
                    return general_chain.run(user_input=user_message)
    except Exception as e:
        # Any unexpected error in retrieval -> fallback
        st.warning(f"⚠️ Retrieval/route error: {e}")

    # Fallback: general conversation chain
    try:
        return general_chain.run(user_input=user_message)
    except Exception as e:
        return f"⚠️ Error in general conversation chain: {e}"

# ------------------------ Expense extraction & visualization ------------------------
def naive_parse_expense_lines(text: str) -> List[Dict]:
    """
    Very simple parser to identify lines with dates and amounts.
    This is intentionally conservative — you'll likely want to replace it with a better parser for your
    bank statement format.
    Returns list of dicts: {"date": "YYYY-MM-DD", "category": "Unknown", "amount": float, "description": str}
    """
    expenses = []
    # Look for patterns like YYYY-MM-DD and amounts like 1,234.56 or 1234 or ₹1,234 or 1234.00 etc.
    date_pattern = r"(\d{4}[-/]\d{2}[-/]\d{2})"
    amount_pattern = r"([₹$]?\s?[\d,]+(?:\.\d{1,2})?)"
    lines = text.splitlines()
    for line in lines:
        d_match = re.search(date_pattern, line)
        a_match = re.search(amount_pattern, line)
        if a_match:
            amount_raw = a_match.group(1)
            # remove currency symbols and commas
            cleaned = re.sub(r"[^\d\.]", "", amount_raw)
            try:
                amount = float(cleaned) if cleaned else 0.0
            except:
                amount = 0.0
            date = d_match.group(1) if d_match else None
            # attempt a very small heuristic for category from line keywords
            lower = line.lower()
            if any(k in lower for k in ["grocery", "groceries", "supermarket", "market"]):
                cat = "Groceries"
            elif any(k in lower for k in ["rent", "apartment", "lease"]):
                cat = "Rent"
            elif any(k in lower for k in ["uber", "ola", "taxi", "transport", "bus", "metro"]):
                cat = "Transport"
            elif any(k in lower for k in ["dining", "restaurant", "cafe", "coffee", "eat"]):
                cat = "Dining"
            else:
                cat = "Other"
            expenses.append({
                "date": date or "",
                "category": cat,
                "amount": amount,
                "description": line.strip()
            })
    return expenses

def add_expense_to_session(expense: Dict):
    if "expenses" not in st.session_state:
        st.session_state["expenses"] = []
    st.session_state["expenses"].append(expense)

def summarize_and_plot_expenses(expenses: List[Dict]):
    df = pd.DataFrame(expenses)
    if df.empty:
        st.info("No expenses to show.")
        return None
    # normalize date
    if "date" in df.columns:
        try:
            df['date'] = pd.to_datetime(df['date'], errors='coerce')
        except:
            df['date'] = pd.NaT
    df['amount'] = pd.to_numeric(df['amount'], errors='coerce').fillna(0.0)
    # category totals
    category_totals = df.groupby('category')['amount'].sum().sort_values(ascending=False).reset_index()
    # monthly totals if date present
    if 'date' in df.columns and df['date'].notna().any():
        df['month'] = df['date'].dt.to_period('M')
        monthly = df.groupby('month')['amount'].sum().reset_index()
    else:
        monthly = pd.DataFrame()
    # Produce a matplotlib figure for categories
    fig, ax = plt.subplots(figsize=(6,4))
    ax.bar(category_totals['category'], category_totals['amount'])
    ax.set_title('Expenses by Category')
    ax.set_xlabel('Category')
    ax.set_ylabel('Amount')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    return {
        "df": df,
        "category_totals": category_totals,
        "monthly": monthly,
        "fig": fig
    }

# ------------------------ UI: App layout ------------------------
st.title("🦙 Chat with Swag AI — Financial Mode + PDF-QA (LLAMA 3.3)")

# initialize memory, expenses, etc
if "memory" not in st.session_state:
    st.session_state.memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "expenses" not in st.session_state:
    st.session_state.expenses = []

# prepare general chain (always available)
general_chain = create_general_chain()

# file uploader
uploaded_files = st.file_uploader("Upload PDF files (bank statements, receipts). Leave empty for general conversation.", type=["pdf"], accept_multiple_files=True)

# handle uploaded files
if uploaded_files:
    all_text = []
    for uploaded_file in uploaded_files:
        save_path = os.path.join(working_dir, uploaded_file.name)
        # save uploaded file
        try:
            with open(save_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            st.success(f"Saved {uploaded_file.name}")
        except Exception as e:
            st.warning(f"Could not save {uploaded_file.name}: {e}")
            continue

        # extract text
        texts = extract_text_from_pdf(save_path)
        if texts:
            all_text.extend(texts)
            st.info(f"Extracted text from {uploaded_file.name} (pages: {len(texts)})")
        else:
            st.warning(f"No text found in {uploaded_file.name}")

        # small naive auto-expense parse: attempt to extract some expense lines from the text and add them to session
        try:
            for page_text in texts:
                expenses_found = naive_parse_expense_lines(page_text)
                for ex in expenses_found:
                    add_expense_to_session(ex)
            if expenses_found:
                st.success(f"Auto-extracted {len(expenses_found)} expense-like lines from {uploaded_file.name}. Check 'Quick Actions' -> 'Visualize Expenses'.")
        except Exception as e:
            st.warning(f"Expense parsing error for {uploaded_file.name}: {e}")

    # if any extracted text, create/update vectorstore + pdf chain
    if all_text:
        try:
            st.session_state.vectorstore = setup_vectorstore(all_text)
            st.session_state.conversation_chain = create_pdf_chain(st.session_state.vectorstore)
            st.success("✅ Vector store and PDF conversation chain ready.")
        except Exception as e:
            st.warning(f"Could not setup vectorstore/chain: {e}")

# Quick-actions sidebar
st.sidebar.header("Quick Actions")
if st.sidebar.button("Create Budget (guided)"):
    # very small guided flow to create a simple 50/30/20 style budget
    st.session_state["flow"] = "create_budget"
    st.experimental_rerun()

if st.sidebar.button("Add Expense"):
    st.session_state["flow"] = "add_expense"
    st.experimental_rerun()

if st.sidebar.button("Visualize Expenses"):
    st.session_state["flow"] = "visualize_expenses"
    st.experimental_rerun()

# flow handlers
flow = st.session_state.get("flow", None)
if flow == "create_budget":
    st.subheader("Create Budget — quick 50/30/20 suggestion")
    monthly_income = st.number_input("Monthly take-home income (numbers only)", min_value=0.0, value=50000.0)
    if st.button("Generate Budget"):
        needs = monthly_income * 0.5
        wants = monthly_income * 0.3
        savings = monthly_income * 0.2
        st.success(f"Suggested budget — Needs: ₹{needs:,.2f}, Wants: ₹{wants:,.2f}, Savings: ₹{savings:,.2f}")
        st.info("This is an educational suggestion, not financial advice.")
    if st.button("Done"):
        st.session_state.pop("flow", None)
        st.experimental_rerun()

elif flow == "add_expense":
    st.subheader("Add Expense")
    e_date = st.text_input("Date (YYYY-MM-DD or leave blank)")
    e_cat = st.selectbox("Category", ["Groceries", "Rent", "Transport", "Dining", "Other"])
    e_amt = st.number_input("Amount", min_value=0.0, value=0.0)
    e_desc = st.text_input("Description")
    if st.button("Add"):
        add_expense_to_session({"date": e_date, "category": e_cat, "amount": float(e_amt), "description": e_desc})
        st.success("Expense added.")
    if st.button("Done"):
        st.session_state.pop("flow", None)
        st.experimental_rerun()

elif flow == "visualize_expenses":
    st.subheader("Expense Visualization")
    viz = summarize_and_plot_expenses(st.session_state.get("expenses", []))
    if viz is not None:
        st.pyplot(viz["fig"])
        st.write("Category totals")
        st.dataframe(viz["category_totals"])
        if not viz["monthly"].empty:
            st.write("Monthly totals")
            st.dataframe(viz["monthly"])
        # CSV download
        csv = viz["df"].to_csv(index=False).encode('utf-8')
        st.download_button("Download expense summary CSV", csv, "expenses_summary.csv", "text/csv")
    if st.button("Done"):
        st.session_state.pop("flow", None)
        st.experimental_rerun()

# Chat area
st.markdown("---")
st.header("Chat")

# Display conversation memory (previous messages)
if "memory" in st.session_state:
    # memory stores messages with .load_memory_variables()
    mem = st.session_state.memory.load_memory_variables({}).get("chat_history", [])
    for msg in mem:
        role = "user" if getattr(msg, "type", "").lower() == "human" else "assistant"
        with st.chat_message(role):
            st.markdown(getattr(msg, "content", str(msg)))

# Input
user_input = st.chat_input("Ask FinBot about your finances or uploaded PDFs...")

# Async wrapper similar to your original
async def get_response_async(user_input_text: str):
    # route and compute response in a background thread (not truly background — executed now)
    return await asyncio.to_thread(route_query, user_input_text)

if user_input:
    with st.chat_message("user"):
        st.markdown(user_input)

    # Run routing & response
    try:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        assistant_response = loop.run_until_complete(get_response_async(user_input))
    except Exception as e:
        assistant_response = f"⚠️ Error: {str(e)}"

    # Show assistant response
    with st.chat_message("assistant"):
        st.markdown(assistant_response)

    # Save to memory
    try:
        st.session_state.memory.save_context({"input": user_input}, {"output": assistant_response})
    except Exception:
        # If memory save fails, at least store in lightweight history
        st.session_state.chat_history.append({"user": user_input, "assistant": assistant_response})

# footer tips
st.markdown("---")
st.markdown("**Tips:**\n- Upload PDFs for document-specific answers. \n- Use the sidebar 'Quick Actions' to add expenses or visualize them. \n- This assistant is educational and not financial advice.")

