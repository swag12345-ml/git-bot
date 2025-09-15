# finchat_fixed.py
# Fixed and hardened version of FinChat (Streamlit + optional LLaMA/Groq/FAISS)
# - Safe fallbacks when optional libraries are missing
# - Keeps conversational flows, personas, and PDF QA logic
# - Does not crash when no PDF is uploaded
# - Clear places to add your own Groq/LLaMA + LangChain init

import streamlit as st
import os
import json
import datetime
import tempfile
from typing import List

# Data libs
import numpy as np
import pandas as pd

# plotting
import matplotlib.pyplot as plt

# Optional heavy libs — imported safely
_has_fitz = False
_has_easyocr = False
_has_pdf2image = False
_has_langchain = False
_has_faiss = False
_has_huggingface_embeddings = False
_has_chrogq = False

try:
    import fitz  # PyMuPDF
    _has_fitz = True
except Exception:
    fitz = None

try:
    import easyocr
    _has_easyocr = True
except Exception:
    easyocr = None

try:
    from pdf2image import convert_from_path  # type: ignore
    _has_pdf2image = True
except Exception:
    convert_from_path = None

# LangChain / Vectorstore / Groq imports (optional)
try:
    from langchain_text_splitters import CharacterTextSplitter  # type: ignore
    from langchain_community.vectorstores import FAISS  # type: ignore
    from langchain_huggingface import HuggingFaceEmbeddings  # type: ignore
    from langchain_groq import ChatGroq  # type: ignore
    from langchain.memory import ConversationBufferMemory  # type: ignore
    from langchain.chains import ConversationalRetrievalChain  # type: ignore
    _has_langchain = True
    _has_faiss = True
    _has_huggingface_embeddings = True
    _has_chrogq = True
except Exception:
    # Keep variables defined for clarity
    CharacterTextSplitter = None
    FAISS = None
    HuggingFaceEmbeddings = None
    ChatGroq = None
    ConversationBufferMemory = None
    ConversationalRetrievalChain = None

# -------------------- Basic config & env --------------------
st.set_page_config(page_title="Chat with Swag AI — FinBot (Fixed)", page_icon="💼", layout="wide")

# Load config.json if present (for GROQ API key)
working_dir = os.path.dirname(os.path.abspath(__file__))

def load_groq_api_key() -> str:
    cfg_path = os.path.join(working_dir, "config.json")
    if os.path.exists(cfg_path):
        try:
            with open(cfg_path, "r") as f:
                return json.load(f).get("GROQ_API_KEY", "") or ""
        except Exception:
            return ""
    return ""

groq_api_key = load_groq_api_key()

# -------------------- Lightweight fallback LLM --------------------
def init_llm_fallback():
    """Return a simple fallback LLM-like callable: llm(system_prompt, user_prompt, chat_history) -> str"""
    def llm(system_prompt: str, user_prompt: str, chat_history: List[tuple]) -> str:
        # Very small heuristic-based replies to make the app useful without real LLM
        q = user_prompt.lower()
        if "expense" in q or "budget" in q or "spend" in q or "saving" in q:
            return (
                "I can help with expenses and budgeting. "
                "Use the Expense Manager in the sidebar to add transactions or upload a CSV with columns (date, category, amount). "
                "Then ask for a summary like: 'What's my largest spending category this month?'"
            )
        if any(k in q for k in ["stock", "trade", "portfolio", "ticker", "share"]):
            return (
                "I can explain stock concepts and simulate portfolio charts. "
                "To analyze real historical prices upload a CSV with columns (date,ticker,price). "
                "I cannot provide personalized investment advice."
            )
        if "pdf" in q or "document" in q or "statement" in q:
            return (
                "I can answer questions about uploaded PDFs when you upload them. "
                "Upload PDFs in the sidebar to enable Document QA."
            )
        if "tax" in q or "legal" in q:
            return (
                "I cannot provide legal or tax advice. For specific tax or legal questions consult a licensed professional. "
                "I can provide high-level educational information instead."
            )
        # default small reply
        return "I'm ready to help — ask about budgeting, expenses, or upload documents for Document QA."
    return llm

llm = init_llm_fallback()

# -------------------- PDF / OCR utilities (safe) --------------------
# Use EasyOCR reader only if easyocr available
_easyocr_reader = None
if _has_easyocr:
    try:
        # Let EasyOCR decide GPU usage automatically; avoid referencing torch directly here.
        _easyocr_reader = easyocr.Reader(["en"], gpu=False)  # Use CPU by default for portability
    except Exception:
        _easyocr_reader = None

def extract_text_from_pdf_path(file_path: str) -> List[str]:
    """
    Safely extract text from a PDF file path.
    Returns a list of page-text strings (or empty list).
    """
    out = []
    if _has_fitz:
        try:
            doc = fitz.open(file_path)
            for p_idx, page in enumerate(doc):
                text = page.get_text("text") or ""
                if text.strip():
                    out.append(f"[page {p_idx+1}] " + text.strip())
            doc.close()
            if out:
                return out
        except Exception as e:
            st.warning(f"PyMuPDF read error for `{os.path.basename(file_path)}`: {e}")
            out = []
    # Fallback to OCR if possible
    if _has_pdf2image and _easyocr_reader is not None:
        try:
            images = convert_from_path(file_path, dpi=150, first_page=1, last_page=5)
            for idx, img in enumerate(images):
                arr = np.array(img)
                try:
                    lines = _easyocr_reader.readtext(arr, detail=0)
                    out.append(f"[page {idx+1}] " + "\n".join(lines))
                except Exception:
                    out.append(f"[page {idx+1}] (OCR failed on this page)")
            return out
        except Exception as e:
            st.warning(f"OCR fallback failed for `{os.path.basename(file_path)}`: {e}")
    # If nothing extracted, return empty list
    return []

# -------------------- Vectorstore & Chain (optional) --------------------
def setup_vectorstore_safe(documents: List[str]):
    """
    Create a FAISS vectorstore if all related libs are available.
    If not available, raise a clear exception for the caller to handle.
    """
    if not (_has_faiss and _has_huggingface_embeddings and CharacterTextSplitter is not None):
        raise RuntimeError("FAISS + HuggingFaceEmbeddings or text splitter not available in this environment.")
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    # Attempt to split text
    splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    text = "\n".join(documents)
    chunks = splitter.split_text(text)
    if not chunks:
        raise ValueError("No text chunks produced for vectorstore.")
    return FAISS.from_texts(chunks, embeddings)

def create_chain_safe(vectorstore):
    """Create ConversationalRetrievalChain using ChatGroq if available."""
    if not _has_chrogq or ConversationBufferMemory is None or ConversationalRetrievalChain is None:
        raise RuntimeError("LangChain ChatGroq/ConversationChain not available.")
    if "memory" not in st.session_state or st.session_state.get("memory") is None:
        st.session_state.memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    # Create LLM client
    try:
        llm_client = ChatGroq(model="llama-3.3-70b-versatile", temperature=0.0, groq_api_key=groq_api_key or None)
    except Exception as e:
        raise RuntimeError(f"Failed to initialize ChatGroq LLM: {e}")
    retriever = vectorstore.as_retriever(search_kwargs={"k": 5})
    chain = ConversationalRetrievalChain.from_llm(
        llm=llm_client,
        retriever=retriever,
        chain_type="stuff",
        memory=st.session_state.memory,
        verbose=False,
    )
    return chain

# -------------------- Prompts / flows --------------------
SYSTEM_PROMPT = (
    "You are FinBot, an empathetic, accurate, and practical financial assistant.\n"
    "Provide clear explanations and step-by-step plans. Ask clarifying questions when needed.\n"
    "Do not provide definitive legal/tax/investment advice—recommend a professional when appropriate.\n"
    "Be concise and use bullets for action steps."
)

PROMPT_TEMPLATES = {
    "General": "You are a helpful financial assistant. User: {user_input}",
    "Budgeting": "You are a budgeting coach. Provide a simple monthly budget template and next steps. User: {user_input}",
    "Investing": "You are an investing guide. Ask for horizon, risk tolerance and current portfolio, then suggest allocation buckets. User: {user_input}",
    "Debt Repayment": "You are a debt counselor. Prioritize debts and suggest avalanche/snowball plans. User: {user_input}",
    "Taxes": "You are a tax-aware assistant offering high-level tips and red flags (recommend professional advice). User: {user_input}",
    "Retirement": "You are a retirement planner. Ask savings, retirement age, and target income; provide simple projection steps. User: {user_input}",
    "Document QA": "You are a document analyst. Summarize PDF content and answer the user's question based on the uploaded PDFs.\nPDF_TEXT: {pdf_text}\nUser question: {user_input}"
}

CONVERSATIONAL_FLOWS = {
    "Budgeting": [
        "What's your monthly take-home income (after taxes)?",
        "List major monthly expenses and approximate amounts (rent, groceries, transport, loans).",
        "Do you have short-term savings goals (0-2 years)? If so, name them and estimated cost.",
    ],
    "Investing": [
        "What's your investment horizon in years?",
        "What's your risk tolerance: conservative, moderate, or aggressive?",
        "How much can you invest monthly or as a lump sum?",
    ],
    "Debt Repayment": [
        "List each debt with its balance, interest rate, and minimum monthly payment.",
        "Do you have any savings/emergency fund? Which payoff method do you prefer: snowball or avalanche?",
    ],
    "Taxes": [
        "Which country & tax year is this for?",
        "What are your income sources (salary, capital gains, self-employment)?",
    ],
    "Retirement": [
        "What's your current retirement savings (amount)?",
        "At what age would you like to retire?",
        "What annual income would you like in retirement (in today's money)?",
    ],
}

# -------------------- UI and session state init --------------------
st.title("🦙 Chat with Swag AI — FinBot (Fixed)")
st.write("Upload financial PDFs to enable Document QA. Choose a persona and start a guided flow or ask directly.")

with st.sidebar:
    st.header("Settings")
    persona = st.selectbox("Assistant persona", ["Practical & Direct", "Friendly Coach", "Conservative Advisor"])
    flow_choice = st.selectbox(
        "Start with a flow",
        ["General", "Budgeting", "Investing", "Debt Repayment", "Taxes", "Retirement", "Document QA"],
    )
    st.markdown("---")
    st.subheader("Upload PDFs (optional)")
    uploaded_files = st.file_uploader("Upload PDF files", type=["pdf"], accept_multiple_files=True)
    st.markdown(" ")
    st.caption("Files will be extracted and optionally indexed for Document QA if FAISS + LangChain are installed.")

# session state defaults
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []  # list of tuples (role, content, timestamp)
if "pdf_texts" not in st.session_state:
    st.session_state.pdf_texts = ""
if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None
if "conversation_chain" not in st.session_state:
    st.session_state.conversation_chain = None
if "current_flow" not in st.session_state:
    st.session_state.current_flow = None
if "flow_step" not in st.session_state:
    st.session_state.flow_step = 0
if "index_status" not in st.session_state:
    st.session_state.index_status = "No index"
if "memory" not in st.session_state:
    st.session_state.memory = None  # will be set if LangChain available

# -------------------- Handle uploaded PDFs and optional indexing --------------------
if uploaded_files:
    all_extracted_text = []
    st.info(f"Processing {len(uploaded_files)} file(s)...")
    for uploaded_file in uploaded_files:
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                tmp_path = tmp.name
                tmp.write(uploaded_file.getbuffer())
            extracted = extract_text_from_pdf_path(tmp_path)
            if extracted:
                all_extracted_text.extend(extracted)
                st.success(f"Extracted text from `{uploaded_file.name}`")
            else:
                st.warning(f"No text found in `{uploaded_file.name}`. OCR attempted or file empty.")
        except Exception as e:
            st.error(f"Error processing `{uploaded_file.name}`: {e}")

    if all_extracted_text:
        st.session_state.pdf_texts = "\n\n".join(all_extracted_text)
        # Attempt to create vectorstore & chain if LangChain & FAISS available
        if _has_langchain and _has_faiss and _has_huggingface_embeddings and _has_chrogq:
            try:
                st.session_state.index_status = "Indexing..."
                with st.spinner("Creating embeddings and FAISS index (this may take a while)..."):
                    st.session_state.vectorstore = setup_vectorstore_safe(all_extracted_text)
                    st.session_state.conversation_chain = create_chain_safe(st.session_state.vectorstore)
                    st.session_state.index_status = "Index ready"
                    st.success("Document index ready for Document QA.")
            except Exception as e:
                st.session_state.index_status = "Index failed"
                st.error(f"Failed to create vectorstore / chain: {e}")
                # leave pdf_texts available for prompt-only Document QA fallback
        else:
            st.session_state.index_status = "Index unavailable (missing libs)"
            st.info("Documents extracted but vectorstore/chain unavailable because required libraries are not installed.")
    else:
        st.session_state.index_status = "No text extracted"

# Top info
st.write(f"**Index status:** {st.session_state.index_status}   |   **Current flow:** {st.session_state.current_flow or 'None'}")

# -------------------- Chat rendering --------------------
chat_col, control_col = st.columns([3, 1])
with chat_col:
    st.subheader("Conversation")
    # show history
    for entry in st.session_state.chat_history:
        # entries may be tuples of length 2 or 3
        if len(entry) == 3:
            role, content, ts = entry
        else:
            role, content = entry
            ts = None
        if role == "user":
            with st.chat_message("user"):
                st.markdown(content)
        else:
            with st.chat_message("assistant"):
                st.markdown(content)

    # input
    try:
        user_input = st.chat_input("Ask FinBot a question...")
    except Exception:
        # older Streamlit fallback
        user_input = st.text_input("Ask FinBot a question...")

# -------------------- Control panel: flows --------------------
with control_col:
    st.subheader("Quick Flows")
    st.write("Start a guided flow to gather structured info.")
    if flow_choice != "General" and st.button("Start selected flow"):
        st.session_state.current_flow = flow_choice
        st.session_state.flow_step = 0
        starter_q = CONVERSATIONAL_FLOWS.get(flow_choice, [f"Starting {flow_choice}."])[0]
        st.session_state.chat_history.append(("assistant", f"Starting **{flow_choice}** flow. {starter_q}", datetime.datetime.now()))
        st.rerun()

    if st.session_state.current_flow:
        cf = st.session_state.current_flow
        steps = CONVERSATIONAL_FLOWS.get(cf, [])
        step_idx = st.session_state.flow_step or 0
        st.markdown(f"**{cf} — step {step_idx+1} of {len(steps)}**")
        step_q = steps[step_idx] if step_idx < len(steps) else None
        if step_q:
            answer = st.text_input(step_q, key=f"flow_input_{step_idx}")
            if st.button("Submit step answer"):
                if not answer.strip():
                    st.warning("Please type an answer before submitting the step.")
                else:
                    st.session_state.chat_history.append(("user", answer, datetime.datetime.now()))
                    if cf == "Document QA":
                        user_prompt = PROMPT_TEMPLATES["Document QA"].format(pdf_text=st.session_state.pdf_texts or "(no PDF text indexed)", user_input=answer)
                    else:
                        user_prompt = PROMPT_TEMPLATES.get(cf, PROMPT_TEMPLATES["General"]).format(user_input=answer)
                    system_context = SYSTEM_PROMPT + f"\nPersona: {persona}."
                    final_question = f"[SYSTEM]\n{system_context}\n[USER]\n{user_prompt}"

                    # Use conversation_chain if available
                    if st.session_state.conversation_chain:
                        try:
                            with st.spinner("FinBot is thinking..."):
                                # ConversationalRetrievalChain interface varies; attempt safe call
                                try:
                                    result = st.session_state.conversation_chain({"question": final_question, "chat_history": st.session_state.memory.chat_memory.messages})
                                    assistant_text = result.get("answer") or result.get("result") or str(result)
                                except Exception:
                                    # Another common signature returns a dict with "answer"
                                    result = st.session_state.conversation_chain.run(final_question)
                                    assistant_text = result or "(chain returned empty)"
                        except Exception as e:
                            assistant_text = f"⚠️ Error invoking chain: {e}"
                    else:
                        # fallback LLM (non-LangChain)
                        assistant_text = llm(system_context, user_prompt, st.session_state.chat_history)

                    st.session_state.chat_history.append(("assistant", assistant_text, datetime.datetime.now()))
                    st.session_state.flow_step = step_idx + 1
                    if st.session_state.flow_step >= len(steps):
                        st.session_state.chat_history.append(("assistant", f"{cf} flow complete. Ask follow-up questions anytime.", datetime.datetime.now()))
                        st.session_state.current_flow = None
                        st.session_state.flow_step = 0
                    st.rerun()
        else:
            st.write("Flow complete.")
    st.markdown("---")
    if st.button("Clear conversation"):
        st.session_state.chat_history = []
        if "memory" in st.session_state:
            st.session_state.memory = None
        st.experimental_rerun()

# -------------------- Main chat logic (direct user_input) --------------------
def call_llm_for_prompt(prompt_text: str) -> str:
    """Try to use conversation_chain if available; else try a direct ChatGroq call; else fallback."""
    system_context = SYSTEM_PROMPT + f"\nPersona: {persona}."
    final_question = f"[SYSTEM]\n{system_context}\n[USER]\n{prompt_text}"

    # 1) LangChain chain
    if st.session_state.conversation_chain:
        try:
            with st.spinner("FinBot is crafting a response..."):
                try:
                    result = st.session_state.conversation_chain({"question": final_question, "chat_history": st.session_state.memory.chat_memory.messages})
                    return result.get("answer") or result.get("result") or str(result)
                except Exception:
                    # Try alternative interface
                    result = st.session_state.conversation_chain.run(final_question)
                    return result or "(chain returned empty)"
        except Exception as e:
            return f"⚠️ Error invoking conversation chain: {e}"

    # 2) Try direct ChatGroq if available
    if _has_chrogq and groq_api_key:
        try:
            llm_client = ChatGroq(model="llama-3.3-70b-versatile", temperature=0.0, groq_api_key=groq_api_key)
            try:
                # LangChain wrapper behavior may vary; attempt a safe .generate-style call
                gen = llm_client.generate([{"role":"system","content": system_context},{"role":"user","content": prompt_text}])
                if hasattr(gen, "generations"):
                    return gen.generations[0][0].text
                return str(gen)
            except Exception:
                # As a last attempt, call the llm as a function
                try:
                    return llm_client(system_context + "\n" + prompt_text)
                except Exception as e2:
                    return f"(Direct ChatGroq call failed: {e2})"
        except Exception as e:
            return f"(Failed to init ChatGroq: {e})"

    # 3) fallback small llm
    return llm(system_context, prompt_text, st.session_state.chat_history)


if 'user_input' not in locals():
    user_input = None

if user_input:
    # save user message
    st.session_state.chat_history.append(("user", user_input, datetime.datetime.now()))

    # Build prompt depending on flow_choice
    if flow_choice == "Document QA":
        prompt = PROMPT_TEMPLATES["Document QA"].format(pdf_text=st.session_state.pdf_texts or "(no PDF text indexed)", user_input=user_input)
    else:
        prompt = PROMPT_TEMPLATES.get(flow_choice, PROMPT_TEMPLATES["General"]).format(user_input=user_input)

    # call LLM / chain safely
    assistant_response = call_llm_for_prompt(prompt)

    # save assistant response
    st.session_state.chat_history.append(("assistant", assistant_response, datetime.datetime.now()))

    # try to save to memory if available
    if "memory" in st.session_state and st.session_state.memory is not None:
        try:
            # ConversationBufferMemory implementations vary; try save_context if present
            st.session_state.memory.save_context({"input": user_input}, {"output": assistant_response})
        except Exception:
            pass

    # rerun to show freshly appended messages
    st.rerun()

# -------------------- (Optional) Analytics / Visualizations area --------------------
st.markdown("---")
st.header("Optional Analytics")
st.write("You can add expense & portfolio visualizations here. This version focuses on safe LLM / Document QA behavior.")
# Simple example: if user uploaded PDFs, show number of characters extracted
if st.session_state.pdf_texts:
    st.subheader("Document summary")
    st.write(f"Extracted text length: {len(st.session_state.pdf_texts)} characters")
    # show a short preview
    st.text(st.session_state.pdf_texts[:1000] + ("..." if len(st.session_state.pdf_texts) > 1000 else ""))

# Developer guidance
st.markdown("---")
st.subheader("Developer notes / next steps")
st.markdown(
    "- To enable full LLM-powered replies, install and configure LangChain + langchain_groq (ChatGroq) and provide `config.json` with `GROQ_API_KEY`.\n"
    "- To enable Document QA indexing, install `langchain_huggingface`, `langchain_community`, `faiss` and ensure `CharacterTextSplitter` is available.\n"
    "- To enable OCR fallback, install `easyocr` and `pdf2image` (plus poppler for pdf2image). For production GPUs, configure the reader accordingly.\n"
    "- If you paste your working `ChatGroq` initialization snippet and LangChain versions, I can integrate them into `create_chain_safe` and the direct-call section."
)




