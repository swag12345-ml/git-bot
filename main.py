# FinChat (modified) — Streamlit + LLaMA (Groq) conversational financial assistant
# Replaces user's original script and adds conversational flows, personas, and system prompts.
# Keep your existing environment (config.json, GPU, LangChain/Groq setup).

import streamlit as st  # Streamlit must be imported first
import os
import json
import torch
from dotenv import load_dotenv
import fitz  # PyMuPDF
import easyocr
from pdf2image import convert_from_path
from langchain_text_splitters import CharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
import numpy as np
import tempfile
import datetime

# -------------------- Basic config & env --------------------
st.set_page_config(page_title="Chat with Swag AI — FinChat", page_icon="💼", layout="wide")
load_dotenv()
working_dir = os.path.dirname(os.path.abspath(__file__))

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
torch.backends.cudnn.benchmark = True

# -------------------- Load GROQ API key --------------------
def load_groq_api_key():
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

# -------------------- OCR / PDF utilities --------------------
reader = easyocr.Reader(["en"], gpu=torch.cuda.is_available())

def extract_text_from_pdf(file_path):
    """Extracts text from PDFs using PyMuPDF; if pages empty, fall back to OCR images extraction."""
    try:
        doc = fitz.open(file_path)
        text_list = []
        for p_idx, page in enumerate(doc):
            text = page.get_text("text") or ""
            if text.strip():
                text_list.append(f"[page {p_idx+1}] " + text)
        doc.close()
        if text_list:
            return text_list
        # otherwise fallback
        return extract_text_from_images(file_path)
    except Exception as e:
        st.error(f"⚠️ Error extracting text from PDF `{os.path.basename(file_path)}`: {e}")
        return []

def extract_text_from_images(pdf_path, max_pages=5):
    """Extract text from first `max_pages` pages using pdf2image + EasyOCR."""
    try:
        images = convert_from_path(pdf_path, dpi=150, first_page=1, last_page=max_pages)
        text_pages = []
        for idx, img in enumerate(images):
            arr = np.array(img)
            try:
                lines = reader.readtext(arr, detail=0)
                text_pages.append(f"[page {idx+1}] " + "\n".join(lines))
            except Exception:
                text_pages.append(f"[page {idx+1}] (OCR failed on this page)")
        return text_pages
    except Exception as e:
        st.error(f"⚠️ Error extracting images for OCR from `{os.path.basename(pdf_path)}`: {e}")
        return []

# -------------------- Vectorstore & chain setup --------------------
def setup_vectorstore(documents):
    """Create FAISS index from list of document text blocks."""
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    # move to GPU if available (attempt)
    try:
        if DEVICE == "cuda" and hasattr(embeddings, "model"):
            embeddings.model = embeddings.model.to(torch.device("cuda"))
    except Exception:
        # non-fatal; embeddings will run on CPU if can't move
        pass

    text_splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    doc_chunks = text_splitter.split_text("\n".join(documents))
    if not doc_chunks:
        raise ValueError("No text chunks available to create vectorstore.")
    return FAISS.from_texts(doc_chunks, embeddings)

def create_chain(vectorstore):
    """Return a ConversationalRetrievalChain backed by ChatGroq."""
    if "memory" not in st.session_state:
        st.session_state.memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    # Configure your Groq LLM here (model name may vary)
    llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0.0, groq_api_key=groq_api_key)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 5})
    chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        chain_type="stuff",
        memory=st.session_state.memory,
        verbose=False,
    )
    return chain

# -------------------- FinBot system prompt, templates, flows --------------------
SYSTEM_PROMPT = """You are FinBot, an empathetic, accurate, and practical financial assistant.
You provide clear explanations, step-by-step plans, and ask clarifying questions when needed.
You must not present legal or tax advice as definitive; recommend a professional when appropriate.
Be concise and use bullets when giving action steps.
"""

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

# -------------------- Page layout --------------------
st.title("🦙 Chat with Swag AI — FinBot (Financial Assistant)")
st.write("Upload financial PDFs to enable Document QA. Choose a persona and start a guided flow or ask a question directly.")

# Sidebar controls
with st.sidebar:
    st.header("Settings")
    persona = st.selectbox("Assistant persona", ["Practical & Direct", "Friendly Coach", "Conservative Advisor"])
    flow_choice = st.selectbox("Start with a flow", ["General", "Budgeting", "Investing", "Debt Repayment", "Taxes", "Retirement", "Document QA"])
    st.markdown("---")
    st.subheader("Upload PDFs (optional)")
    uploaded_files = st.file_uploader("Upload PDF files", type=["pdf"], accept_multiple_files=True)
    st.markdown(" ")
    st.caption("Files will be extracted and indexed for Document QA. Indexing may take time depending on file size and GPU/CPU.")

# Initialize session state
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []  # list of (role, content, timestamp)
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

# -------------------- Handle uploaded PDFs and indexing --------------------
if uploaded_files:
    # Use temporary paths to avoid accidental root directory writes
    all_extracted_text = []
    st.info(f"Processing {len(uploaded_files)} file(s)...")
    for uploaded_file in uploaded_files:
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                tmp_path = tmp.name
                tmp.write(uploaded_file.getbuffer())
            extracted_text = extract_text_from_pdf(tmp_path)
            if extracted_text:
                all_extracted_text.extend(extracted_text)
                st.success(f"✅ Extracted text from `{uploaded_file.name}`")
            else:
                st.warning(f"⚠️ No text found in `{uploaded_file.name}` (file may be scanned images). OCR attempted.")
            # cleanup temp file optionally: keep for debugging
        except Exception as e:
            st.error(f"⚠️ Error processing `{uploaded_file.name}`: {e}")

    if all_extracted_text:
        try:
            st.session_state.index_status = "Indexing..."
            with st.spinner("Creating embeddings and FAISS index (this may take a while)..."):
                st.session_state.vectorstore = setup_vectorstore(all_extracted_text)
                st.session_state.conversation_chain = create_chain(st.session_state.vectorstore)
            st.session_state.pdf_texts = "\n\n".join(all_extracted_text)
            st.session_state.index_status = "Index ready"
            st.success("🔎 Document index ready for Document QA.")
        except Exception as e:
            st.session_state.index_status = "Index failed"
            st.error(f"⚠️ Failed to create vectorstore: {e}")

# Top info bar
st.write(f"**Index status:** {st.session_state.index_status}   |   **Current flow:** {st.session_state.current_flow or 'None'}")

# -------------------- Chat history rendering --------------------
chat_col, control_col = st.columns([3, 1])
with chat_col:
    st.subheader("Conversation")
    # Render message history
    for role, content, ts in st.session_state.chat_history:
        if role == "user":
            with st.chat_message("user"):
                st.markdown(content)
        else:
            with st.chat_message("assistant"):
                st.markdown(content)

    # Chat input
    user_input = st.chat_input("Ask FinBot a question...")

# -------------------- Quick Flow controls --------------------
with control_col:
    st.subheader("Quick Flows")
    st.write("Start a guided flow to gather structured info.")
    if flow_choice != "General" and st.button("Start selected flow"):
        st.session_state.current_flow = flow_choice
        st.session_state.flow_step = 0
        # push assistant starter message
        starter_q = CONVERSATIONAL_FLOWS.get(flow_choice, [])[0] if CONVERSATIONAL_FLOWS.get(flow_choice) else f"Starting {flow_choice}."
        st.session_state.chat_history.append(("assistant", f"Starting **{flow_choice}** flow. {starter_q}", datetime.datetime.now()))
        st.experimental_rerun()

    if st.session_state.current_flow:
        cf = st.session_state.current_flow
        steps = CONVERSATIONAL_FLOWS.get(cf, [])
        step_idx = st.session_state.flow_step
        st.markdown(f"**{cf} — step {step_idx+1} of {len(steps)}**")
        step_q = steps[step_idx] if step_idx < len(steps) else None
        if step_q:
            answer = st.text_input(step_q, key=f"flow_input_{step_idx}")
            if st.button("Submit step answer"):
                if answer.strip():
                    # save user answer
                    st.session_state.chat_history.append(("user", answer, datetime.datetime.now()))
                    # Prepare question for chain
                    # Merge flow context into user prompt to give LLM context
                    user_prompt = f"{step_q}\nUser answer: {answer}"
                    # If it's Document QA, include pdf_texts
                    if cf == "Document QA":
                        user_prompt = PROMPT_TEMPLATES["Document QA"].format(pdf_text=st.session_state.pdf_texts, user_input=answer)
                    else:
                        user_prompt = PROMPT_TEMPLATES.get(cf, PROMPT_TEMPLATES["General"]).format(user_input=answer)

                    # Build final question with system prompt and persona hint
                    system_context = SYSTEM_PROMPT + f"\nPersona: {persona}."
                    final_question = f"[SYSTEM]\n{system_context}\n[USER]\n{user_prompt}"

                    # Call the chain
                    if st.session_state.conversation_chain:
                        try:
                            with st.spinner("FinBot is thinking..."):
                                response = st.session_state.conversation_chain(
                                    {"question": final_question, "chat_history": st.session_state.memory.chat_memory.messages}
                                )
                                assistant_text = response.get("answer") or response.get("result") or str(response)
                        except Exception as e:
                            assistant_text = f"⚠️ Error invoking chain: {e}"
                    else:
                        assistant_text = "(No document index / chain found. Provide PDFs and index them for Document QA, or run without Document QA.)"

                    st.session_state.chat_history.append(("assistant", assistant_text, datetime.datetime.now()))
                    st.session_state.flow_step += 1
                    # end flow when steps exhausted
                    if st.session_state.flow_step >= len(steps):
                        st.session_state.chat_history.append(("assistant", f"{cf} flow complete. Ask follow-up questions anytime.", datetime.datetime.now()))
                        st.session_state.current_flow = None
                        st.session_state.flow_step = 0
                    st.experimental_rerun()
                else:
                    st.warning("Please type an answer before submitting the step.")
        else:
            st.write("Flow complete.")
    st.markdown("---")
    if st.button("Clear conversation"):
        st.session_state.chat_history = []
        # keep memory but clear it if you prefer:
        if "memory" in st.session_state:
            st.session_state.memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
        st.experimental_rerun()

# -------------------- Main chat interaction (direct questions) --------------------
if user_input:
    # save user message
    st.session_state.chat_history.append(("user", user_input, datetime.datetime.now()))

    # Build prompt based on selected flow & persona
    if flow_choice == "Document QA":
        # include the extracted PDF texts (if any) in the prompt
        prompt = PROMPT_TEMPLATES["Document QA"].format(pdf_text=st.session_state.pdf_texts or "(no PDF text indexed)", user_input=user_input)
    else:
        prompt = PROMPT_TEMPLATES.get(flow_choice, PROMPT_TEMPLATES["General"]).format(user_input=user_input)

    system_context = SYSTEM_PROMPT + f"\nPersona: {persona}."
    final_question = f"[SYSTEM]\n{system_context}\n[USER]\n{prompt}"

    # If conversation_chain available, call it; else, give an informative fallback
    if st.session_state.conversation_chain:
        try:
            with st.spinner("FinBot is crafting a response..."):
                # ConversationalRetrievalChain expects a dict input with question and chat_history
                response = st.session_state.conversation_chain(
                    {"question": final_question, "chat_history": st.session_state.memory.chat_memory.messages}
                )
                assistant_response = response.get("answer") or response.get("result") or str(response)
        except Exception as e:
            assistant_response = f"⚠️ Error invoking conversation chain: {e}"
    else:
        # Basic fallback if no index: create a small Groq LLM call directly (optional)
        try:
            # Create a direct LLM client only if user hasn't indexed docs; this keeps behavior predictable
            llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0.0, groq_api_key=groq_api_key)
            # We will just call llm directly by prompting: (LangChain ChatGroq exposes `generate` or call method depending on version)
            # Using llm.__call__ style may vary by LangChain version; here we rely on llm to accept a simple prompt in a .generate-like way.
            # To avoid version mismatch errors, we'll try a safe wrapper:
            try:
                gen = llm.generate([{"role":"system","content": system_context}, {"role":"user","content": prompt}])
                assistant_response = gen.generations[0][0].text if hasattr(gen, "generations") else str(gen)
            except Exception:
                # fallback textual reply
                assistant_response = "(LLM direct call failed — ensure LangChain ChatGroq bindings compatible or create vectorstore.)"
        except Exception as e:
            assistant_response = f"(No chain available and direct LLM initialization failed: {e})"

    # save assistant response
    st.session_state.chat_history.append(("assistant", assistant_response, datetime.datetime.now()))
    # Also save to memory (if configured)
    if "memory" in st.session_state:
        try:
            st.session_state.memory.save_context({"input": user_input}, {"output": assistant_response})
        except Exception:
            # Some memory implementations differ; ignore if it fails
            pass
    # re-run to show updated history (in practice, Streamlit will show as appended)
    st.rerun()

