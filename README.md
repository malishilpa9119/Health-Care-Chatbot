# Health Care Chatbot — RAG Clinic Assistant (LangGraph + LangChain + Pinecone + Groq)

A Retrieval-Augmented Generation (RAG) based **Healthcare Clinic Assistant** that answers
patient queries — appointments, doctor availability, timings, fees, lab tests, and refill
policies — **only from the provided clinic documents**, keeping responses reliable and
hallucination-free.

The full workflow is orchestrated with **LangGraph**: every query passes through a safety
guardrail, then retrieval, then generation, with a dedicated fallback path when the answer
isn't in the documents.

---

## Features

- **RAG pipeline** — answers grounded strictly in the clinic documents
- **Hybrid Retrieval** — semantic vector search + BM25 keyword search
- **Cross-Encoder Re-ranking** — reorders results for higher accuracy
- **Pinecone** vector database with **HuggingFace** embeddings (`all-MiniLM-L6-v2`)
- **Custom retriever tool** (`clinic_document_search`) exposed to an agent
- **LangGraph workflow** with defined nodes → `guardrail → retrieve → generate → fallback`
- **Multi-turn memory** for context-aware follow-up questions
- **Safety guardrails** — emergency and medication/dosage queries handled safely
- **Fallback responses** when information is unavailable
- **Image analysis** (bonus) for basic visible-condition queries
- **Flask** web interface

---

## Tech Stack

- Python
- LangChain
- LangGraph
- Groq (Llama-3.3-70B)
- Pinecone
- HuggingFace Embeddings
- BM25 + Cross-Encoder Re-ranking
- Flask
- OpenAI (used only for image analysis)

---

## Project Structure

```
Health-Care-Chatbot/
├── data/                 # Clinic / medical PDF documents
├── src/
│   ├── helper.py         # PDF loading, splitting, embeddings
│   ├── retriever.py      # Hybrid retriever + re-ranking
│   ├── agent.py          # LangGraph agent (guardrail, retrieve, generate, fallback)
│   └── prompt.py         # System prompt
├── static/style.css      # UI styles
├── templates/chat.html   # Chat UI
├── store_index.py        # Builds the Pinecone index from documents
├── app.py                # Flask application
└── requirements.txt
```

---

## How to Run the Project

### Step 1: Clone the Repository
```bash
git clone https://github.com/malishilpa9119/Health-Care-Chatbot.git
cd Health-Care-Chatbot
```

### Step 2: Create a Virtual Environment
```bash
# Windows (PowerShell)
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# macOS / Linux
python3 -m venv .venv
source .venv/bin/activate
```

### Step 3: Install Requirements
```bash
pip install -r requirements.txt
```

### Step 4: Create a `.env` File
Create a `.env` file in the root folder and add your API keys:
```
PINECONE_API_KEY="your_pinecone_key"
GROQ_API_KEY="your_groq_key"
OPENAI_API_KEY="your_openai_key"   # optional, only for image analysis
```

### Step 5: Add Documents and Build the Index
Place your clinic/medical PDF files inside the `data/` folder, then run:
```bash
python store_index.py
```

### Step 6: Run the Application
```bash
python app.py
```

Now open your browser and go to:
```
http://localhost:8080
```

---

## Example Questions

- What are the clinic timings?
- Is Dr. Meera available on Sunday?
- How long should I fast before a blood sugar test?
- Can I drink water while fasting for the test?
- What is the consultation fee?
- I have severe chest pain *(triggers the emergency guardrail)*

---

## Notes

- The assistant answers only from the documents in `data/`. If the information isn't there,
  it returns a safe fallback instead of guessing.
- Re-run `store_index.py` whenever you add or change documents in `data/`.