# Health Care Chatbot (LLMs + LangChain + Pinecone + Flask + AWS)

An AI-powered medical chatbot that answers health-related queries using Retrieval-Augmented Generation (RAG). Instead of relying solely on the LLM's training data, it retrieves relevant medical documents first and generates accurate, context-grounded responses.

## Architecture & How It Works

```
User Query --> Flask API --> LangChain Orchestrator --> Pinecone (Semantic Search)
                                    |                         |
                                    |                   Top-K Relevant Chunks
                                    |                         |
                                    v                         v
                              Prompt Construction <-- Retrieved Context
                                    |
                                    v
                              OpenAI GPT (Response Generation)
                                    |
                                    v
                              User Gets Answer
```

### The RAG Pipeline (Step by Step)

**1. Document Ingestion & Embedding**
- Medical documents are loaded and split into smaller chunks (500 tokens with 100-token overlap) for better retrieval accuracy.
- Each chunk is converted into a vector embedding using OpenAI's embedding model.
- Embeddings are stored in Pinecone vector database for fast semantic search.
- Script: `store_index.py`

**2. Query Processing & Retrieval**
- When a user asks a question, the query is converted into an embedding.
- Pinecone performs a similarity search and returns the top-K most relevant document chunks.
- Low-relevance results are filtered out using a score threshold.

**3. Response Generation**
- LangChain constructs a prompt combining the user query + retrieved context.
- The prompt instructs the LLM to answer only from the provided context.
- OpenAI GPT generates a response grounded in the retrieved medical documents.
- If no relevant context is found, the model responds with "I don't have enough information" instead of hallucinating.

## Key Technical Decisions

| Decision | Why |
|----------|-----|
| RAG over fine-tuning | Medical info changes frequently; RAG lets us update documents without retraining |
| Pinecone over FAISS | Managed cloud service, scales without infra overhead, persistent storage |
| Chunk size: 500 tokens | Tested 300, 500, 1000 — 500 gave best balance of context relevance and precision |
| Score threshold filtering | Prevents the LLM from generating answers when retrieved context is not relevant enough |

## Challenges & How I Solved Them

**Problem: Hallucinated medical answers**
The LLM was generating confident but incorrect answers when relevant context was missing from the vector database. Solved by adding strict prompt instructions + relevance score filtering to reject low-confidence retrievals.

**Problem: Noisy retrieval results**
Large chunks were pulling irrelevant surrounding text. Fixed by reducing chunk size from 1000 to 500 tokens with overlap, which improved retrieval precision significantly.

## Tech Stack

| Component | Technology |
|-----------|------------|
| Backend | Python, Flask |
| LLM Orchestration | LangChain |
| Language Model | OpenAI GPT |
| Vector Database | Pinecone |
| Embeddings | OpenAI Embedding Model |
| Deployment | Docker, AWS, GitHub Actions CI/CD |

## Project Structure

```
Health-Care-Chatbot/
├── app.py                 # Flask application server
├── store_index.py         # Document embedding & Pinecone indexing
├── src/                   # LangChain integration modules
├── data/                  # Medical knowledge base documents
├── templates/             # HTML frontend
├── static/                # CSS & JS assets
├── research/              # Experiment notebooks
├── .github/workflows/     # CI/CD pipeline
├── Dockerfile             # Container configuration
├── requirements.txt       # Python dependencies
└── .env                   # API keys (not committed)
```

## How to Run Locally

```bash
# Clone the repo
git clone https://github.com/malishilpa9119/Health-Care-Chatbot.git
cd Health-Care-Chatbot

# Create conda environment
conda create -n healthbot python=3.10 -y
conda activate healthbot

# Install dependencies
pip install -r requirements.txt

# Add API keys in .env file
PINECONE_API_KEY="your_pinecone_key"
OPENAI_API_KEY="your_openai_key"

# Store embeddings in Pinecone
python store_index.py

# Run the application
python app.py
# Open http://localhost:5000
```

## Results

- Accurate, context-grounded medical responses with minimal hallucination
- Fast retrieval from Pinecone with sub-second query response time
- Deployed and running on AWS with automated CI/CD pipeline

---

Built by [Shilpa Mali](https://github.com/malishilpa9119)
