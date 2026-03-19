# Health Care Chatbot (LLMs + LangChain + Pinecone + Flask + AWS)

This project is a Health Care Chatbot built using LLMs.  
It uses LangChain, Pinecone, and OpenAI to provide intelligent medical-related responses.  
This project also includes AWS CI/CD deployment using Docker and GitHub Actions.

---

# How to Run the Project

## Step 1: Clone the Repository
git clone https://github.com/malishilpa9119/Health-Care-Chatbot.git  
cd Health-Care-Chatbot

---

## Step 2: Create Conda Environment
conda create -n healthbot python=3.10 -y  
conda activate healthbot

---

## Step 3: Install Requirements
pip install -r requirements.txt

---

## Step 4: Create .env File
Create a `.env` file in the root folder and add your API keys:

PINECONE_API_KEY="your_pinecone_key"  
OPENAI_API_KEY="your_openai_key"

---

## Step 5: Store Embeddings in Pinecone
python store_index.py

---

## Step 6: Run the Application
python app.py

Now open your browser and go to:  
http://localhost:5000

---

# Tech Stack Used
- Python  
- LangChain  
- Flask  
- OpenAI (GPT)  
- Pinecone  

---
