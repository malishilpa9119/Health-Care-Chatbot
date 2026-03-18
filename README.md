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

# AWS CI/CD Deployment

## Step 1: Login to AWS
Login to AWS Console.

---

## Step 2: Create IAM User
Give access:
- EC2 → used as server (virtual machine)  
- ECR → used to store Docker images  

Required Policies:
- AmazonEC2FullAccess  
- AmazonEC2ContainerRegistryFullAccess  

---

## Step 3: Create ECR Repository
Create a repository in ECR and save the URI:  
xxxxxxxx.dkr.ecr.us-east-1.amazonaws.com/medicalbot

---

## Step 4: Launch EC2 Instance
- Create EC2 instance (Ubuntu)  
- Connect to your instance  

---

## Step 5: Install Docker on EC2
sudo apt-get update -y  
sudo apt-get upgrade -y  

curl -fsSL https://get.docker.com -o get-docker.sh  
sudo sh get-docker.sh  

sudo usermod -aG docker ubuntu  
newgrp docker  

---

## Step 6: Setup Self-Hosted Runner
Go to: GitHub → Settings → Actions → Runner  
- Click "New self-hosted runner"  
- Choose OS  
- Run given commands in EC2  

---

## Step 7: Add GitHub Secrets
Go to: Repository → Settings → Secrets → Actions  

Add the following:
AWS_ACCESS_KEY_ID  
AWS_SECRET_ACCESS_KEY  
AWS_DEFAULT_REGION  
ECR_REPO  
PINECONE_API_KEY  
OPENAI_API_KEY  

---

# CI/CD Flow
- Push code to GitHub  
- GitHub Actions runs automatically  
- Docker image is created  
- Image is pushed to ECR  
- EC2 pulls image and runs it  
- Application is deployed successfully 

---

# Project Summary
This project demonstrates how to:
- Build an AI chatbot using LLM  
- Use LangChain for processing  
- Store embeddings in Pinecone  
- Build backend using Flask  
- Deploy application on AWS  
- Automate deployment using CI/CD  

---
