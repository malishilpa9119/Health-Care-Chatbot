from flask import Flask, render_template, jsonify, request, session
from src.helper import download_hugging_face_embeddings
from langchain_pinecone import PineconeVectorStore
from langchain_openai import ChatOpenAI
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage
from langchain.chains.history_aware_retriever import create_history_aware_retriever
from dotenv import load_dotenv
from openai import OpenAI
from src.prompt import *
import os
import base64


app = Flask(__name__)
app.secret_key = os.urandom(24)


load_dotenv()

PINECONE_API_KEY=os.environ.get('PINECONE_API_KEY')
OPENAI_API_KEY=os.environ.get('OPENAI_API_KEY')

os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY


embeddings = download_hugging_face_embeddings()

index_name = "health-care-chatbot"
docsearch = PineconeVectorStore.from_existing_index(
    index_name=index_name,
    embedding=embeddings
)


retriever = docsearch.as_retriever(search_type="similarity", search_kwargs={"k":3})

chatModel = ChatOpenAI(model="gpt-4o-mini")

contextualize_q_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "Given a chat history and the latest user question "
         "which might reference context in the chat history, "
         "formulate a standalone question which can be understood "
         "without the chat history. Do NOT answer the question, "
         "just reformulate it if needed and otherwise return it as is."),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]
)

history_aware_retriever = create_history_aware_retriever(
    chatModel, retriever, contextualize_q_prompt
)

qa_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]
)

question_answer_chain = create_stuff_documents_chain(chatModel, qa_prompt)
rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)



@app.route("/")
def index():
    session.clear()
    return render_template('chat.html')



@app.route("/get", methods=["GET", "POST"])
def chat():
    msg = request.form["msg"]
    print(msg)

    chat_history = []
    for entry in session.get("chat_history", []):
        chat_history.append(HumanMessage(content=entry["human"]))
        chat_history.append(AIMessage(content=entry["ai"]))

    response = rag_chain.invoke({"input": msg, "chat_history": chat_history})
    answer = response["answer"]
    print("Response : ", answer)

    if "chat_history" not in session:
        session["chat_history"] = []
    session["chat_history"].append({"human": msg, "ai": answer})
    session.modified = True

    return str(answer)


@app.route("/analyze-image", methods=["POST"])
def analyze_image():
    msg = request.form.get("msg", "").strip()
    image_file = request.files.get("image")

    if not image_file:
        return "No image uploaded.", 400

    image_data = base64.b64encode(image_file.read()).decode("utf-8")
    mime_type = image_file.content_type or "image/jpeg"

    user_text = msg if msg else "Please analyze this image for any visible health conditions and suggest possible solutions."

    try:
        client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "system",
                    "content": "You are a medical health assistant. Analyze the uploaded image for any visible health conditions "
                               "(such as acne, rashes, skin conditions, injuries, etc.). Describe what you observe, suggest possible "
                               "conditions, and recommend treatments or next steps. Always remind the user to consult a healthcare "
                               "professional for proper diagnosis."
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": user_text},
                        {"type": "image_url", "image_url": {"url": f"data:{mime_type};base64,{image_data}"}}
                    ]
                }
            ],
            max_tokens=1000
        )

        answer = response.choices[0].message.content
        print("Image Analysis Response:", answer)
    except Exception as e:
        print("Image analysis error:", str(e))
        return f"Error analyzing image: {str(e)}", 500

    if "chat_history" not in session:
        session["chat_history"] = []
    session["chat_history"].append({"human": f"[Image uploaded] {user_text}", "ai": answer})
    session.modified = True

    return str(answer)


if __name__ == '__main__':
    app.run(host="0.0.0.0", port= 8080, debug= True)
