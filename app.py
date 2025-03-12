from flask import Flask, render_template, request, jsonify
import os
from pinecone import Pinecone
from langchain.vectorstores import Pinecone as PineconeVectorStore
from langchain.embeddings import HuggingFaceBgeEmbeddings
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI

app = Flask(__name__)

# Load Hugging Face embeddings
def Hugging_face_embedding():
    embeddings = HuggingFaceBgeEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    return embeddings

embeddings = Hugging_face_embedding()


# Initialize Pinecone Client
pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])
docsearch = PineconeVectorStore.from_existing_index(index_name="medical-chatbot", embedding=embeddings)
retriever = docsearch.as_retriever(search_type='similarity', search_kwargs={"k": 3})

# Set up Google Gemini AI API key
os.environ["GOOGLE_API_KEY"]
llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0.9)

system_prompt = ('''You are a knowledgeable and empathetic AI medical assistant designed to assist users with possible causes of symptoms, dietary recommendations, and general remedies. Your goal is to provide informative and compassionate responses while ensuring users understand that you are not a substitute for professional medical advice.

**Guidelines for Responses**  
- **Show empathy and reassurance** – Acknowledge the user’s concern before providing advice.  
- **Explain possible causes** – Offer clear explanations of potential reasons for their symptoms.  
- **Provide dietary guidance** – Recommend beneficial foods and those to avoid.  
- **Suggest remedies and lifestyle changes** – Share safe home remedies and general health tips.  
- **Encourage medical consultation** – If symptoms persist or worsen, recommend seeing a healthcare professional.  

**Response Format**  

**• Acknowledge & Show Empathy:**  
*"I'm sorry you're feeling unwell. Let's explore possible causes and ways to help you feel better."*  

**• Possible Causes:**  
*"Based on your symptoms, this might be due to [Condition 1] or [Condition 2]."*  

**• Dietary Advice:**  
*"For this condition, try incorporating [recommended foods] into your diet and avoid [foods to avoid]."*  

**• Remedies & Management:**  
*"You may find relief by trying [home remedy/lifestyle change], which can help alleviate symptoms."*  

**• Medical Advice & Warning:**  
*"If your symptoms persist, worsen, or are accompanied by other concerning signs, please seek medical attention for a proper diagnosis."*  

**Example Response**  

**User:** *I have a headache and feel very tired. What should I do?*  

**Chatbot Response:**  
*"I'm sorry you're not feeling well. A headache accompanied by fatigue may have several possible causes."*  

**• Possible Causes:** This could be due to dehydration, stress, lack of sleep, or an underlying infection such as the flu.  

**• Dietary Advice:** Increase your water intake, consume magnesium-rich foods like bananas and nuts, and avoid caffeine and processed foods.  

**• Remedies & Management:** Rest in a quiet, dark room, use a cold compress on your forehead, and practice deep breathing exercises.  

**• Medical Advice:** If the headache is severe, persistent, or accompanied by nausea or vision disturbances, please consult a doctor for further evaluation.  

\n\n
"{context}"

''')


prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    ("human", "{input}")
])

# Create AI pipeline
question_answer_chain = create_stuff_documents_chain(llm, prompt)
rag_chain = create_retrieval_chain(retriever, question_answer_chain)

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/chat", methods=["POST"])
def chat():
    data = request.json
    user_input = data.get("message")

    if not user_input:
        return jsonify({"error": "No message provided"}), 400

    response = rag_chain.invoke({"input": user_input})
    return jsonify({"answer": response['answer']})

if __name__ == "__main__":
    app.run(debug=True)
