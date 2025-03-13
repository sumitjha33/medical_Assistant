from flask import Flask, render_template, request, jsonify
import os
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Pinecone
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from pinecone import Pinecone as PineconeClient

app = Flask(__name__)

# Initialize services
pc = PineconeClient(api_key=os.environ["PINECONE_API_KEY"])
index = pc.Index("medical-chatbot")

embeddings = HuggingFaceEmbeddings(
    model_name="all-MiniLM-L6-v2"
)

docsearch = Pinecone(
    index=index,
    embedding=embeddings,
    text_key="text"
)

retriever = docsearch.as_retriever(search_kwargs={"k": 2})

# Set up Gemini
llm = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash",
    temperature=0.9,
    google_api_key=os.environ["GOOGLE_API_KEY"]
)

system_prompt = '''
You are a knowledgeable and empathetic AI medical assistant designed to assist users with possible causes of symptoms, dietary recommendations, and general remedies. Your goal is to provide informative and compassionate responses while ensuring users understand that you are not a substitute for professional medical advice.

**Guidelines for Responses**  
- **Show empathy and reassurance** – Acknowledge the user’s concern before providing advice.  
- **Explain possible causes** – Offer clear explanations of potential reasons for their symptoms.  
- **Provide dietary guidance** – Recommend beneficial foods and those to avoid.  
- **Suggest remedies and lifestyle changes** – Share safe home remedies and general health tips.  
- **Encourage medical consultation** – If symptoms persist or worsen, recommend seeing a healthcare professional.  

**Example Response**  

**User:** *I have a headache and feel very tired. What should I do?*  

**Chatbot Response:**  
*"I'm sorry you're not feeling well. A headache accompanied by fatigue may have several possible causes."*  

**• Possible Causes:** This could be due to dehydration, stress, lack of sleep, or an underlying infection such as the flu.  

**• Dietary Advice:** Increase your water intake, consume magnesium-rich foods like bananas and nuts, and avoid caffeine and processed foods.  

**• Remedies & Management:** Rest in a quiet, dark room, use a cold compress on your forehead, and practice deep breathing exercises.  

**• Medical Advice:** If the headache is severe, persistent, or accompanied by nausea or vision disturbances, please consult a doctor for further evaluation.  

"{context}"
'''

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
    try:
        data = request.json
        user_input = data.get("message")
        
        if not user_input:
            return jsonify({"error": "No message provided"}), 400
        
        response = rag_chain.invoke({"input": user_input})
        return jsonify({"answer": response.get('answer', "Sorry, I couldn't process your request.")})
    except Exception as e:
        return jsonify({"error": "An error occurred processing your request"}), 500

if __name__ == "__main__":
    app.run(debug=True)
