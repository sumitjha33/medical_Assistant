from flask import Flask, render_template, request, jsonify
import os
import gc
import logging
import tempfile
from pathlib import Path
from pinecone import Pinecone
from langchain_community.vectorstores import Pinecone as PineconeVectorStore
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from huggingface_hub import login
from transformers import AutoTokenizer, AutoModel

# Configure logging and environment
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Set up model caching directory
CACHE_DIR = os.getenv('TRANSFORMER_CACHE', os.path.join(tempfile.gettempdir(), 'huggingface'))
os.makedirs(CACHE_DIR, exist_ok=True)

# Set up HuggingFace token
HUGGINGFACE_TOKEN = os.environ.get("HUGGINGFACE_TOKEN")
if HUGGINGFACE_TOKEN:
    login(token=HUGGINGFACE_TOKEN)
else:
    logger.warning("HUGGINGFACE_TOKEN not found in environment variables")

app = Flask(__name__)

def initialize_embeddings():
    try:
        # Use a smaller, more memory-efficient model
        return HuggingFaceBgeEmbeddings(
            model_name="sentence-transformers/paraphrase-MiniLM-L3-v2",  # Smaller model
            model_kwargs={
                'device': 'cpu',
                'torch_dtype': 'float32',  # Use float32 instead of float64
                'max_length': 128  # Limit sequence length
            },
            encode_kwargs={
                'normalize_embeddings': True,
                'batch_size': 8  # Smaller batch size
            }
        )
    except Exception as e:
        logger.error(f"Embedding initialization failed: {e}")
        raise

def clean_memory():
    gc.collect()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass

# Update initialization section
try:
    clean_memory()
    # Initialize Pinecone
    pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])
    index = pc.Index("medical-chatbot")
    
    # Initialize embeddings
    embeddings = initialize_embeddings()
    clean_memory()
    
    # Initialize vector store with smaller batch size
    docsearch = PineconeVectorStore(
        index=index,
        embedding=embeddings,
        text_key="text",
        batch_size=8  # Smaller batch size for memory efficiency
    )
    
    retriever = docsearch.as_retriever(
        search_type='similarity',
        search_kwargs={"k": 2}  # Reduced from 3 to 2 for memory efficiency
    )
    clean_memory()

except Exception as e:
    logger.error(f"Initialization error: {str(e)}")
    raise

# Set up Google Gemini AI API key
GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    raise ValueError("GOOGLE_API_KEY is missing in environment variables.")

llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0.9)

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
        clean_memory()
        return jsonify({"answer": response.get('answer', "Sorry, I couldn't process your request.")})
    except Exception as e:
        logger.error(f"Chat error: {e}")
        return jsonify({"error": "An error occurred processing your request"}), 500

if __name__ == "__main__":
    app.run(debug=True)
