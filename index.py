import os
from flask import Flask, request, jsonify
from flask_cors import CORS
from langchain_groq import ChatGroq
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.retrieval import create_retrieval_chain
from langchain.vectorstores import FAISS
from langchain.document_loaders import PyPDFDirectoryLoader
from langchain.embeddings import HuggingFaceEmbeddings
import pickle



from dotenv import load_dotenv

load_dotenv()


from langsmith import utils
utils.tracing_is_enabled()
app = Flask(__name__)


CORS(app)

# Set API keys
os.environ['GROQ_API_KEY'] = os.getenv('GROQ_API_KEY')
os.environ['HUGGINGFACE_API_KEY'] = os.getenv('HUGGINGFACE_API_KEY')

# Initialize the language model
llm = ChatGroq(model="mixtral-8x7b-32768")

# Define the prompt template
prompt = ChatPromptTemplate.from_template(
    """
    Please provide the answer based on the given prompt.
    Please provide the accurate answer based on the given prompt.
    <context>
    {context}
    </context>

    Question: {input}
    """
)

# Define the vector embedding and storage function
def vector_embeddings(data_dir="data", max_documents=50, persist_path="faiss_index.pkl"):
    # Initialize embeddings
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    
    # Load PDF documents
    loader = PyPDFDirectoryLoader(data_dir)
    documents = loader.load()[:max_documents]
    
    # Split documents into chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    texts = text_splitter.split_documents(documents)
    
    # Create FAISS vector store
    faiss_db = FAISS.from_documents(texts, embeddings)
    
    # Save FAISS index to file
    with open(persist_path, "wb") as f:
        pickle.dump(faiss_db, f)
    print(f"FAISS index stored successfully at {persist_path}")
    return faiss_db

# Create or load the FAISS index
persist_path = "faiss_index.pkl"
if not os.path.exists(persist_path):
    vector = vector_embeddings()
else:
    with open(persist_path, "rb") as f:
        vector = pickle.load(f)

# User input and retrieval
# user_input = input("Enter your query: ")



@app.route("/query-rag",methods=['POST'])

def handle_input():
  
    data=request.json
    user_query = data.get("query","")

    if not user_query:
        return jsonify({"error":"Query is required"}),400
    
      # Create document chain
    document_chain = create_stuff_documents_chain(llm, prompt)
    
    # Set up retriever
    retriever = vector.as_retriever()

    # Create retrieval chain
    retrieval_chain = create_retrieval_chain(retriever, document_chain)
    
    # Get response
    response = retrieval_chain.invoke({"input": user_query})
    
    # return response
    return jsonify({"response":response['answer']})

if __name__ == "__main__":
    app.run(debug=False)