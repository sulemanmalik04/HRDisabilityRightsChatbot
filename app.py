from flask import Flask, request, jsonify, render_template
from flask_cors import CORS  # Enable CORS if needed
import os
import time
import logging
import threading
from openai import OpenAI
from pinecone import Pinecone, ServerlessSpec
from PyPDF2 import PdfReader
from PyPDF2.errors import PdfReadError
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Check for required environment variables
required_env_vars = ["OPENAI_API_KEY", "PINECONE_API_KEY"]
for var in required_env_vars:
    if not os.getenv(var):
        raise ValueError(f"Missing required environment variable: {var}")

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Initialize OpenAI and Pinecone
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))

# Constants
INDEX_NAME = "hrdisability-rights"
DATA_FOLDER = "data"
DEFAULT_RESPONSE = "Hello! I am your HR Disability Rights chatbot. How can I help you today regarding disability inclusion, policies, or accommodations?"

# Create or connect to Pinecone index
if INDEX_NAME not in pc.list_indexes().names():
    pc.create_index(
        name=INDEX_NAME,
        dimension=1536,  # 1536 is the dimension of OpenAI's embeddings
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1")
    )
index = pc.Index(INDEX_NAME)
logging.info(f"Using Pinecone index: {INDEX_NAME}")

# Track indexed files and their last modified time
indexed_files = {}

# Function to extract text from PDFs
def extract_text_from_pdf(pdf_path):
    try:
        reader = PdfReader(pdf_path)
        text = ""
        for page in reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text
        return text
    except PdfReadError as e:
        logging.error(f"Error reading PDF {pdf_path}: {e}")
        return None

# Function to split text into chunks
def split_text_into_chunks(text, max_length=500):
    words = text.split()
    chunks = []
    current_chunk = []
    current_length = 0

    for word in words:
        if current_length + len(word) + 1 <= max_length:
            current_chunk.append(word)
            current_length += len(word) + 1
        else:
            chunks.append(" ".join(current_chunk))
            current_chunk = [word]
            current_length = len(word)

    if current_chunk:
        chunks.append(" ".join(current_chunk))

    return chunks

# Function to generate embeddings using OpenAI
def generate_embedding(text):
    time.sleep(1)  # Add a delay to avoid rate limits
    response = client.embeddings.create(
        input=text,
        model="text-embedding-ada-002"
    )
    return response.data[0].embedding

# Improved function to index PDFs with detailed logging
def index_pdfs():
    logging.info(f"Checking files in {DATA_FOLDER}...")
    if not os.path.exists(DATA_FOLDER):
        logging.error(f"Data folder {DATA_FOLDER} does not exist!")
        return

    files = os.listdir(DATA_FOLDER)
    logging.info(f"Files found: {files}")

    for filename in files:
        if filename.endswith(".pdf"):
            pdf_path = os.path.join(DATA_FOLDER, filename)
            last_modified = os.path.getmtime(pdf_path)

            if filename not in indexed_files or indexed_files[filename] < last_modified:
                try:
                    text = extract_text_from_pdf(pdf_path)
                    if text:
                        logging.info(f"Extracted text from {filename}: {text[:300]}...")
                        chunks = split_text_into_chunks(text)
                        for i, chunk in enumerate(chunks):
                            logging.info(f"Generating embedding for chunk {i} of {filename}")
                            embedding = generate_embedding(chunk)
                            index.upsert([(f"{filename}_chunk{i}", embedding, {"text": chunk})])
                            logging.info(f"Upserted chunk {i} of {filename}")
                        indexed_files[filename] = last_modified
                        logging.info(f"Indexed {filename} successfully!")
                    else:
                        logging.warning(f"No text extracted from {filename}, skipping indexing.")
                except Exception as e:
                    logging.error(f"Error indexing {filename}: {e}")
            else:
                logging.info(f"{filename} has not changed, skipping.")
        else:
            logging.info(f"Skipping non-PDF file: {filename}")

def query_chatbot(user_query):
    query_embedding = generate_embedding(user_query)
    results = index.query(
        vector=query_embedding,
        top_k=3,
        include_metadata=True
    )
    
    if not results['matches']:
        return "I'm sorry, but I can only answer questions related to HR policies and disability rights."

    context = ""
    for match in results['matches']:
        context += match['metadata']['text'] + "\n"
    
    if not context.strip():
        return "I'm sorry, but I can only answer questions related to HR policies and disability rights."
    
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a helpful assistant specializing in HR policies and disability rights. "
                    "Provide clear, accurate, and supportive answers about disability accommodations, inclusion, and related HR topics only. "
                    "If a question is unrelated, politely decline."
                )
            },
            {"role": "user", "content": f"Context: {context}\n\nQuestion: {user_query}"}
        ]
    )
    
    return response.choices[0].message.content

# Auto-index PDFs on startup and periodically check for updates
def auto_index_pdfs():
    logging.info("Started auto-indexing thread.")
    while True:
        index_pdfs()
        time.sleep(60)  # Check every 60 seconds

# Flask app for handling frontend requests
app = Flask(__name__)
CORS(app)

@app.route('/')
def home():
    return render_template('index.html')  # Serve the HTML file

@app.route('/query', methods=['POST'])
def handle_query():
    data = request.json
    if not data or 'query' not in data:
        return jsonify({"error": "Missing 'query' field in request"}), 400

    user_query = data.get('query')
    response = query_chatbot(user_query)
    return jsonify({"response": response})

@app.route('/health')
def health_check():
    return jsonify({"status": "healthy"}), 200

# Graceful shutdown for the background thread
def stop_indexing_thread():
    indexing_thread.join(timeout=1)

import atexit
atexit.register(stop_indexing_thread)

# Run the Flask app and auto-indexing in parallel
if __name__ == "__main__":
    indexing_thread = threading.Thread(target=auto_index_pdfs)
    indexing_thread.daemon = True
    indexing_thread.start()

    app.run(debug=True)
