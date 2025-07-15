from dotenv import load_dotenv
import os

# Load environment variables from .env file
load_dotenv()

# Fetch API keys from environment variables
openai_api_key = os.getenv("OPENAI_API_KEY")
pinecone_api_key = os.getenv("PINECONE_API_KEY")
pinecone_environment = os.getenv("PINECONE_ENVIRONMENT")

# Debug: Print the values to verify they are being fetched correctly
print("OpenAI API Key:", openai_api_key)
print("Pinecone API Key:", pinecone_api_key)
print("Pinecone Environment:", pinecone_environment)

# Check if the keys are loaded
if not openai_api_key or not pinecone_api_key or not pinecone_environment:
    print("Error: One or more API keys are missing. Please check your .env file.")
else:
    print("All API keys loaded successfully!")