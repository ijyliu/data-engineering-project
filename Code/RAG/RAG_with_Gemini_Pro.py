# RAG with Gemini Pro
# Note: Place your Google API key in the Google API Key folder of the directory this code is in, as the contents of a file called 'data-engineering-project.txt'.

##################################################################################################

# Packages
from RAG_Functions import *
import time
from sentence_transformers import SentenceTransformer
from pymilvus import Collection, connections

# Google API Setup
print('\nConfiguring Google API...')
import google.generativeai as genai
import os
# Load API key from './Google API Key/data-engineering-project.txt'
with open(os.path.expanduser('./Google API Key/data-engineering-project.txt')) as f:
    GOOGLE_API_KEY = f.read().strip()
genai.configure(api_key=GOOGLE_API_KEY)

##################################################################################################

# Embedding Model
print('\nLoading Embedding Model...')
embedding_model = SentenceTransformer("mixedbread-ai/mxbai-embed-large-v1")
embedding_model

##################################################################################################

# Chat Model
print('\nSetting Up Chat Model...')
chat_model = genai.GenerativeModel('gemini-1.0-pro-latest')
chat_model

##################################################################################################

# Milvus Connection
print('\nConnecting to Milvus...')
connections.connect(host='localhost', port='19530')
collection = Collection("text_embeddings")      # Get an existing collection.
collection.load()

##################################################################################################

# Perform Chat
print('\nBeginning Chat. Type "exit" to quit.')

# Chat Loop
while True:

    # Chat with model
    input_text = input("\nUser: ")
    # Exit if user types 'exit'
    if input_text == 'exit':
        break

    # Get embedding of input
    input_embedding = get_mixedbread_of_query(embedding_model, input_text)

    # Milvus query
    # Start timing query
    start_time = time.time()
    # Top5 sentences
    top5_sentences, documents_cited, milvus_query_time = return_top_5_sentences(collection, input_embedding)
    # End timing query
    end_time = time.time()
    # query time
    query_time = end_time - start_time

    # Construct prompt
    prompt_lines = ["Context That May Be Helpful (You May Disregard if Not Helpful):"] + top5_sentences + ["User Query:\n" + input_text]
    prompt = "\n".join(prompt_lines)
    #print(prompt)

    # Get response
    # Start timer
    start_time = time.time()
    response = chat_model.generate_content(prompt)
    # End timer
    end_time = time.time()
    # Format response for user
    response_for_user = "Assistant: " + response.text + "\nDocuments Cited: " + ', '.join(documents_cited) + "\nMilvus Query Time: " + str(round(milvus_query_time, 2)) + ' seconds' + "\nChat Model Response Time: " + str(round(end_time - start_time, 2)) + ' seconds'
    print(response_for_user)
