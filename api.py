import weaviate
import numpy as np
from numpy import dot
from numpy.linalg import norm
import pandas as pd
from dotenv import load_dotenv
import os
from flask import Flask
load_dotenv()

app = Flask(__name__)

WEAVIATE_URL = os.getenv('WEAVIATE_URL')
WEAVIATE_API_KEY = os.getenv('WEAVIATE_API_KEY')

@app.route("/")
def hello_world():
    return "<p>Hello, World!</p>"

@app.route("/user_vector/<username>")
def get_user_vector(username):

    auth_config = weaviate.AuthApiKey(api_key=WEAVIATE_API_KEY)  # Replace w/ your Weaviate instance API key

    # Instantiate the client with the auth config
    client = weaviate.Client(
        url=WEAVIATE_URL,  # Replace w/ your endpoint
        auth_client_secret=auth_config
    )

    query_chats = (
        client.query.get("Chat", ['username','role','content','timestamp','vector'])
        .with_additional(["id vector"])
    ).do()

    chat_embeddings = [x for x in query_chats['data']['Get']['Chat'] if x['username'] == username]
    chat_embeddings = [x for x in chat_embeddings if 'vector' in x]
    chat_embeddings = [x for x in chat_embeddings if x['role'] == 'user']
    chat_embeddings = [x['vector'] for x in chat_embeddings]
    chat_embeddings = np.array(chat_embeddings)

    if chat_embeddings.shape[0] > 0:
        print("Assigning product score from embeddings")
        avg_chat = np.mean(chat_embeddings, axis=0).tolist()
    else:
        avg_chat = []

    return {'vector': avg_chat}
