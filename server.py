from flask import Flask, json, request, jsonify
from flask_cors import CORS
import pandas as pd
import numpy as np
from ast import literal_eval
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import torch
import joblib
from google.cloud import storage
from google.oauth2 import service_account
import base64

import os

# # Initialize a Google Cloud Storage client using the service account key
credentials_base64 = os.environ.get('GOOGLE_CREDENTIALS_JSON_BASE64')
credentials_json = base64.b64decode(credentials_base64).decode('utf-8')
credentials_info = json.loads(credentials_json)
credentials = service_account.Credentials.from_service_account_info(credentials_info)

client = storage.Client(credentials=credentials)

# load model from GCS bucket titled 'recipe-recommender-model'
# client = storage.Client()
bucket = client.get_bucket('recipe-recommender-model')
blobs_all = list(bucket.list_blobs())

# Download sentence_transformer_model directory from bucket
blobs = [b for b in blobs_all if b.id.__contains__('sentence_transformer_model/')]
for blob in blobs:
   if not blob.name.endswith('/'):
        file_path = f'./{blob.name}'
        os.makedirs(os.path.dirname(file_path), exist_ok=True)  # Ensure directories exist
        print(f'Downloading file [{blob.name}] to [{file_path}]')
        blob.download_to_filename(file_path)

# Download other files from bucket
blob = bucket.blob('tfidf_vectorizer.pkl')
blob.download_to_filename('tfidf_vectorizer.pkl')
blob = bucket.blob('description_embeddings.pt')
blob.download_to_filename('description_embeddings.pt')
blob = bucket.blob('genre_matrix.npy')
blob.download_to_filename('genre_matrix.npy')
blob = bucket.blob('description_similarities.npy')
blob.download_to_filename('description_similarities.npy')
blob = bucket.blob('genre_similarities.npy')
blob.download_to_filename('genre_similarities.npy')
blob = bucket.blob('combined_similarities.npy')
blob.download_to_filename('combined_similarities.npy')
blob = bucket.blob('recipes_data_10.csv')
blob.download_to_filename('recipes_data_10.csv')



app = Flask(__name__)

CORS(app)

# Load data
df = pd.read_csv('recipes_data_10.csv')

# Load SentenceTransformer model
model = SentenceTransformer('sentence_transformer_model')

# Load TF-IDF vectorizer
tfidf = joblib.load('tfidf_vectorizer.pkl')

# Load embeddings and similarity matrices
description_embeddings = torch.load('description_embeddings.pt', map_location=torch.device('cpu'), weights_only=True)
genre_matrix = np.load('genre_matrix.npy')
description_similarities = np.load('description_similarities.npy')
genre_similarities = np.load('genre_similarities.npy')
combined_similarities = np.load('combined_similarities.npy')

@app.route('/recommend', methods=['POST'])
def get_recommendations():
    data = request.json
    input_movie = data.get('input_movie')
    custom_description = data.get('custom_description')
    custom_genres = data.get('custom_genres')
    n_recommendations = data.get('n_recommendations', 10)

    if input_movie:
        idx = df[df['title'] == input_movie].index[0]
        similarities = combined_similarities[idx]
    elif custom_description and custom_genres:
        custom_description_embedding = model.encode([custom_description], convert_to_tensor=True, device='cpu')
        custom_genre_vector = tfidf.transform([' '.join(custom_genres)])

        description_sim = torch.nn.functional.cosine_similarity(custom_description_embedding, description_embeddings).cpu().numpy()
        genre_sim = cosine_similarity(custom_genre_vector, genre_matrix).flatten()

        similarities = 0.7 * description_sim + 0.3 * genre_sim
    else:
        return jsonify({"error": "Please provide either an input movie or a custom description and genres."}), 400

    top_indices = similarities.argsort()[-n_recommendations-1:-1][::-1]
    recommendations = df.iloc[top_indices][['title', 'directions', 'ingredients', 'link']]

    # Ensure directions are in array format
    recommendations['directions'] = recommendations['directions'].apply(eval)
    recommendations['ingredients'] = recommendations['ingredients'].apply(eval)

    return jsonify(recommendations.to_dict(orient='records'))

@app.route('/recipe_titles', methods=['GET'])
def get_recipe_titles():
    titles = df['title'].tolist()
    return jsonify(titles)

@app.route('/random_recipe', methods=['GET'])
def get_random_recipe():
    random_recipe = df.sample(1).to_dict(orient='records')[0]
    
    # Ensure directions are in array format
    if 'directions' in random_recipe:
        random_recipe['directions'] = eval(random_recipe['directions'])
    if 'ingredients' in random_recipe:
        random_recipe['ingredients'] = eval(random_recipe['ingredients'])

    print(random_recipe)

    return jsonify(random_recipe)

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 8080))  # Default to 8080 if PORT is not set
    app.run(host='0.0.0.0', port=port)
