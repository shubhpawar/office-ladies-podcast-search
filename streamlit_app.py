import os
import pinecone
from PIL import Image
import streamlit as st
from sentence_transformers import SentenceTransformer


PINECONE_API_KEY = os.environ.get('PINECONE_API_KEY')
INDEX_NAME = "shubhams-index"
MODEL_NAME = "multi-qa-MiniLM-L6-cos-v1"

@st.cache_resource
def init_pinecone():
    pinecone.init(
        api_key=st.secrets["PINECONE_API_KEY"],
        environment="us-west4-gcp"
    )

@st.cache_resource
def load_model(model_name):
    model = SentenceTransformer(model_name)
    return model

model = load_model(MODEL_NAME)

@st.cache_data
def query_index(query, episode_title=None, num_results=5):
    init_pinecone()
    index = pinecone.Index(INDEX_NAME)

    query_embedding = model.encode(query, show_progress_bar=False).tolist()

    metadata_filter = {"episode_title": {"$eq": episode_title}} if episode_title else None

    results = index.query(query_embedding, top_k=num_results, include_metadata=True, filter=metadata_filter)

    return [(
        match['metadata']['episode_number'], 
        match['metadata']['episode_title'], 
        match['metadata']['text']
    ) for match in results['matches']]

st.title("Office Ladies Podcast Search")

image = Image.open('office-ladies-podcast-image.jpeg')
st.image(image)

st.text("By Shubham Pawar")

query = st.text_input('Enter your search query:', placeholder='Search for a phrase or an answer in the podcast transcripts', label_visibility='hidden')

if st.button('Search', type='primary'):
    results = query_index(query)
    
    for episode_number, episode_title, text in results:
        st.subheader(f":blue[{episode_number}: {episode_title}]")
        st.write(text)