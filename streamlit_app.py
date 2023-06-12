import os
import openai
import pinecone
from PIL import Image
import streamlit as st
from sentence_transformers import SentenceTransformer


INDEX_NAME = "shubhams-index"
MODEL_NAME = "multi-qa-MiniLM-L6-cos-v1"

DEFAULT_PROMPT = [{"role": "user", "content": "Use the following context to answer the user query. If the user query is a question, provide an answer using the context. If the user query is a statement or a phrase, provide the best response using the context.\n\nContext:\n\n[CONTEXT]\n\nQuery: [QUERY]\n\nResponse:"}]

@st.cache_resource
def init_pinecone():
    pinecone.init(
        api_key=st.secrets["PINECONE_API_KEY"],
        environment="us-west4-gcp"
    )

@st.cache_resource
def init_openai():
    openai.api_key = st.secrets["OPENAI_API_KEY"]

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

@st.cache_data
def format_prompt(results, query):
    context = ""
    for episode_number, episode_title, text in results:
        context += f"{episode_number} ({episode_title}): {text}\n\n"

    prompt = [message.copy() for message in DEFAULT_PROMPT]
    prompt[-1]["content"] = prompt[-1]["content"].replace("[CONTEXT]", context)
    prompt[-1]["content"] = prompt[-1]["content"].replace("[QUERY]", query)

    return prompt

@st.cache_data
def get_model_response(prompt):
    init_openai()

    response = openai.ChatCompletion.create(
        model='gpt-3.5-turbo', 
        messages=prompt, 
        temperature=0.7,
        max_tokens=256,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0
    )
    
    return response['choices'][0]['message']['content'].strip()

st.title("Office Ladies Podcast Search")

image = Image.open('office-ladies-podcast-image.jpeg')
st.image(image)

st.text("By Shubham Pawar")

query = st.text_input('Enter your search query:', placeholder='Search for a phrase or an answer in the podcast transcripts', label_visibility='hidden')

if st.button('Search', type='primary'):
    results = query_index(query)

    prompt = format_prompt(results, query)
    
    response = get_model_response(prompt)
    st.markdown(f":red[{response}]")
    
    for episode_number, episode_title, text in results:
        st.subheader(f":blue[{episode_number}: {episode_title}]")
        st.write(text)