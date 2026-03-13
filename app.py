import streamlit as st
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# Page config
st.set_page_config(page_title="AI FAQ Chatbot", page_icon="🤖")

st.title("🤖 AI FAQ Chatbot")
st.write("Ask me anything about accounts, billing, technical issues, and more.")

# Load data
faq_df = pd.read_csv("models/faq_knowledge_base.csv")
question_embeddings = np.load("models/question_embeddings.npy")

# Load model
@st.cache_resource
def load_model():
    return SentenceTransformer("all-MiniLM-L6-v2")

model = load_model()

# Chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Function to get best answer
def get_best_answer(query):
    query_embedding = model.encode([query])
    similarity = cosine_similarity(query_embedding, question_embeddings)
    best_match = np.argmax(similarity)
    answer = faq_df.iloc[best_match]["answer"]
    return answer

# Show previous chat
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# User input
user_input = st.chat_input("Ask your question...")

if user_input:

    # show user message
    st.session_state.messages.append({"role": "user", "content": user_input})

    with st.chat_message("user"):
        st.markdown(user_input)

    # chatbot response
    response = get_best_answer(user_input)

    st.session_state.messages.append({"role": "assistant", "content": response})

    with st.chat_message("assistant"):
        st.markdown(response)