import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Load the CSV file
faq_data = pd.read_csv('pmsss_faqs.csv')

# Preprocess the questions
faq_questions = faq_data['Question'].tolist()

# Use TF-IDF Vectorizer
vectorizer = TfidfVectorizer()
faq_vectors = vectorizer.fit_transform(faq_questions)

# Function to get the similar FAQ answer
def get_similar_faq_answer(user_query):
    user_query_vector = vectorizer.transform([user_query])
    similarity_scores = cosine_similarity(user_query_vector, faq_vectors)
    max_similarity_index = similarity_scores.argmax()
    
    if similarity_scores[0][max_similarity_index] > 0.5:
        return faq_data.iloc[max_similarity_index]['Answer']
    else:
        return "Sorry, I couldn't find an answer for your question."

# Streamlit App
st.title("PMSSS Chatbot")

user_input = st.text_input("Ask a question about PMSSS")

if user_input:
    answer = get_similar_faq_answer(user_input)
    st.write(answer)

