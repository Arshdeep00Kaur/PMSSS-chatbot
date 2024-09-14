import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Load the CSV file
@st.cache_data
def load_faq_data():
    return pd.read_csv('pmsss_faqs.csv')

faq_data = load_faq_data()

# Preprocess the questions
faq_questions = faq_data['Question'].tolist()

# Cache the vectorization process
@st.cache_resource
def get_vectorizer_and_vectors(faq_questions):
    vectorizer = TfidfVectorizer()
    faq_vectors = vectorizer.fit_transform(faq_questions)
    return vectorizer, faq_vectors

vectorizer, faq_vectors = get_vectorizer_and_vectors(faq_questions)

# Function to get the similar FAQ answer
def get_similar_faq_answer(user_query):
    user_query_vector = vectorizer.transform([user_query])
    similarity_scores = cosine_similarity(user_query_vector, faq_vectors)
    max_similarity_index = similarity_scores.argmax()

    if similarity_scores[0][max_similarity_index] > 0.5:
        return faq_data.iloc[max_similarity_index]['Answer']
    else:
        return "Sorry, I couldn't find an answer for your question."

# Initialize chat session history if not already present
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []  # Initialize chat history

# Streamlit App
st.title("PMSSS Chatbot")

# Injecting custom CSS to change the background color
st.markdown(
    """
    <style>
    .stApp {
        background-color: #D3D3D3;
    }
    </style>
    """,
    unsafe_allow_html=True
)
# configure streamlit page settings
st.set_page_config(
   
    page_icon=":brain",   # favicon emoji
    layout="centered"     # layout option
)

# Input for user question
user_input = st.text_input("Ask your PMSSS queries, and the bot will assist you!")

if user_input:
    # Get answer based on user input
    answer = get_similar_faq_answer(user_input)
    
    # Store the question and answer in the session state chat history
    st.session_state.chat_history.append({"user": user_input, "bot": answer})

# Display chat history
if st.session_state.chat_history:
    for chat in st.session_state.chat_history:
        st.write(f"**You:** {chat['user']}")
        st.write(f"**Bot:** {chat['bot']}")
        st.write("---")  # Divider between each Q&A
