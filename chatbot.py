import streamlit as st
import nltk
import string
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize


nltk.download('punkt_tab')
nltk.download('stopwords')

# ----------- Prétraitement ----------
def preprocess(text):
    # Découper en phrases
    sentences = sent_tokenize(text.lower())
    
        # Nettoyage de ponctuation et stopwords
    stop_words = set(stopwords.words("french"))
    clean_sentences = []

    for sentence in sentences:
        words = sentence.translate(str.maketrans('', '', string.punctuation)).split()
        words = [w for w in words if w not in stop_words]
        clean_sentences.append(' '.join(words))
    
    return sentences, clean_sentences

# ----------- Similarité ----------
def get_most_relevant_sentence(user_input, original_sentences, clean_sentences):
    all_sentences = clean_sentences + [user_input]
    vectorizer = TfidfVectorizer()
    tfidf = vectorizer.fit_transform(all_sentences)
    
    # Similarité entre la question (dernier élément) et toutes les autres
    similarity = cosine_similarity(tfidf[-1], tfidf[:-1])
    index = similarity.argmax()
    return original_sentences[index]

# ----------- Fonction chatbot ----------
def chatbot(user_input, text):
    original_sentences, clean_sentences = preprocess(text)
    user_input_clean = user_input.lower().translate(str.maketrans('', '', string.punctuation))
    return get_most_relevant_sentence(user_input_clean, original_sentences, clean_sentences)

# ----------- Interface Streamlit ----------
def main():
    st.title("🤖 Chatbot sur la Data Science")

    # Charger le fichier texte
    with open("My_field.txt", "r", encoding="utf-8") as file:
        text = file.read()
        
            # Zone de saisie utilisateur
    user_input = st.text_input("Posez une question sur la science des données :")

    if user_input:
        response = chatbot(user_input, text)
        st.markdown(f"**Réponse :** {response}")

if __name__ == "__main__":
    main()
    
    