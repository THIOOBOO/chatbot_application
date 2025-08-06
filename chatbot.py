# Voici le contenu de mon fichier texte
# ABOUT DATA SCIENCE

# La data science, ou science des données, est un domaine interdisciplinaire qui utilise des méthodes scientifiques, des algorithmes et des systèmes pour extraire des connaissances et des informations significatives à partir de données. Elle combine des compétences en informatique, statistiques, mathématiques et domaine d'expertise pour analyser de grands ensembles de données et aider à la prise de décision et à la résolution de problèmes. 
# En d'autres termes, la data science permet d'exploiter les données pour comprendre le passé, analyser le présent et prédire le futur. 
# En résumé, la data science :
# Collecte, nettoie et transforme les données: pour les rendre exploitables.

# Utilise des techniques statistiques et d'apprentissage automatique: pour analyser les données et identifier des tendances et des modèles. 
# Développe des modèles prédictifs: pour anticiper les événements futurs. 
# Aide à la prise de décision: en fournissant des informations et des insights précieux. 
# Applications de la data science :


# Secteur privé:
# Amélioration de l'expérience client, optimisation des opérations, développement de nouveaux produits.
# Secteur public:
# Analyse des données pour améliorer les politiques publiques, lutte contre la criminalité, prévision des catastrophes naturelles.
# Recherche et développement:
# Avancées scientifiques dans divers domaines, de la médecine à l'astronomie.
# Métiers liés à la data science :


# Data scientist:
# Responsable de l'analyse des données, de la modélisation et de l'interprétation des résultats.
# Data analyst:
# Spécialisé dans l'analyse de données pour aider à la prise de décision.
# Data engineer:
# Responsable de la collecte, du stockage et de la préparation des données.



# Data manager:
# Responsable de la qualité et de la gestion des données.
# La data science est un domaine en constante évolution, avec des technologies et des méthodes qui se développent rapidement. Elle offre des opportunités passionnantes pour ceux qui souhaitent travailler avec les données et avoir un impact significatif dans divers domaines.


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
    

    
