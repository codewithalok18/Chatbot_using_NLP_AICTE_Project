
import os
import json
import datetime
import csv
import nltk
import ssl
import streamlit as st
import random
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

ssl._create_default_https_context = ssl._create_unverified_context
nltk.data.path.append(os.path.abspath("nltk_data"))
nltk.download('punkt')

# Load intents from the JSON file
file_path = os.path.abspath(r"C:\Users\alokk\Implementation of chatBot using NLP/intents.json")
with open(file_path, "r") as file:
    intents = json.load(file)

# Create the vectorizer and classifier
vectorizer = TfidfVectorizer()
clf = LogisticRegression(random_state=0, max_iter=10000)

# Preprocess the data
tags = []
patterns = []
for intent in intents:
    for pattern in intent['patterns']:
        tags.append(intent['tag'])
        patterns.append(pattern)

# training the model
x = vectorizer.fit_transform(patterns)
y = tags
clf.fit(x, y)

def chatbot(input_text):
    input_text = vectorizer.transform([input_text])
    tag = clf.predict(input_text)[0]
    for intent in intents:
        if intent['tag'] == tag:
            response = random.choice(intent['responses'])
            return response
        
counter = 0

def main():
    global counter
    st.markdown("<h1 style='text-align: center; color: #4CAF50;'>Implementation Of Chatbot with NLP</h1>", unsafe_allow_html=True)

    # Sidebar menu
    menu = ["Home", "Conversation History", "About"]
    choice = st.sidebar.selectbox("Menu", menu)

    if choice == "Home":
        st.markdown("<p style='text-align: center; color: #555;'>Welcome to the chatbot! Start chatting below.</p>", unsafe_allow_html=True)

        if not os.path.exists('chat_log.csv'):
            with open('chat_log.csv', 'w', newline='', encoding='utf-8') as csvfile:
                csv_writer = csv.writer(csvfile)
                csv_writer.writerow(['User Input', 'Chatbot Response', 'Timestamp'])

        counter += 1
        user_input = st.text_input("You:", key=f"user_input_{counter}")

        if user_input:
            user_input_str = str(user_input)
            response = chatbot(user_input)
            st.markdown(f"<div style='background-color:#8507e8 ; padding:10px; border-radius:10px;'>{response}</div>", unsafe_allow_html=True)

            timestamp = datetime.datetime.now().strftime(f"%Y-%m-%d %H:%M:%S")

            with open('chat_log.csv', 'a', newline='', encoding='utf-8') as csvfile:
                csv_writer = csv.writer(csvfile)
                csv_writer.writerow([user_input_str, response, timestamp])

            if response.lower() in ['goodbye', 'bye']:
                st.markdown("<p style='text-align: center; color: #FF5733;'>Thank you for chatting! Have a great day!</p>", unsafe_allow_html=True)
                st.stop()

    elif choice == "Conversation History":
        st.markdown("<h2 style='color: #3358ff;'>Conversation History</h2>", unsafe_allow_html=True)
        with open('chat_log.csv', 'r', encoding='utf-8') as csvfile:
            csv_reader = csv.reader(csvfile)
            next(csv_reader)
            for row in csv_reader:
                st.markdown(f"<div style='background-color:#ff5733; padding:5px; border-radius:5px;'><b>User:</b> {row[0]}<br><b>Chatbot:</b> {row[1]}<br><b>Timestamp:</b> {row[2]}</div>", unsafe_allow_html=True)
                st.markdown("---")

    elif choice == "About":
        st.markdown("<h2 style='color: #e807c9 ;'>About the Project</h2>", unsafe_allow_html=True)
        st.write("The goal of this project is to create a chatbot that can understand and respond to user input based on intents. The chatbot is built using NLP techniques and Logistic Regression for accurate response generation.")
        st.markdown("<h3>Project Overview:</h3>", unsafe_allow_html=True)
        st.write("- NLP techniques and Logistic Regression are used for training.")
        st.write("- Streamlit web framework is used for building the interface.")

        st.markdown("<h3>Dataset:</h3>", unsafe_allow_html=True)
        st.write("The dataset contains intents with multiple patterns and responses.")

        st.markdown("<h3>Conclusion:</h3>", unsafe_allow_html=True)
        st.write("This chatbot can be extended using more sophisticated NLP techniques or deep learning models for improved responses.")

if __name__ == '__main__':
    main()

