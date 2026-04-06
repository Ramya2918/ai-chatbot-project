from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import pickle
import nltk
from nltk.stem import PorterStemmer
import numpy as np
import random
import datetime

stemmer = PorterStemmer()

# Load trained model
model, words, labels, data = pickle.load(open("model.pkl", "rb"))

app = Flask(__name__)
CORS(app)

# Home route
@app.route('/')
def home():
    return render_template("index.html")

# Convert text → numbers
def bag_of_words(sentence):
    tokens = nltk.word_tokenize(sentence.lower())
    tokens = [stemmer.stem(w) for w in tokens]
    return np.array([1 if w in tokens else 0 for w in words])

# Chatbot logic
def get_response(msg):
    msg_lower = msg.lower()

    # 🔥 Smart features
    if "time" in msg_lower:
        return "Current time is " + datetime.datetime.now().strftime("%H:%M")

    if "stress" in msg_lower or "motivate" in msg_lower:
        return "Stay consistent and believe in yourself 💪 You can do it!"

    if "name" in msg_lower:
        return "I am your Smart Student Assistant 🤖"

    # ML prediction
    bow = bag_of_words(msg)
    probs = model.predict_proba([bow])[0]
    max_prob = max(probs)
    result_index = np.argmax(probs)

    # 🔥 Smart fallback
    if max_prob < 0.4:
        return "I'm not sure. Try asking about study, internship, placement, or interview tips."

    tag = labels[result_index]

    for intent in data['intents']:
        if intent['tag'] == tag:
            return random.choice(intent['responses'])

    return "I don't understand."

# API
@app.route('/chat', methods=['POST'])
def chat():
    user_msg = request.json['message']
    response = get_response(user_msg)
    return jsonify({"response": response})

if __name__ == '__main__':
    app.run(debug=True)