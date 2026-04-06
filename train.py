import json
import numpy as np
import nltk
from nltk.stem import PorterStemmer
from sklearn.naive_bayes import MultinomialNB
import pickle

nltk.download('punkt')
nltk.download('punkt_tab')

stemmer = PorterStemmer()

with open('intents.json') as file:
    data = json.load(file)

words = []
labels = []
docs = []

for intent in data['intents']:
    for pattern in intent['patterns']:
        tokens = nltk.word_tokenize(pattern)
        words.extend(tokens)
        docs.append((tokens, intent['tag']))
        if intent['tag'] not in labels:
            labels.append(intent['tag'])

words = [stemmer.stem(w.lower()) for w in words if w.isalpha()]
words = sorted(set(words))

X = []
y = []

for doc in docs:
    bag = []
    token_words = [stemmer.stem(w.lower()) for w in doc[0]]

    for w in words:
        bag.append(1 if w in token_words else 0)

    X.append(bag)
    y.append(labels.index(doc[1]))

X = np.array(X)
y = np.array(y)

model = MultinomialNB()
model.fit(X, y)

pickle.dump((model, words, labels, data), open("model.pkl", "wb"))

print("✅ Model trained successfully!")