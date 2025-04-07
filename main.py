import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras import Input
from tensorflow.keras.models import load_model
import os
import random
import json
import nltk
from nltk.stem.lancaster import LancasterStemmer

stemmer = LancasterStemmer()
nltk.download('punkt_tab')

with open('./data.json', 'r') as local_data:
    data = json.load(local_data)
    # print(data['intents'])

# Pre-processing
words = []
docs_questions = []
docs_intents = []
labels = []
for intent in data["intents"]:
    for question in intent["questions"]:
        tokens = nltk.word_tokenize(question)
        words.extend(tokens)
        docs_questions.append(tokens)
        docs_intents.append(intent['tag'])
    if intent["tag"] not in labels:
        labels.append(intent["tag"])

# stemming
words = [stemmer.stem(token.lower()) for token in words if token != ['>', '<', '\\', ':', '-', ',', '#','[' , ']', '/', '//', '_', '(', ')']]
words = sorted(list(set(words)))
labels = sorted(labels)

# creating one hot encoded Bag of Words
training = []
output = []
output_empty = [0 for _ in range(len(labels))]

for x, doc in enumerate(docs_questions):
    bow = []
    tokens = [stemmer.stem(token) for token in doc]

    # append words to bag of words list

    for token in words:
        if token in tokens:
            bow.append(1)
        else:
            bow.append(0)

    output_row = output_empty[:]
    output_row[labels.index(docs_intents[x])] = 1

    training.append(bow)
    output.append(output_row)
    # print(words, labels, training, output)


# convert bows to np.arrays
training = np.array(training)
output = np.array(output)

# ***MODEL BEGINS ***
model = Sequential()
model.add(Dense(8, input_shape=(len(training[0]),), activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(len(output[0]), activation='softmax'))
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

#  ***MODEL ENDS ***
if os.path.exists("./simplechat_model.keras"):
    model = load_model("./simplechat_model.keras")
else:
    model.fit(training, output, epochs=1000, batch_size=8, verbose=1)
    model.save("simplechat_model.keras")


# ***PREDICTIONS BEGIN ***
def bag_of_words(sentence, words):
    bag = np.zeros(len(words))
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [stemmer.stem(word.lower()) for word in sentence_words]

    for sw in sentence_words:
        for i, word in enumerate(words):
            if word == sw:
                bag[i] += 1

    return np.array(bag)


def predict():
    print("Chat with SimpleChat! (Type 'quit' to exit)")
    while True:
        inp = input("You: ")
        if inp.lower() == "quit":
            break
        vec = bag_of_words(inp, words)
        vec = vec.reshape(1, -1)  # Reshape to (1, len(words))
        results = model.predict(vec)[0]
        results_index = np.argmax(results)
        tag = labels[results_index]

        if results[results_index] > 0.7:
            for tg in data["intents"]:
                if tg["tag"] == tag:
                    responses = tg['responses']
            print(random.choice(responses))
        else:
            print("I don't quite understand. Try again or ask a different question.")

if __name__ == '__main__':
    predict()