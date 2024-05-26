#!/usr/bin/env python
# coding: utf-8

import random
import pickle
import numpy as np
import pandas as pd
from nltk.tokenize import RegexpTokenizer
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Activation
from tensorflow.keras.optimizers import RMSprop

def preprocess_text(file_path):
    text_df = pd.read_csv(file_path)
    text = list(text_df.text.values)
    joined_text = " ".join(text)
    with open("joined_text.txt", "w", encoding="utf-8") as f:
        f.write(joined_text)
    return joined_text

def tokenize_text(text, max_tokens=1000000):
    tokenizer = RegexpTokenizer(r"\w+")
    tokens = tokenizer.tokenize(text.lower()[:max_tokens])
    unique_tokens = np.unique(tokens)
    unique_token_index = {token: index for index, token in enumerate(unique_tokens)}
    return tokens, unique_tokens, unique_token_index

def create_datasets(tokens, unique_token_index, n_words=10):
    input_words = []
    next_word = []
    for i in range(len(tokens) - n_words):
        input_words.append(tokens[i:i + n_words])
        next_word.append(tokens[i + n_words])

    X = np.zeros((len(input_words), n_words, len(unique_token_index)), dtype=bool)
    y = np.zeros((len(next_word), len(unique_token_index)), dtype=bool)

    for i, words in enumerate(input_words):
        for j, word in enumerate(words):
            X[i, j, unique_token_index[word]] = 1
        y[i, unique_token_index[next_word[i]]] = 1

    return X, y

def build_model(input_shape, num_unique_tokens):
    model = Sequential()
    model.add(LSTM(128, input_shape=input_shape, return_sequences=True))
    model.add(LSTM(128))
    model.add(Dense(num_unique_tokens))
    model.add(Activation("softmax"))
    optimizer = RMSprop(learning_rate=0.01)
    model.compile(loss="categorical_crossentropy", optimizer=optimizer, metrics=["accuracy"])
    return model

def train_model(model, X, y, epochs=10, batch_size=128):
    history = model.fit(X, y, batch_size=batch_size, epochs=epochs, shuffle=True).history
    return model, history

def save_model_and_history(model, history, model_path="text_gen_model.h5", history_path="history.p"):
    model.save(model_path)
    with open(history_path, "wb") as f:
        pickle.dump(history, f)

def load_model_and_history(model_path="text_gen_model.h5", history_path="history.p"):
    model = load_model(model_path)
    history = pickle.load(open(history_path, "rb"))
    return model, history

def predict_next_word(model, input_text, unique_token_index, unique_tokens, n_words=10, n_best=5):
    input_text = input_text.lower()
    X = np.zeros((1, n_words, len(unique_token_index)))
    for i, word in enumerate(input_text.split()):
        X[0, i, unique_token_index[word]] = 1
    predictions = model.predict(X)[0]
    return np.argpartition(predictions, -n_best)[-n_best:]

def generate_text(model, input_text, unique_token_index, unique_tokens, n_words=10, num_words=100, creativity=3):
    tokenizer = RegexpTokenizer(r"\w+")
    word_sequence = input_text.split()
    current = 0
    for _ in range(num_words):
        sub_sequence = " ".join(tokenizer.tokenize(" ".join(word_sequence).lower())[current:current+n_words])
        try:
            choice = unique_tokens[random.choice(predict_next_word(model, sub_sequence, unique_token_index, unique_tokens, n_words, creativity))]
        except:
            choice = random.choice(unique_tokens)
        word_sequence.append(choice)
        current += 1
    return " ".join(word_sequence)

if __name__ == "__main__":
    joined_text = preprocess_text("fake_or_real_news.csv")
    tokens, unique_tokens, unique_token_index = tokenize_text(joined_text)
    X, y = create_datasets(tokens, unique_token_index)
    model = build_model((X.shape[1], X.shape[2]), len(unique_token_index))
    model, history = train_model(model, X, y)
    save_model_and_history(model, history)

    # Example usage
    print(generate_text(model, "I will have to look into this thing because I", unique_token_index, unique_tokens, 10, 100, 10))
    print(generate_text(model, "The president of the United States announced yesterday that he", unique_token_index, unique_tokens, 10, 100, 10))
