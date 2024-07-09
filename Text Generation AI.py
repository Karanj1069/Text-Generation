import random
import numpy as np
import pandas as pd
from nltk.tokenize import RegexpTokenizer
from sklearn.preprocessing import OneHotEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Activation
from tensorflow.keras.optimizers import RMSprop

def preprocess_text(file_path):
    text_df = pd.read_csv(file_path)
    text = " ".join(text_df.text.values).lower()  # Join all text and convert to lowercase
    return text

def tokenize_text(text, max_tokens=1000000):
    tokenizer = RegexpTokenizer(r"\w+")
    tokens = tokenizer.tokenize(text)[:max_tokens]
    unique_tokens = sorted(set(tokens))
    unique_token_index = {token: index for index, token in enumerate(unique_tokens)}
    return tokens, unique_tokens, unique_token_index

def create_datasets(tokens, unique_token_index, n_words=10):
    X, y = [], []
    for i in range(len(tokens) - n_words):
        X.append(tokens[i:i + n_words])
        y.append(tokens[i + n_words])
    
    # One-hot encode X and y using sklearn's OneHotEncoder
    encoder = OneHotEncoder(categories=[list(unique_token_index.values())], sparse=False)
    X_encoded = encoder.fit_transform([[unique_token_index[word] for word in words] for words in X])
    y_encoded = encoder.transform([[unique_token_index[word]] for word in y])
    
    return X_encoded, y_encoded

def build_model(input_shape, num_unique_tokens):
    model = Sequential([
        LSTM(128, input_shape=input_shape, return_sequences=True),
        LSTM(128),
        Dense(num_unique_tokens, activation="softmax")
    ])
    model.compile(loss="categorical_crossentropy", optimizer=RMSprop(learning_rate=0.01), metrics=["accuracy"])
    return model

def train_model(model, X, y, epochs=10, batch_size=128):
    history = model.fit(X, y, batch_size=batch_size, epochs=epochs, shuffle=True).history
    return model, history

def generate_text(model, input_text, unique_token_index, unique_tokens, n_words=10, num_words=100):
    tokenizer = RegexpTokenizer(r"\w+")
    word_sequence = input_text.lower().split()
    
    for _ in range(num_words):
        sub_sequence = " ".join(word_sequence[-n_words:])
        X = np.zeros((1, n_words, len(unique_token_index)))
        
        for i, word in enumerate(sub_sequence.split()):
            if word in unique_token_index:
                X[0, i, unique_token_index[word]] = 1
        
        predictions = model.predict(X)[0]
        next_word_index = np.argmax(predictions)
        next_word = unique_tokens[next_word_index]
        word_sequence.append(next_word)
        
    return " ".join(word_sequence)

if __name__ == "__main__":
    # Example usage
    file_path = "fake_or_real_news.csv"
    joined_text = preprocess_text(file_path)
    tokens, unique_tokens, unique_token_index = tokenize_text(joined_text)
    X, y = create_datasets(tokens, unique_token_index)
    
    model = build_model((X.shape[1], X.shape[2]), len(unique_token_index))
    model, history = train_model(model, X, y)
    
    input_texts = [
        "I will have to look into this thing because I",
        "The president of the United States announced yesterday that he"
    ]
    
    for input_text in input_texts:
        generated_text = generate_text(model, input_text, unique_token_index, unique_tokens)
        print(f"Input: {input_text}\nGenerated Text: {generated_text}\n")
