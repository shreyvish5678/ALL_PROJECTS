import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Embedding, Dropout
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

dataset = pd.read_csv('wordle.csv')

# Sample data (replace with your 12,000+ words and likelihoods)
words = list(dataset['word'])  # Add all your words here
likelihoods = list(dataset['occurrence'])  # Corresponding likelihoods

# Tokenize and pad sequences
tokenizer = Tokenizer(char_level=True)
tokenizer.fit_on_texts(words)
sequences = tokenizer.texts_to_sequences(words)
padded_sequences = pad_sequences(sequences, padding='post')

# Model architecture
model = Sequential()
model.add(Embedding(input_dim=len(tokenizer.word_index) + 1, output_dim=64, input_length=padded_sequences.shape[1]))
model.add(LSTM(128, return_sequences=True))
model.add(Dropout(0.5))
model.add(LSTM(128))
model.add(Dropout(0.5))
model.add(Dense(1, activation='linear'))

# Compile and train
model.compile(optimizer='adam', loss='mse')
model.fit(padded_sequences, np.array(likelihoods), epochs=200, batch_size=128, validation_split=0.1)

# Predict
predictions = model.predict(padded_sequences)
print(predictions)




