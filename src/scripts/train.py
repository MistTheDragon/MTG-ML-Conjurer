#import os
#os.environ["PYTHONIOENCODING"] = "utf-8"

#import sys
#sys.stdout.reconfigure(encoding="utf-8")

import pandas as pd
import tensorflow as tf
from tensorflow.keras.layers import TextVectorization
from tensorflow.keras import layers, Model


# First we create tokens based off of the card text
training = pd.read_csv(r"MTG-ML-Conjurer\src\scryfall\creature_train.csv", encoding="utf-8")

inputs = training["input"].values
outputs = training["output"].values

#creating the vocab.txt file 
vectorizer = TextVectorization(
    max_tokens=12000,
    output_mode="int",
    output_sequence_length=256)

vectorizer.adapt(list(training["input"]) + list(training["output"]))
vocab = vectorizer.get_vocabulary()

with open("vocab.txt", "w", encoding="utf-8") as f:
    for token in vocab:
        if token.strip() != "":
            f.write(token + "\n")

vectorizer = TextVectorization(
    max_tokens=12000,
    output_mode="int",
    output_sequence_length=256,
    vocabulary="vocab.txt"
)

# Now we create model structure (Input text → Encoder → Decoder → Output tokens)
embed_dim = 256
latent_dim = 512

encoder_inputs = layers.Input(shape=(None,))
x = layers.Embedding(12000, embed_dim)(encoder_inputs)
_, state_h, state_c = layers.LSTM(latent_dim, return_state=True)(x)
encoder_states = [state_h, state_c]

decoder_inputs = layers.Input(shape=(None,))
x = layers.Embedding(12000, embed_dim)(decoder_inputs)
x = layers.LSTM(latent_dim, return_sequences=True)(x, initial_state=encoder_states)
decoder_outputs = layers.Dense(12000, activation="softmax")(x)

model = Model([encoder_inputs, decoder_inputs], decoder_outputs)
model.compile(loss="sparse_categorical_crossentropy", optimizer="adam")

# Save the model for use in test_model.py
model.vectorizer = vectorizer
model.save("creature_model.keras")
