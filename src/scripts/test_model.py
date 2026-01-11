import tensorflow as tf
import pickle
import numpy as np

model = tf.keras.models.load_model("creature_model.keras")
vectorizer = model.vectorizer
vocab = vectorizer.get_vocabulary()
index_to_word = dict(enumerate(vocab))
word_to_index = {v: k for k, v in index_to_word.items()}

def encode(text):
    return vectorizer([text])

def predict(input_text, max_len=80):
    encoder_input = encode(input_text)

    # Start token
    decoded = ["<START>"]

    for _ in range(max_len):
        decoder_input = vectorizer([" ".join(decoded)])

        preds = model([encoder_input, decoder_input])
        next_token = np.argmax(preds[0, len(decoded)-1])

        word = index_to_word[next_token]

        if word == "<END>":
            break

        decoded.append(word)

    return " ".join(decoded[1:])


test = """POWER=6 TOUGHNESS=6
TYPE=Dragon
TEXT=Flying, trample. Whenever this creature deals combat damage to a player, draw a card."""

print(predict(test))