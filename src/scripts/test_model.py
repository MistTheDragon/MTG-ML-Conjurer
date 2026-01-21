import tensorflow as tf
import numpy as np

# Now we load the model
model = tf.keras.models.load_model("PT_model.keras")

# And rebuild the vectorizer from vocab.txt
vectorizer = tf.keras.layers.TextVectorization(
    max_tokens=12000,
    output_mode="int",
    output_sequence_length=256,
    vocabulary="PT_vocab.txt"
)

vocab = vectorizer.get_vocabulary()
idx_to_token = dict(enumerate(vocab))


# First we encode the tokens
def encode(text):
    return vectorizer([text])

# Then we generate an output using an encoded input
def generate(input_text, max_len=80):
    encoder_input = encode(input_text)

    # Decoder starts empty
    decoded_tokens = []

    for i in range(max_len):
        decoder_text = " ".join(decoded_tokens)
        decoder_input = encode(decoder_text)

        preds = model([encoder_input, decoder_input])
        token_id = np.argmax(preds[0, i])

        token = idx_to_token.get(token_id, "")

        # Stop if model outputs nothing
        if token == "" or token == "[UNK]":
            break

        decoded_tokens.append(token)

        return " ".join(decoded_tokens)


# Test to see if it works (it doesnt)
print(generate("POWER=6 TOUGHNESS=6"))
print(generate("POWER=3 TOUGHNESS=7"))
print(generate("POWER=3 TOUGHNESS=1"))