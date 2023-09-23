from flask import Flask, request, jsonify
import pickle
import re
import numpy as np
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
app = Flask(__name__)


# Load the saved model
with open('trained_model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

    # Load the saved tokenizer (you should have saved it during training)
    with open('tokenizer.pkl', 'rb') as tokenizer_file:
        tokenizer = pickle.load(tokenizer_file)

    # Preprocess the input phrase
    input_phrase = "neutral"  # Replace with your actual input
    input_phrase = input_phrase.lower()
    input_phrase = re.sub('[^a-zA-z0-9\s]', '', input_phrase)
    MAX_SEQUENCE_LENGTH = 393  # Replace with the same value you used for training
    # Tokenize and pad the input sequence
    input_seq = tokenizer.texts_to_sequences([input_phrase])
    input_seq = pad_sequences(input_seq, maxlen=MAX_SEQUENCE_LENGTH)  # Use the same MAX_SEQUENCE_LENGTH as during training

    # Make predictions
    predicted_sentiment = model.predict(input_seq)

    # Convert predictions to sentiment labels
    sentiment_labels = ['negative', 'positive']
    predicted_label = sentiment_labels[np.argmax(predicted_sentiment)]

    print("Predicted Sentiment:", predicted_label)


@app.route('/')
def home():
    return 'Hello World!'


if __name__ == '__main__':
    app.run(debug=True)
