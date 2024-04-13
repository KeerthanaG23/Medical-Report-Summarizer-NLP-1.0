
# Import from standard library
import logging
import joblib
# Import from 3rd party libraries
import streamlit as st
import streamlit.components.v1 as components

import pandas as pd
import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences



df=pd.read_csv(r"C:\Users\Keerthana\Desktop\train.csv")
df=df[:5]




max_len = 75
x = df["input"]
y = "sostok " + df["output"] + " eostok"

token = Tokenizer()
token.fit_on_texts(x)
token.fit_on_texts(y)

# Helper Functions
def text_to_token(text):
    return token.texts_to_sequences(text)

def token_to_text(tok):
    return token.sequences_to_texts(tok)

x = text_to_token(x)
y = text_to_token(y)

x = pad_sequences(x,  maxlen=max_len, padding='post')
y = pad_sequences(y,  maxlen=max_len, padding='post')

Model = joblib.load(r"C:\Users\Keerthana\Desktop\Completed_model.joblib")

from tensorflow.keras.layers import Input, LSTM, Embedding, Dense, TimeDistributed
from tensorflow.keras.models import Model

latent_dim = 300
embedding_dim = 200

voc = len(token.word_index) + 1

# Encoder
encoder_inputs = Input(shape=(max_len, ))

# Embedding layer
enc_emb = Embedding(voc, embedding_dim, trainable=True)(encoder_inputs)

# Encoder LSTM 1
encoder_lstm1 = LSTM(latent_dim, return_sequences=True, return_state=True, dropout=0.4, recurrent_dropout=0.4)
(encoder_output1, state_h1, state_c1) = encoder_lstm1(enc_emb)

# Encoder LSTM 2
encoder_lstm2 = LSTM(latent_dim, return_sequences=True, return_state=True, dropout=0.4, recurrent_dropout=0.4)
(encoder_output2, state_h2, state_c2) = encoder_lstm2(encoder_output1)

# Encoder LSTM 3
encoder_lstm3 = LSTM(latent_dim, return_state=True, return_sequences=True, dropout=0.4, recurrent_dropout=0.4)
(encoder_outputs, state_h, state_c) = encoder_lstm3(encoder_output2)

# Set up the decoder, using encoder_states as the initial state
decoder_inputs = Input(shape=(None, ))

# Embedding layer
dec_emb_layer = Embedding(voc, embedding_dim, trainable=True)
dec_emb = dec_emb_layer(decoder_inputs)

# Decoder LSTM
decoder_lstm = LSTM(latent_dim, return_sequences=True, return_state=True, dropout=0.4, recurrent_dropout=0.2)
(decoder_outputs, decoder_fwd_state, decoder_back_state) = decoder_lstm(dec_emb, initial_state=[state_h, state_c])

# Dense layer
decoder_dense = TimeDistributed(Dense(voc, activation='softmax'))
decoder_outputs = decoder_dense(decoder_outputs)

# Define the model
model = Model([encoder_inputs, decoder_inputs], decoder_outputs)

    # Inference Models
# Encode the input sequence to get the feature vector
encoder_model = Model(inputs=encoder_inputs, outputs=[encoder_outputs, state_h, state_c])

# Decoder setup

# Below tensors will hold the states of the previous time step
decoder_state_input_h = Input(shape=(latent_dim, ))
decoder_state_input_c = Input(shape=(latent_dim, ))
decoder_hidden_state_input = Input(shape=(max_len, latent_dim))

# Get the embeddings of the decoder sequence
dec_emb2 = dec_emb_layer(decoder_inputs)

# To predict the next word in the sequence, set the initial states to the states from the previous time step
(decoder_outputs2, state_h2, state_c2) = decoder_lstm(dec_emb2, initial_state=[decoder_state_input_h, decoder_state_input_c])

# A dense softmax layer to generate prob dist. over the target vocabulary
decoder_outputs2 = decoder_dense(decoder_outputs2)

# Final decoder model
decoder_model = Model([decoder_inputs] + [decoder_hidden_state_input, decoder_state_input_h, decoder_state_input_c], [decoder_outputs2] + [state_h2, state_c2])
reverse_target_word_index = token.index_word
reverse_source_word_index = token.index_word
target_word_index = token.word_index

def decode_sequence(input_seq):

    # Encode the input as state vectors.
    (e_out, e_h, e_c) = encoder_model.predict(input_seq, verbose=0)

    # Generate empty target sequence of length 1
    target_seq = np.zeros((1, 1))

    # Populate the first word of target sequence with the start word.
    target_seq[0, 0] = target_word_index['sostok']

    stop_condition = False
    decoded_sentence = ''

    while not stop_condition:
        (output_tokens, h, c) = decoder_model.predict([target_seq] + [e_out, e_h, e_c], verbose=0)

        # Sample a token
        sampled_token_index = np.argmax(output_tokens[0, -1, :])
        sampled_token = reverse_target_word_index[sampled_token_index] if sampled_token_index != 0 else "eostok"

        if sampled_token != 'eostok':
            decoded_sentence += ' ' + sampled_token

        # Exit condition: either hit max length or find the stop word.
        if sampled_token == 'eostok' or sampled_token_index == 0:
            stop_condition = True

        # Update the target sequence (of length 1)
        target_seq = np.zeros((1, 1))
        target_seq[0, 0] = sampled_token_index

        # Update internal states
        (e_h, e_c) = (h, c)

    return decoded_sentence


import streamlit as st
from transformers import pipeline

# Load the summarization model
summarizer = pipeline("summarization", model="Falconsai/medical_summarization")

# Set up Streamlit app layout
st.set_page_config(
    page_title="Medical Text Summarizer",
    page_icon=":microscope:",
    layout="centered"
)

# App title and description
st.title("Medical Text Summarizer")
st.write("This app summarizes medical text using the Falcon-7B-Medical-Summarization model.")

# Input text area
input_text = st.text_area("Enter medical text to summarize:")

# Length of the summary
summary_length = st.slider("Summary Length", min_value=50, max_value=500, value=150, step=50)

# Button to generate summary
if st.button("Generate Summary"):
    if input_text.strip() != "":
        # Generate summary
        summary = summarizer(input_text, max_length=summary_length, min_length=int(summary_length/2), do_sample=False)[0]['summary_text']
        
        # Display summary
        st.subheader("Summary:")
        st.write(summary)
    else:
        st.warning("Please enter some medical text to summarize.")