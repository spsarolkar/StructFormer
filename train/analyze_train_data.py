import tensorflow as tf
import keras
from tensorboard import program
import numpy as np
import pickle
import os
import keras_hub
import sentencepiece as spm
import random

vocab_size=3000
sequence_length=100



# Load the .npz file
data = np.load("preprocessed/train_data.npz")

# Access individual arrays by key
encoder_inputs = data['encoder_input_ids']
decoder_inputs = data['decoder_input_ids']
decoder_labels     = data['decoder_labels']



sp = spm.SentencePieceProcessor()
sp.load("preprocessed/sentencepiece_model.model")

# print("encoder_input_ids shape:", encoder_inputs.shape)
# print("decoder_input_ids shape:", decoder_inputs.shape)
# print("decoder_labels shape:", decoder_labels.shape)



for _ in range(3):
    index = random.choice(np.arange(encoder_inputs.shape[0]))
    input_sentence_encoded = encoder_inputs[index]
    decoder_input_sentence_encoded = decoder_inputs[index]
    decoder_label_sentence_encoded = decoder_labels[index]
    print("input_sentence_encoded shape:", input_sentence_encoded)
    print("decoder_input_sentence_encoded shape:", decoder_input_sentence_encoded)
    print("decoder_label_sentence_encoded shape:", decoder_label_sentence_encoded)
    input_sentence = sp.decode(input_sentence_encoded.tolist())
    decoder_input_sentence = sp.decode(decoder_input_sentence_encoded.tolist())
    decoder_label_sentence = sp.decode(decoder_label_sentence_encoded.tolist())
    print('-'*10)
    print("input_sentence:", input_sentence)
    print("decoder_input_sentence:", decoder_input_sentence)
    print("decoder_label_sentence:", decoder_label_sentence)
    