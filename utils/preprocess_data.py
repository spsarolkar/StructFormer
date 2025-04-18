from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
from transformers import AutoTokenizer
import pickle

import os
import keras
import keras_hub
import tensorflow as tf
from keras.models import Model
from sklearn.model_selection import train_test_split
import sentencepiece as spm
import os
# os.environ["KERAS_BACKEND"] = "torch"
import keras

keras.__version__
import keras_hub
from keras.models import Model
import sentencepiece as spm
from tensorflow.keras.preprocessing.sequence import pad_sequences

sequence_length=100
vocab_size=3000
PAD_ID=0
# Load data
errors_df = pd.read_csv('../data/validation_errors.csv')
adjustments_df = pd.read_csv('../data/generated_adjustments.csv')

# Merge on ErrorID
merged_df = pd.merge(errors_df, adjustments_df, on='ErrorID')

# Prepare input-output pairs
merged_df['input_text'] = ("TradeID=" + merged_df['TradeID'].astype(str) +
                           " AccountID=" + merged_df['AccountID'] +
                           " ErrorType=" + merged_df['ErrorType'])

merged_df['output_text'] = [adj for adj in merged_df['SQLAdjustment']]

# Final dataset
data = merged_df[['input_text', 'output_text']]


# Save data to text file for sentensepiece training
data.to_csv('../train/preprocessed/sentencepiece_training_data.csv', index=False)

# Train SentencePiece model
spm.SentencePieceTrainer.train(
    input='../train/preprocessed/sentencepiece_training_data.csv',
    model_prefix='../train/preprocessed/sentencepiece_model',
    vocab_size=vocab_size,  # you can tune this
    model_type='unigram',  # or 'bpe'
    pad_id=PAD_ID,
    unk_id=1,
    bos_id=2,
    eos_id=3,
    user_defined_symbols=['<SEP>', '<VAR>', '<NUM>']  # optional
)

#Create Train Test Validation Split

train_df, test_df = train_test_split(data, test_size=0.1, random_state=42)
train_df, val_df = train_test_split(train_df, test_size=0.2, random_state=42)

#tokenize the data
#Initialize the tokenizer


sp = spm.SentencePieceProcessor()
sp.load("../train/preprocessed/sentencepiece_model.model")


def tokenize_data(df, sp):
    input_text_list=df['input_text'].tolist()
    output_text_list=df['output_text'].tolist()
    encoder_input_tokenized=sp.encode(input_text_list, out_type=int, add_bos=True, add_eos=True)
    decoder_tokenized=sp.encode(output_text_list, out_type=int, add_bos=True, add_eos=True)

    
    # Convert to numpy
    encoder_input_tokenized= pad_sequences(
    encoder_input_tokenized,
    maxlen=sequence_length,
    dtype='int32',
    padding='post',      # or 'pre'
    truncating='post',   # or 'pre'
    value=PAD_ID         # usually sp.pad_id() or 0
    )
    decoder_inputs= pad_sequences(
    [seq[:-1] for seq in decoder_tokenized],
    maxlen=sequence_length,
    dtype='int32',
    padding='post',      # or 'pre'
    truncating='post',   # or 'pre'
    value=PAD_ID         # usually sp.pad_id() or 0
    )
    decoder_outputs= pad_sequences(
    [seq[1:] for seq in decoder_tokenized],
    maxlen=sequence_length,
    dtype='int32',
    padding='post',      # or 'pre'
    truncating='post',   # or 'pre'
    value=PAD_ID         # usually sp.pad_id() or 0
    )
    # Prepare decoder inputs and outputs
    # decoder_inputs = decoder_tokenized[:, :-1]
    # decoder_outputs = decoder_tokenized[:, 1:]
    # print('input_text_list',input_text_list[0])
    # print('output_text_list',output_text_list[0])
    # print("decoder_tokenized ", decoder_tokenized[0])
    # print("encoder_input_tokenized:", encoder_input_tokenized[0])
    # print("decoder_inputs shape:", decoder_inputs[0])
    # print("decoder_outputs shape:", decoder_outputs[0])


    return encoder_input_tokenized, decoder_inputs, decoder_outputs

# Tokenize training data
encoder_inputs, decoder_inputs, decoder_outputs = tokenize_data(train_df, sp)
# Tokenize validation data
val_encoder_inputs, val_decoder_inputs, val_decoder_outputs = tokenize_data(val_df, sp)
# Tokenize test data
test_encoder_inputs, test_decoder_inputs, test_decoder_outputs = tokenize_data(test_df, sp)




# Save processed arrays for training
np.savez("../train/preprocessed/train_data.npz",
         encoder_input_ids=encoder_inputs,
         decoder_input_ids=decoder_inputs,
         decoder_labels=decoder_outputs)

# Save processed arrays for validation
np.savez("../train/preprocessed/val_data.npz",
         encoder_input_ids=val_encoder_inputs,
         decoder_input_ids=val_decoder_inputs,
         decoder_labels=val_decoder_outputs)

# Save processed arrays for training
np.savez("../inference/data/test_data.npz",
         test_encoder_input_ids=test_encoder_inputs,
         test_decoder_input_ids=test_decoder_inputs,
         test_decoder_labels=test_decoder_outputs)



print("Training data prepared and saved.")
