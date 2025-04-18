import tensorflow as tf
import keras
from tensorboard import program
import numpy as np
import pickle
import os

from model.transformer_blocks import TransformerEncoder, TransformerDecoder, get_model, save_model_with_weights, load_model,compile_model
# Model config
vocab_size=3000
sequence_length=100
hidden_dim = 256
intermediate_dim = 2048
num_heads = 8



# Define the Transformer model
transformer = get_model(vocab_size, sequence_length, hidden_dim, intermediate_dim, num_heads)
#Compile the model
transformer=compile_model(transformer, learning_rate=0.001, clipnorm=0.3)
transformer.summary()

# Tensorboard Setup
log_dir = os.path.join(f'logs')


def get_run_log_dir():
  import time
  run_id = time.strftime("run_%Y_%m_%d-%H_%M_%S")
  return os.path.join(log_dir, "tf", run_id)

tf_log=get_run_log_dir()

print(f'log_dir {tf_log}')

tb = program.TensorBoard()
tb.configure(argv=[None, '--logdir', tf_log])
url = tb.launch()
print(f"Tensorflow listening on {url}")






# Load the .npz file
data = np.load("preprocessed/train_data.npz")

# Access individual arrays by key
encoder_inputs = data['encoder_input_ids']
decoder_inputs = data['decoder_input_ids']
decoder_labels     = data['decoder_labels']

print("encoder_input_ids shape:", encoder_inputs.shape)
print("decoder_input_ids shape:", decoder_inputs.shape)
print("decoder_labels shape:", decoder_labels.shape)

# Load the .npz file
val_data = np.load("preprocessed/train_data.npz")
# Access individual arrays by key
val_encoder_input_ids = val_data['encoder_input_ids']
val_decoder_input_ids = val_data['decoder_input_ids']
val_decoder_labels     = val_data['decoder_labels']

print("val_encoder_input_ids shape:", val_encoder_input_ids.shape)
print("val_decoder_input_ids shape:", val_decoder_input_ids.shape)
print("val_decoder_labels shape:", val_decoder_labels.shape)

print(decoder_labels.shape, decoder_labels.dtype)
print(decoder_inputs.shape, decoder_inputs.dtype)
print(encoder_inputs.shape, encoder_inputs.dtype)
print(val_encoder_input_ids.shape, decoder_labels.dtype)
print(val_decoder_input_ids.shape, decoder_inputs.dtype)
print(val_decoder_labels.shape, encoder_inputs.dtype)


# for v in transformer.trainable_variables:
#     print(v.name, v.shape)
# Callbacks
csv_logger = keras.callbacks.CSVLogger(f'{tf_log}/csv.log', separator=',', append=False)
early_stopping = keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=7, restore_best_weights=True)
filepath = f"{tf_log}/model.keras"
# checkpoint = keras.callbacks.ModelCheckpoint(filepath, monitor='val_accuracy', verbose=1, save_best_only=True, mode='max')

checkpoint_cb = keras.callbacks.ModelCheckpoint(
    filepath=filepath,
    save_best_only=True,
    monitor='val_loss',
    mode='min',
    save_weights_only=False,  # important!
    verbose=1
)

tb_callback = keras.callbacks.TensorBoard(log_dir=tf_log)

history = transformer.fit(x=[encoder_inputs,decoder_inputs],y=decoder_labels,validation_data=([val_encoder_input_ids,val_decoder_input_ids], val_decoder_labels),batch_size=32, epochs=250, callbacks=[checkpoint_cb,early_stopping, tb_callback, csv_logger,]) #lr_scheduler

save_model_with_weights(transformer, model_path=f"{tf_log}/model.keras", weights_path=f"{tf_log}/model.weights.h5")
np.savez(os.path.join(tf_log, "history.npz"),
         history=history.history)