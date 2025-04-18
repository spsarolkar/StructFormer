import tensorflow as tf
import sentencepiece as spm
import random
import numpy as np
import tensorflow as tf
from structformer_utils import tokenize_data
import keras
import keras_hub
import sys
sys.path.append("..")
from train.model.transformer_blocks import TransformerEncoder, TransformerDecoder, get_model, save_model_with_weights, load_model,compile_model

sp = spm.SentencePieceProcessor()
sp.load("../train/preprocessed/sentencepiece_model.model")
# spa_index_lookup = dict(zip(range(len(spa_vocab)), spa_vocab))
data = np.load("../train/preprocessed/train_data.npz")
# Access individual arrays by key
encoder_inputs = data['encoder_input_ids']
decoder_inputs = data['decoder_input_ids']
decoder_labels = data['decoder_labels']
START_TOKEN=sp.piece_to_id("<s>")
END_TOKEN=sp.piece_to_id("</s>")
PAD_ID=sp.pad_id()

# Model config
vocab_size = 3000
sequence_length = 100
hidden_dim = 256
intermediate_dim = 2048
num_heads = 8

# Load model
transformer_model = get_model(vocab_size, sequence_length, hidden_dim, intermediate_dim, num_heads)

transformer_model = load_model(
    "../train/logs/tf/run_2025_04_18-18_23_18/model.keras",
    # compile=True  # will recompile with previously saved optimizer config
         custom_objects={
        "TransformerEncoder": TransformerEncoder,
        "TransformerEncoder": TransformerDecoder,
    }
)
# compile_model(model)
# model.load_weights("logs/tf/run_2025_04_18-18_23_18/model.keras")


def generate_output(tokenized_input_sentence):
    tokenized_decoder_input = tf.constant([[START_TOKEN]], dtype=tf.int32)
    for _ in range(sequence_length):
        padded_decoder_input = tf.concat(
            [tokenized_decoder_input, tf.zeros((1, sequence_length - tf.shape(tokenized_decoder_input)[1]), dtype=tf.int32)],
            axis=1
        )
        prediction = transformer_model.predict([tokenized_input_sentence, padded_decoder_input], verbose=0)
        next_token_logits = prediction[0, tokenized_decoder_input.shape[1] - 1, :]
        next_token_id = np.argmax(next_token_logits).astype(np.int32)

        tokenized_decoder_input = tf.concat([tokenized_decoder_input, [[next_token_id]]], axis=-1)
        if next_token_id == END_TOKEN:
            break

    output_ids = tokenized_decoder_input.numpy().flatten().tolist()[1:]
    if END_TOKEN in output_ids:
        output_ids = output_ids[:output_ids.index(END_TOKEN)]
    return sp.decode(output_ids)


# Example usage
if __name__ == "__main__":
    index = random.choice(np.arange(encoder_inputs.shape[0]))
    input_encoded = encoder_inputs[index]
    input_sentence = sp.decode(input_encoded.tolist())

    print("\n—")
    print("🧾 Input (decoded):", input_sentence)
    print("🎯 Expected:", sp.decode(decoder_labels[index].tolist()))
    print("🧪 Predicted:", generate_output(np.expand_dims(input_encoded, axis=0)))






# print('START_TOKEN',START_TOKEN)
# print('END_TOKEN',END_TOKEN)
# print('PAD_ID',PAD_ID)

# from keras.models import load_model
# from keras.losses import SparseCategoricalCrossentropy

# @keras.saving.register_keras_serializable(package="transformerEncoder")
# class TransformerEncoder(keras.layers.Layer):
#     def __init__(self,hidden_dim,intermediate_dim,num_heads,dropout_rate=0.1,name=None, **kwargs):
#         super().__init__(name=name, **kwargs)
#         key_dim = hidden_dim
#         self.intermediate_dim = intermediate_dim
#         self.num_heads = num_heads
#         self.dropout_rate = dropout_rate
#         self.self_attention = keras.layers.MultiHeadAttention(num_heads,key_dim)
#         self.self_attn_layer_norm = keras.layers.LayerNormalization()
#         self.ff_1 = keras.layers.Dense(intermediate_dim,activation="relu")
#         self.ff_2 = keras.layers.Dense(hidden_dim)
#         self.ff_layer_norm = keras.layers.LayerNormalization()
#         # self.dropout_layer=keras.layers.Dropout(dropout_rate)

#     def call(self,source,source_mask):
#       residual = x = source
#       mask = source_mask[:,None,:]
#       x = self.self_attention(query = x,value = x,key = x)#, attention_mask=tf.cast(mask, tf.float32)) # This is specifically required for M1 Mac
#       x = x + residual
#       x =self.self_attn_layer_norm(x)
#       residual = x
#       x = self.ff_1(x)
#       x = self.ff_2(x)
#       x = x+residual
#       x = self.ff_layer_norm(x)
#       return x
    
#     def build(self, input_shape):
#         # No new weights to create manually, but mark it as built
#         super().build(input_shape)

# @keras.saving.register_keras_serializable(package="transformerDecoder")
# class TransformerDecoder(keras.layers.Layer):
#     def __init__(self, hidden_dim, intermediate_dim, num_heads, **kwargs):
#         super().__init__(**kwargs)
#         key_dim = hidden_dim // num_heads
#         self.self_attention = keras.layers.MultiHeadAttention(num_heads, key_dim)
#         self.self_attention_layernorm = keras.layers.LayerNormalization()
#         self.cross_attention = keras.layers.MultiHeadAttention(num_heads, key_dim)
#         self.cross_attention_layernorm = keras.layers.LayerNormalization()
#         self.feed_forward_1 = keras.layers.Dense(intermediate_dim, activation="relu")
#         self.feed_forward_2 = keras.layers.Dense(hidden_dim)
#         self.feed_forward_layernorm = keras.layers.LayerNormalization()

#     def call(self, target, source, source_mask):
#         residual = x = target
#         x = self.self_attention(query=x, key=x, value=x, use_causal_mask=True)
#         x = x + residual
#         x = self.self_attention_layernorm(x)
#         residual = x
#         mask = source_mask[:, None, :]
#         x = self.cross_attention(
#             query=x, key=source, value=source#, attention_mask=tf.cast(mask, tf.float32) # This is specifically required for M1 Mac
#         )
#         x = x + residual
#         x = self.cross_attention_layernorm(x)
#         residual = x
#         x = self.feed_forward_1(x)
#         x = self.feed_forward_2(x)
#         x = x + residual
#         x = self.feed_forward_layernorm(x)
#         return x

#     def build(self, input_shape):
#         # No new weights to create manually, but mark it as built
#         super().build(input_shape)


# hidden_dim = 256
# intermediate_dim = 2048
# num_heads = 8

# source = keras.Input(shape=(None,), dtype="int32", name="val_err")


# source_embedding_layer = keras.layers.Embedding(vocab_size, hidden_dim, name="source_embedding")
# source_pos_embedding = keras_hub.layers.PositionEmbedding(sequence_length=sequence_length, name="source_pos_emb")

# x_source = source_embedding_layer(source)
# x_source += source_pos_embedding(x_source)
# # x_source = keras.layers.Embedding(vocab_size, hidden_dim)(source)
# # x_source = keras_hub.layers.PositionEmbedding(
# #     sequence_length=sequence_length
# # )(x_source)
# encoder_output = TransformerEncoder(hidden_dim, intermediate_dim, num_heads)(
#     source=x_source,
#     source_mask=source != 0,
# )

# target = keras.Input(shape=(None,), dtype="int32", name="adj")
# target_embedding_layer = keras.layers.Embedding(vocab_size, hidden_dim, name="target_embedding")
# target_pos_embedding = keras_hub.layers.PositionEmbedding(sequence_length=sequence_length, name="target_pos_emb")
# x_target = target_embedding_layer(target)
# x_target += target_pos_embedding(x_target)
# # x_target = keras.layers.Embedding(vocab_size, hidden_dim)(target)
# # x_target = keras_hub.layers.PositionEmbedding(
# #     sequence_length=sequence_length
# # )(x_target)
# x = TransformerDecoder(hidden_dim, intermediate_dim, num_heads)(
#     target=x_target,
#     source=encoder_output,
#     source_mask=source != 0,
# )
# x = keras.layers.Dropout(0.5)(x)
# target_predictions = keras.layers.Dense(vocab_size, activation="softmax")(x)
# transformer = keras.Model([source, target], target_predictions)



# # transformer.load_weights(
# #     "../train/logs/tf/run_2025_04_18-00_16_31/model.keras",)

# transformer_model = load_model(
#     "../train/logs/tf/run_2025_04_18-14_42_42/model.keras",
#     # compile=True  # will recompile with previously saved optimizer config
#          custom_objects={
#         "TransformerEncoder": TransformerEncoder,
#         "TransformerEncoder": TransformerDecoder,
#     }
# )

# def generate_output(tokenized_input_sentence):
#     tokenized_decoder_input = tf.constant([[START_TOKEN]], dtype=tf.int32)

#     for _ in range(sequence_length):
#         pad_len = sequence_length - tokenized_decoder_input.shape[1]
#         tokenized_decoder_input_padded = tf.concat([
#             tokenized_decoder_input,
#             tf.zeros((1, pad_len), dtype=tf.int32)
#         ], axis=1)

#         # Predict
#         predictions = transformer.predict([tokenized_input_sentence, tokenized_decoder_input_padded], verbose=0)
#         next_token_logits = predictions[0, tokenized_decoder_input.shape[1] - 1, :]

#         sampled_token_index = int(np.argmax(next_token_logits))
#         tokenized_decoder_input = tf.concat([
#             tokenized_decoder_input,
#             tf.constant([[sampled_token_index]], dtype=tf.int32)
#         ], axis=1)

#         if sampled_token_index == END_TOKEN:
#             break

#     # Convert to list and trim BOS and EOS
#     decoded_ids = tokenized_decoder_input.numpy().flatten().tolist()
#     decoded_ids = decoded_ids[1:]  # Remove BOS
#     if END_TOKEN in decoded_ids:
#         decoded_ids = decoded_ids[:decoded_ids.index(END_TOKEN)]
#     decoded_sentence = sp.decode(decoded_ids)
#     return decoded_sentence


# for _ in range(3):
#     index = random.choice(np.arange(encoder_inputs.shape[0]))
#     input_encoded = encoder_inputs[index]
#     input_sentence = sp.decode(input_encoded.tolist())

#     print("\n—")
#     print("🧾 Input (decoded):", input_sentence)
#     print("🎯 Expected:", sp.decode(decoder_labels[index].tolist()))
#     print("🧪 Predicted:", generate_output(np.expand_dims(input_encoded, axis=0)))