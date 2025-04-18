import keras
import keras_hub
import tensorflow as tf
import numpy as np


@keras.saving.register_keras_serializable(package="transformerEncoder")
class TransformerEncoder(keras.layers.Layer):
    def __init__(self,hidden_dim,intermediate_dim,num_heads,dropout_rate=0.1,name=None, **kwargs):
        super().__init__(name=name, **kwargs)
        key_dim = hidden_dim
        self.intermediate_dim = intermediate_dim
        self.num_heads = num_heads
        self.dropout_rate = dropout_rate
        self.self_attention = keras.layers.MultiHeadAttention(num_heads,key_dim)
        self.self_attn_layer_norm = keras.layers.LayerNormalization()
        self.ff_1 = keras.layers.Dense(intermediate_dim,activation="relu")
        self.ff_2 = keras.layers.Dense(hidden_dim)
        self.ff_layer_norm = keras.layers.LayerNormalization()
        # self.dropout_layer=keras.layers.Dropout(dropout_rate)

    def call(self,source,source_mask):
      residual = x = source
      mask = source_mask[:,None,:]
      x = self.self_attention(query = x,value = x,key = x)#, attention_mask=tf.cast(mask, tf.float32)) # This is specifically required for M1 Mac
      x = x + residual
      x =self.self_attn_layer_norm(x)
      residual = x
      x = self.ff_1(x)
      x = self.ff_2(x)
      x = x+residual
      x = self.ff_layer_norm(x)
      return x


@keras.saving.register_keras_serializable(package="transformerDecoder")
class TransformerDecoder(keras.layers.Layer):
    def __init__(self, hidden_dim, intermediate_dim, num_heads,name=None, **kwargs):
        super().__init__(name=name, **kwargs)
        key_dim = hidden_dim // num_heads
        self.self_attention = keras.layers.MultiHeadAttention(num_heads, key_dim)
        self.self_attention_layernorm = keras.layers.LayerNormalization()
        self.cross_attention = keras.layers.MultiHeadAttention(num_heads, key_dim)
        self.cross_attention_layernorm = keras.layers.LayerNormalization()
        self.feed_forward_1 = keras.layers.Dense(intermediate_dim, activation="relu")
        self.feed_forward_2 = keras.layers.Dense(hidden_dim)
        self.feed_forward_layernorm = keras.layers.LayerNormalization()

    def call(self, target, source, source_mask):
        residual = x = target
        x = self.self_attention(query=x, key=x, value=x, use_causal_mask=True)
        x = x + residual
        x = self.self_attention_layernorm(x)
        residual = x
        mask = source_mask[:, None, :]
        x = self.cross_attention(
            query=x, key=source, value=source#, attention_mask=tf.cast(mask, tf.float32) # This is specifically required for M1 Mac
        )
        x = x + residual
        x = self.cross_attention_layernorm(x)
        residual = x
        x = self.feed_forward_1(x)
        x = self.feed_forward_2(x)
        x = x + residual
        x = self.feed_forward_layernorm(x)
        return x


def get_model(vocab_size, sequence_length, hidden_dim, intermediate_dim, num_heads):
    source = keras.Input(shape=(None,), dtype="int32", name="val_err")


    source_embedding_layer = keras.layers.Embedding(vocab_size, hidden_dim, name="source_embedding")
    source_pos_embedding = keras_hub.layers.PositionEmbedding(sequence_length=sequence_length, name="source_pos_emb")

    x_source = source_embedding_layer(source)
    x_source += source_pos_embedding(x_source)
    # x_source = keras.layers.Embedding(vocab_size, hidden_dim)(source)
    # x_source = keras_hub.layers.PositionEmbedding(
    #     sequence_length=sequence_length
    # )(x_source)
    encoder_output = TransformerEncoder(hidden_dim, intermediate_dim, num_heads, name="transformer_encoder")(
        source=x_source,
        source_mask=source != 0,
    )

    target = keras.Input(shape=(None,), dtype="int32", name="adj")
    target_embedding_layer = keras.layers.Embedding(vocab_size, hidden_dim, name="target_embedding")
    target_pos_embedding = keras_hub.layers.PositionEmbedding(sequence_length=sequence_length, name="target_pos_emb")
    x_target = target_embedding_layer(target)
    x_target += target_pos_embedding(x_target)
    # x_target = keras.layers.Embedding(vocab_size, hidden_dim)(target)
    # x_target = keras_hub.layers.PositionEmbedding(
    #     sequence_length=sequence_length
    # )(x_target)
    x = TransformerDecoder(hidden_dim, intermediate_dim, num_heads, name="transformer_decoder")(
        target=x_target,
        source=encoder_output,
        source_mask=source != 0,
    )
    x = keras.layers.Dropout(0.5)(x)
    target_predictions = keras.layers.Dense(vocab_size, activation="softmax")(x)
    transformer = keras.Model([source, target], target_predictions)
    return transformer


def compile_model(model, learning_rate=0.001, clipnorm=0.3):
    # transformer.compile(loss=keras.losses.SparseCategoricalCrossentropy(from_logits=False, reduction="sum_over_batch_size"),optimizer=keras.optimizers.Adam(learning_rate=0.001,clipnorm=0.3), weighted_metrics=['accuracy']) # /*clipvalue=0.1,,clipnorm=0.1*/

    model.compile(
        loss=keras.losses.SparseCategoricalCrossentropy(from_logits=False),
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate, clipnorm=clipnorm),
        weighted_metrics=["accuracy"]
    )
    return model


def save_model_with_weights(model, model_path="model.keras", weights_path="model_weights.h5"):
    model.save(model_path)
    model.save_weights(weights_path)


def load_model(model_path, custom_objects):
    return keras.models.load_model(model_path, custom_objects=custom_objects)

