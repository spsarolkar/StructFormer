import sentencepiece as spm
from tensorflow.keras.preprocessing.sequence import pad_sequences


def tokenize_data(plain_text, sp,sequence_length,PAD_ID=0,add_bos=True,add_eos=True):

    plain_text_tokenized=sp.encode(plain_text, out_type=int, add_bos=add_bos, add_eos=add_eos)
    # Convert to numpy
    plain_text_tokenized= pad_sequences(
    plain_text_tokenized,
    maxlen=sequence_length,
    dtype='int32',
    padding='post',      # or 'pre'
    truncating='post',   # or 'pre'
    value=PAD_ID         # usually sp.pad_id() or 0
    )


    return plain_text_tokenized


def decode_data(df,encoded, sp):

    plain_text_tokenized=sp.decode(encoded)
    # Convert to numpy
    plain_text_tokenized= pad_sequences(
    plain_text_tokenized,
    maxlen=sequence_length,
    dtype='int32',
    padding='post',      # or 'pre'
    truncating='post',   # or 'pre'
    value=PAD_ID         # usually sp.pad_id() or 0
    )


    return encoder_input_tokenized, decoder_inputs, decoder_outputs