import numpy as np
from keras.layers import Input, Embedding, LSTM, Dense, RepeatVector, Bidirectional, Dropout, merge, TimeDistributed
from keras.models import Model
from keras.layers import Dense
from keras.callbacks import ModelCheckpoint
from keras.preprocessing import sequence

from text_preprocessing import *
import os

MAX_LEN = 20
EMBEDDING_DIM = 100
VOCAB_SIZE = 1500
TF_CPP_MIN_LOG_LEVEL=2
batch_size = 128
epochs = 150
frac = 5


def seq2seq_model_builder(HIDDEN_DIM, embedding_layer):
    
    encoder_inputs = Input(shape=(MAX_LEN, ), dtype='int32',)
    encoder_embedding = embedding_layer(encoder_inputs)
    encoder_LSTM = LSTM(HIDDEN_DIM, return_state=True)
    encoder_outputs, state_h, state_c = encoder_LSTM(encoder_embedding)
    
    decoder_inputs = Input(shape=(MAX_LEN, ), dtype='int32',)
    decoder_embedding = embedding_layer(decoder_inputs)
    decoder_LSTM = LSTM(HIDDEN_DIM, return_state=True, return_sequences=True)
    decoder_outputs, _, _ = decoder_LSTM(decoder_embedding, initial_state=[state_h, state_c])
    
    # dense_layer = Dense(VOCAB_SIZE, activation='softmax')
    outputs = TimeDistributed(Dense(VOCAB_SIZE, activation='softmax'))(decoder_outputs)
    model = Model([encoder_inputs, decoder_inputs], outputs)
    
    return model


def train():
    model_filepath = os.path.join(os.getcwd(), 'ckpt', 'model')

    convos = load_conversations()
    enc_input, dec_input = convos_to_xy(convos[:int(len(convos)/frac)])
    dec_input = tagger(dec_input)

    tokenizer = Tokenizer(num_words=VOCAB_SIZE)
    tokenizer.fit_on_texts(enc_input + dec_input)
    enc_input = tokenizer.texts_to_sequences(enc_input)
    dec_input = tokenizer.texts_to_sequences(dec_input)

    enc_input, dec_input = padding(enc_input, dec_input, MAX_LEN)
    enc_train, enc_val, enc_test, dec_train, dec_val, dec_test = data_spliter(enc_input, dec_input)

    # One-hot encode decoder input to make targets (output)
    dec_output_train = decoder_output_creater(dec_train, len(dec_train), MAX_LEN, VOCAB_SIZE)
    dec_output_test = decoder_output_creater(dec_test, len(dec_test), MAX_LEN, VOCAB_SIZE)
    dec_output_val = decoder_output_creater(dec_val, len(dec_val), MAX_LEN, VOCAB_SIZE)

    word2idx, idx2word = tokenize_words(convos[:int(len(convos)/frac)], VOCAB_SIZE)
    embeddings_index = gen_glove_dict()
    emb_matrix = gen_embedding_matrix(embeddings_index, word2idx, EMBEDDING_DIM)
    
    embedding_layer = Embedding(input_dim = VOCAB_SIZE, 
                              output_dim = EMBEDDING_DIM,
                              input_length = MAX_LEN,
                              weights = [emb_matrix],
                              trainable = False)

    model = seq2seq_model_builder(300, embedding_layer)

    _, columns = os.popen('stty size', 'r').read().split()
    columns = int(columns)
    
    if os.path.exists(model_filepath):
        print("="*columns)
        print("Loading model from weights at", model_filepath)
        print("="*columns)

        model.load_weights(model_filepath)
    else:
        print("="*columns)
        print("Compiling new model")
        print("="*columns)

        model.compile(
            optimizer="Adam", loss="categorical_crossentropy", metrics=["accuracy"]
        )


    model_checkpoint_callback = ModelCheckpoint(
        filepath=model_filepath,
        save_weights_only=True,
        monitor='val_accuracy',
        mode='max',
        save_best_only=False,
        save_freq=batch_size*2)

    model.fit(
        [enc_train, dec_train],
        dec_output_train,
        batch_size=batch_size,
        epochs=epochs,
        validation_data=([enc_val, dec_val], dec_output_val),
        callbacks=[model_checkpoint_callback]
    )

    model.save('data/model.h5')


if __name__ == '__main__':
    train()