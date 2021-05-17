from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
import os
import numpy as np
from sklearn.model_selection import train_test_split


# DATA SHAPE:
# (num_features, num_timestamps, batch_size)
# num_features: size of the word embedding vector
# num_timestamps: max length of sentence in words
# batch_size: number of samples to go thru at once


def load_conversations(data_path='./data/movie_lines.txt', convo_path='./data/movie_conversations.txt'):
    """Loads data from the data paths."""
    lines = {}
    convos = []

    with open(data_path, 'r', encoding='iso-8859-1') as f:
        for line in f:
            line = line.split(' +++$+++ ')
            lines[line[0]] = line[4].rstrip()
    with open(convo_path, 'r', encoding='iso-8859-1') as f:
        for line in f:
            line = line.split(' +++$+++ ')
            arr = line[3].rstrip()
            arr = arr[2:-2]
            arr = arr.split('\', \'')
            #conv = [bos + lines[c] + eos for c in arr]
            conv = [lines[c] for c in arr]
            convos.append(conv)
    return convos


def tagger(decoder_input):
    bos = "<BOS> "
    eos = " <EOS>"
    
    target = [bos + line + eos for line in decoder_input]
    
    return target


# Vocab creator
def tokenize_words(convos, vocab_size):
    tokenizer = Tokenizer(num_words=vocab_size)
    tokenizer.fit_on_texts(convos)
    word_dict = tokenizer.word_index

    w2i = {}
    i2w = {}
    for word, idx in word_dict.items():
        if idx < vocab_size:
            w2i[word] = idx
            i2w[idx] = word

    return w2i, i2w


def text_to_seq(enc_text, dec_text, vocab_size):
    tokenizer = Tokenizer(num_words=vocab_size)
    tokenizer.fit_on_texts(enc_text + dec_text)
    print("length:",len(tokenizer.word_index))
    encoder_seq = tokenizer.texts_to_sequences(enc_text)
    decoder_seq = tokenizer.texts_to_sequences(dec_text)
    return encoder_seq, decoder_seq, tokenizer


def convos_to_xy(convos):
    """Turns conversation set into back and forth pairs of texts."""
    X = []
    y = []

    for conv in convos:
        for i in range(len(conv)-1):
            if len(conv[i]) <= 1 or len(conv[i+1]) <= 1:
                continue
            X.append(conv[i])
            y.append(conv[i+1])

    return X, y


def tokenize_xy(X, y, vocab_size):
    """Transforms word in X and y into integers up to top 200 most frequent words."""
    tokenizer = Tokenizer(num_words=vocab_size)
    tokenizer.fit_on_texts(y)
    X = tokenizer.texts_to_sequences(X)
    y = tokenizer.texts_to_sequences(y)
    return X, y


def padding(enc_seq, dec_seq, max_length):
    encoder_input = pad_sequences(enc_seq, maxlen=max_length, dtype='int32', padding='post', truncating='post')
    decoder_input = pad_sequences(dec_seq, maxlen=max_length, dtype='int32', padding='post', truncating='post')
    
    return encoder_input, decoder_input


def gen_embedding_matrix(embeddings_index, word_index, embedding_dim=100):
  embedding_matrix = np.zeros((len(word_index) + 1, embedding_dim))
  for word, i in word_index.items():
      embedding_vector = embeddings_index.get(word)
      if embedding_vector is not None:
          # words not found in embedding index will be all-zeros.
          embedding_matrix[i] = embedding_vector
  return embedding_matrix


def gen_glove_dict(glove_dir='../GloVe/'):
    f = open(os.path.join(glove_dir, 'glove.6B.100d.txt'))
    embedding_index = {}
    for line in f:
        values = line.split()
        word = values[0]
        coefs = np.asarray(values[1:], dtype='float32')
        embedding_index[word] = coefs
    
    f.close()
    return embedding_index



def decoder_output_creater(decoder_input_data, num_samples, MAX_LEN, VOCAB_SIZE):
  
    decoder_output_data = np.zeros((num_samples, MAX_LEN, VOCAB_SIZE), dtype="float32")

    for i, seqs in enumerate(decoder_input_data):
        for j, seq in enumerate(seqs):
            if j > 0:
                decoder_output_data[i][j][seq] = 1.
    print(decoder_output_data.shape)
    
    return decoder_output_data



def data_spliter(encoder_input_data, decoder_input_data, test_size1=0.2, test_size2=0.3):

    en_train, en_test, de_train, de_test = train_test_split(encoder_input_data, decoder_input_data, test_size=test_size1)
    en_train, en_val, de_train, de_val = train_test_split(en_train, de_train, test_size=test_size2)
    
    return en_train, en_val, en_test, de_train, de_val, de_test


if __name__ == '__main__':
    convos = load_conversations()
    print(convos[:10])
    X, y = convos_to_xy(convos)
    print(X[:10])
    X, y = tokenize_xy(X, y)
    print(X[:10])
