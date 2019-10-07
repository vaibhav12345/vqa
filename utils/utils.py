import codecs
import h5py

from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
import numpy as np

def createTokenizer(max_nb_words = 100000, oov_token = True, text):
  print("tokenizing input data...")
  tokenizer = Tokenizer(num_words=max_nb_words, oov_token=oov_token, lower=True, char_level=False)
  tokenizer.fit_on_texts(text)
  word_index = tokenizer.word_index
  print("dictionary size: ", len(word_index))
  return tokenizer, word_index


def createEncodedPaddedText(tokenizer, text):
  sequences = tokenizer.texts_to_sequences(text)
  padded_text = pad_sequences(sequences, maxlen=time_steps, padding='post')
  return sequences, padded_text

def saveToH5Py(data, fileName):
    with h5py.File(fileName + '.h5', 'w') as hf:
        hf.create_dataset(fileName,  data=data)

def readFromH5Py(fileName):
    with h5py.File(fileName + '.h5', 'r') as hf:
        data = hf[fileName][:]
    return data

def loadWordEmbeddings(filePath, embed_size = 300):
    #load embeddings
    print('loading word embeddings...')
    embeddings_index = {}
    f = codecs.open(filePath,  errors = 'ignore', encoding='utf-8')
    for line in f:
        values = line.split()
        word = ''.join(values[:-embed_size])
        coefs = np.asarray(values[-embed_size:], dtype='float32')
        embeddings_index[word] = coefs
    f.close()
    print('found %s word vectors' % len(embeddings_index))
    return embeddings_index

def createEmbeddingMatrix(word_index, embeddings_index):
    #embedding matrix
    print('preparing embedding matrix...')
    words_not_found = []
    nb_words = len(word_index)+1#vocabsize
    embedding_matrix = np.zeros((nb_words, embed_size))
    for word, i in word_index.items():
        embedding_vector = embeddings_index.get(word)
        if (embedding_vector is not None):
            # words not found in embedding index will be all-zeros.
            embedding_matrix[i] = embedding_vector
        else:
            words_not_found.append(word)
    print('number of null word embeddings: %d' % np.sum(np.sum(embedding_matrix, axis=1) == 0))
    return embedding_matrix, words_not_found