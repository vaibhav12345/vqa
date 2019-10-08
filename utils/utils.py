import codecs
import h5py

from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
import numpy as np

def createTokenizer(text, max_nb_words = 100000, oov_token = True):
  print("tokenizing input data...")
  tokenizer = Tokenizer(num_words=max_nb_words, oov_token=oov_token, lower=True, char_level=False)
  tokenizer.fit_on_texts(text)
  word_index = tokenizer.word_index
  print("dictionary size: ", len(word_index))
  return tokenizer, word_index


def createEncodedPaddedText(tokenizer, text, time_steps):
  sequences = tokenizer.texts_to_sequences(text)
  padded_text = pad_sequences(sequences, maxlen=time_steps, padding='post')
  return sequences, padded_text

def saveToH5Py(data, filePath, fileName):
    with h5py.File(filePath + fileName + '.h5', 'w') as hf:
        hf.create_dataset(fileName,  data=data)

def readFromH5Py(filePath, fileName):
    with h5py.File(filePath + fileName + '.h5', 'r') as hf:
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

def createEmbeddingMatrix(word_index, embeddings_index, embed_size = 300):
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

contractions = {
"ain't": "am not",
"aren't": "are not",
"can't": "cannot",
"can't've": "cannot have",
"'cause": "because",
"could've": "could have",
"couldn't": "could not",
"couldn't've": "could not have",
"didn't": "did not",
"doesn't": "does not",
"don't": "do not",
"hadn't": "had not",
"hadn't've": "had not have",
"hasn't": "has not",
"haven't": "have not",
"he'd": "he would",
"he'd've": "he would have",
"he'll": "he will",
"he'll've": "he will have",
"he's": "he is",
"how'd": "how did",
"how'd'y": "how do you",
"how'll": "how will",
"how's": "how is",
"i'd": "I would",
"i'd've": "I would have",
"i'll": "I will",
"i'll've": "I will have",
"i'm": "I am",
"i've": "I have",
"isn't": "is not",
"it'd": "it would",
"it'd've": "it would have",
"it'll": "it will",
"it'll've": "it will have",
"it's": "it is",
"let's": "let us",
"ma'am": "madam",
"mayn't": "may not",
"might've": "might have",
"mightn't": "might not",
"mightn't've": "might not have",
"must've": "must have",
"mustn't": "must not",
"mustn't've": "must not have",
"needn't": "need not",
"needn't've": "need not have",
"o'clock": "of the clock",
"oughtn't": "ought not",
"oughtn't've": "ought not have",
"shan't": "shall not",
"sha'n't": "shall not",
"shan't've": "shall not have",
"she'd": "sshe would",
"she'd've": "she would have",
"she'll": "she will",
"she'll've": "she will have",
"she's": "she is",
"should've": "should have",
"shouldn't": "should not",
"shouldn't've": "should not have",
"so've": "so have",
"so's": "so is",
"that'd": "that would",
"that'd've": "that would have",
"that's": "that is",
"there'd": "there would",
"there'd've": "there would have",
"there's": "there is",
"they'd": "they would",
"they'd've": "they would have",
"they'll": "they will",
"they'll've": "they will have",
"they're": "they are",
"they've": "they have",
"to've": "to have",
"wasn't": "was not",
"we'd": "we would",
"we'd've": "we would have",
"we'll": "we will",
"we'll've": "we will have",
"we're": "we are",
"we've": "we have",
"weren't": "were not",
"what'll": "what will",
"what'll've": "what will have",
"what're": "what are",
"what's": "what is",
"what've": "what have",
"when's": "when is",
"when've": "when have",
"where'd": "where did",
"where's": "where is",
"where've": "where have",
"who'll": "who will",
"who'll've": "who will have",
"who's": "who is",
"who've": "who have",
"why's": "why is",
"why've": "why have",
"will've": "will have",
"won't": "will not",
"won't've": "will not have",
"would've": "would have",
"wouldn't": "would not",
"wouldn't've": "would not have",
"y'all": "you all",
"y'all'd": "you all would",
"y'all'd've": "you all would have",
"y'all're": "you all are",
"y'all've": "you all have",
"you'd": "you would",
"you'd've": "you would have",
"you'll": "you will",
"you'll've": "you will have",
"you're": "you are",
"you've": "you have"
}

def removeApostrophe(text):
    for word in text.split():
        if word.lower() in contractions:
            text = text.replace(word, contractions[word.lower()])
        elif len(word)>=3 and word[-1]=='s' and word[-2]=="'":
            text = text.replace(word,word[:-2])
    return text