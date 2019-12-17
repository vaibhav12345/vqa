from keras.layers import concatenate, Embedding, Bidirectional, LSTM, Input, Dense, merge, Lambda, Multiply, Dropout
from keras.models import load_model,Model
from keras.initializers import glorot_uniform
from keras.engine.topology import Layer
from keras import initializers, regularizers, constraints
from keras.applications.vgg16 import VGG16
import keras.backend as K

class Attention(Layer):
    def __init__(self, step_dim,
                 W_regularizer=None, b_regularizer=None,
                 W_constraint=None, b_constraint=None,
                 bias=True, **kwargs):
        """
        Keras Layer that implements an Attention mechanism for temporal data.
        Supports Masking.
        Follows the work of Raffel et al. [https://arxiv.org/abs/1512.08756]
        # Input shape
            3D tensor with shape: `(samples, steps, features)`.
        # Output shape
            2D tensor with shape: `(samples, features)`.
        :param kwargs:
        Just put it on top of an RNN Layer (GRU/LSTM/SimpleRNN) with return_sequences=True.
        The dimensions are inferred based on the output shape of the RNN.
        Example:
            model.add(LSTM(64, return_sequences=True))
            model.add(Attention())
        """
        self.supports_masking = True
        #self.init = initializations.get('glorot_uniform')
        self.init = initializers.get('glorot_uniform')

        self.W_regularizer = regularizers.get(W_regularizer)
        self.b_regularizer = regularizers.get(b_regularizer)

        self.W_constraint = constraints.get(W_constraint)
        self.b_constraint = constraints.get(b_constraint)

        self.bias = bias
        self.step_dim = step_dim
        self.features_dim = 0
        super(Attention, self).__init__(**kwargs)

    def build(self, input_shape):
        assert len(input_shape) == 3

        self.W = self.add_weight((input_shape[-1],),
                                 initializer=self.init,
                                 name='{}_W'.format(self.name),
                                 regularizer=self.W_regularizer,
                                 constraint=self.W_constraint)
        self.features_dim = input_shape[-1]

        if self.bias:
            self.b = self.add_weight((input_shape[1],),
                                     initializer='zero',
                                     name='{}_b'.format(self.name),
                                     regularizer=self.b_regularizer,
                                     constraint=self.b_constraint)
        else:
            self.b = None

        self.built = True

    def compute_mask(self, input, input_mask=None):
        # do not pass the mask to the next layers
        return None

    def call(self, x, mask=None):
        # eij = K.dot(x, self.W) TF backend doesn't support it

        # features_dim = self.W.shape[0]
        # step_dim = x._keras_shape[1]

        features_dim = self.features_dim
        step_dim = self.step_dim

        eij = K.reshape(K.dot(K.reshape(x, (-1, features_dim)), K.reshape(self.W, (features_dim, 1))), (-1, step_dim))

        if self.bias:
            eij += self.b

        eij = K.tanh(eij)

        a = K.exp(eij)

        # apply mask after the exp. will be re-normalized next
        if mask is not None:
            # Cast the mask to floatX to avoid float64 upcasting in theano
            a *= K.cast(mask, K.floatx())

        # in some cases especially in the early stages of training the sum may be almost zero
        a /= K.cast(K.sum(a, axis=1, keepdims=True) + K.epsilon(), K.floatx())

        a = K.expand_dims(a)
        weighted_input = x * a
    #print weigthted_input.shape
        return K.sum(weighted_input, axis=1)

    def compute_output_shape(self, input_shape):
        #return input_shape[0], input_shape[-1]
        return input_shape[0],  self.features_dim



def vgg16Model(num_classes=1000):
    model = VGG16(include_top=True, weights='imagenet', classes=num_classes)
    ip = model.input
    op = model.layers[-2].output
    vgg16Model = Model(ip,op)
    for layer in vgg16Model.layers:
        layer.trainable = False
    return vgg16Model

def lstmModel(embedding_matrix, trainable, embed_size, vocab_size, time_steps, unit_length):
    inputs = Input(shape=(time_steps,))
    x = Embedding(output_dim=embed_size, input_dim=vocab_size, weights=[embedding_matrix],input_length=time_steps, trainable=trainable)(inputs)
    # default merge_mode is concatenate others are mul,ave,sum
    y= Bidirectional(LSTM(unit_length, return_sequences=True), input_shape=(time_steps, 1), merge_mode='mul')(x)
    a= Bidirectional(LSTM(unit_length, return_sequences=False), input_shape=(time_steps, 1), merge_mode='sum')(y)
    model = Model(inputs=inputs, outputs=a)
    return model
def vqaModelSimple(embedding_matrix, trainable=False, num_classes=1000,embed_size=100, vocab_size=10000, time_steps=20, unit_length=512, dropout=0.5):
    #LSTM MODEL
    inputsLSTM = Input(shape=(time_steps,))
    x = Embedding(output_dim=embed_size, input_dim=vocab_size, weights=[embedding_matrix],input_length=time_steps, trainable=trainable)(inputsLSTM)
    lstm1, state_h1, state_c1 = LSTM(unit_length, return_sequences=True, return_state=True, input_shape=(time_steps,1))(x)
    lstm2, state_h2, state_c2 = LSTM(unit_length, return_sequences=False, return_state=True, input_shape=(time_steps,1))(lstm1)
    mergedLSTM = concatenate([state_h1,state_c1,state_h2,state_c2])
    outputsLSTM = Dense(1024, activation='tanh')(mergedLSTM)
    modelLSTM = Model(inputs=inputsLSTM, outputs=outputsLSTM)

    #VGG16 MODEL
    modelVGG16 = VGG16(include_top=True, weights='imagenet', classes=1000)
    inputsVGG16 = modelVGG16.input
    op = modelVGG16.layers[-2].output
    l2_norm = Lambda(lambda  x: K.l2_normalize(x,axis=1))(op)
    outputsVGG16 = Dense(1024, activation='tanh')(l2_norm)
    modelVGG16New = Model(inputs=inputsVGG16,outputs=outputsVGG16)
    for layer in modelVGG16New.layers[:-2]:
        layer.trainable = False
    mergedLayers = Multiply()([modelVGG16New.output, modelLSTM.output])
    dense1 = Dense(1000, activation='tanh')(mergedLayers)
    dense1 = Dropout(dropout)(dense1)
    dense2 = Dense(1000, activation='tanh')(dense1)
    dense2 = Dropout(dropout)(dense2)

    output =  Dense(num_classes, activation='softmax')(dense2)

    model = Model(inputs=[modelLSTM.input,modelVGG16New.input], outputs=output)
    return model

def vqaModelBiLSTM(embedding_matrix, trainable=False, num_classes=1000,embed_size=100, vocab_size=10000, time_steps=20, unit_length=512, dropout=0.5):
    #LSTM
    inputsLSTM = Input(shape=(time_steps,))
    x = Embedding(output_dim=embed_size, input_dim=vocab_size, weights=[embedding_matrix],input_length=time_steps, trainable=trainable)(inputsLSTM)
    # default merge_mode is concatenate others are mul,ave,sum
    y= Bidirectional(LSTM(unit_length, return_sequences=True), input_shape=(time_steps, 1), merge_mode='mul')(x)
    a= Bidirectional(LSTM(unit_length, return_sequences=False), input_shape=(time_steps, 1), merge_mode='sum')(y)
    modelLSTM = Model(inputs=inputsLSTM, outputs=a)

    #VGG16
    modelVGG16 = VGG16(include_top=True, weights='imagenet', classes=1000)
    ip = modelVGG16.input
    op = modelVGG16.layers[-2].output
    modelVGG16New = Model(ip,op)
    for layer in modelVGG16New.layers:
        layer.trainable = False
    
    merged = concatenate([modelLSTM.output, modelVGG16New.output])
    dense1 = Dense(1000, activation='tanh')(merged)
    dense1 = Dropout(dropout)(dense1)
    output =  Dense(num_classes, activation='softmax')(dense1)
    model = Model(inputs=[modelLSTM.input,modelVGG16New.input], outputs=output)
    return model

def vqaModelSimpleFeatures(embedding_matrix, image_feature_dims=4096, trainable=False, num_classes=1000,embed_size=100, vocab_size=10000, time_steps=20, unit_length=512, dropout=0.5):
    #LSTM MODEL
    inputsLSTM = Input(shape=(time_steps,))
    x = Embedding(output_dim=embed_size, input_dim=vocab_size, weights=[embedding_matrix],input_length=time_steps, trainable=trainable)(inputsLSTM)
    lstm1, state_h1, state_c1 = LSTM(unit_length, return_sequences=True, return_state=True, input_shape=(time_steps,1))(x)
    lstm2, state_h2, state_c2 = LSTM(unit_length, return_sequences=False, return_state=True, input_shape=(time_steps,1))(lstm1)
    mergedLSTM = concatenate([state_h1,state_c1,state_h2,state_c2])
    outputsLSTM = Dense(1024, activation='tanh')(mergedLSTM)
    modelLSTM = Model(inputs=inputsLSTM, outputs=outputsLSTM)

    #Image MODEL
    inputsImage = Input(shape=(image_feature_dims,))
    l2_norm = Lambda(lambda  x: K.l2_normalize(x,axis=1))(inputsImage)
    outputsImage = Dense(1024, activation='tanh')(l2_norm)
    modelVGG16New = Model(inputs=inputsImage,outputs=outputsImage)

    mergedLayers = Multiply()([modelVGG16New.output, modelLSTM.output])
    dense1 = Dense(1000, activation='tanh')(mergedLayers)
    dense1 = Dropout(dropout)(dense1)
    dense2 = Dense(1000, activation='tanh')(dense1)
    dense2 = Dropout(dropout)(dense2)

    output =  Dense(num_classes, activation='softmax')(dense2)

    model = Model(inputs=[modelLSTM.input,modelVGG16New.input], outputs=output)
    return model

def vqaModelBiLSTMFeatures(embedding_matrix, image_feature_dims=4096, trainable=False, num_classes=1000,embed_size=100, vocab_size=10000, time_steps=20, unit_length=512, dropout=0.5):
    #LSTM
    inputsLSTM = Input(shape=(time_steps,))
    x = Embedding(output_dim=embed_size, input_dim=vocab_size, weights=[embedding_matrix],input_length=time_steps, trainable=trainable)(inputsLSTM)
    # default merge_mode is concatenate others are mul,ave,sum
    y= Bidirectional(LSTM(unit_length, return_sequences=True), input_shape=(time_steps, 1), merge_mode='mul')(x)
    a= Bidirectional(LSTM(unit_length, return_sequences=False), input_shape=(time_steps, 1), merge_mode='sum')(y)
    modelLSTM = Model(inputs=inputsLSTM, outputs=a)

    #Image MODEL
    inputsImage = Input(shape=(image_feature_dims,))
    outputsImage = Lambda(lambda  x: K.l2_normalize(x,axis=1))(inputsImage)
    modelVGG16New = Model(inputsImage, outputsImage)

    merged = concatenate([modelLSTM.output, modelVGG16New.output])
    dense1 = Dense(1000, activation='tanh')(merged)
    dense1 = Dropout(dropout)(dense1)
    output =  Dense(num_classes, activation='softmax')(dense1)
    model = Model(inputs=[modelLSTM.input,modelVGG16New.input], outputs=output)
    return model


def vqaModel(embedding_matrix, trainable=False, num_classes=1000,embed_size=100, vocab_size=10000, time_steps=20, unit_length=256):
    encoded_question=vgg16Model(1000)
    encoded_image=lstmModel(embedding_matrix, trainable, embed_size, vocab_size, time_steps, unit_length)
    merged = concatenate([encoded_question.output, encoded_image.output])
    output = Dense(num_classes, activation='softmax')(merged)
    vqa_model = Model(inputs=[encoded_question.input, encoded_image.input], outputs=output)
    return vqa_model