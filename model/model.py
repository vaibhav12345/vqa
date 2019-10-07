from keras.layers import concatenate, Embedding, Bidirectional, LSTM, Input, Add,Dense, Conv2D, MaxPooling2D, AveragePooling2D, ZeroPadding2D, Dropout, Flatten, merge, Reshape, Activation
from keras.models import load_model,Model
from keras.initializers import glorot_uniform
from keras.applications.vgg16 import VGG16


def vgg16Model(numClasses=1000):
    model = VGG16(include_top=True, weights='imagenet', classes=numClasses)
    ip = model.input
    op = model.layers[-2].output
    vgg16Model = Model(ip,op)
    for layer in vgg16Model.layers:
        layer.trainable = False
    return vgg16Model

def lstmModel(embedding_matrix, trainable, embed_size, vocab_size, time_steps, unit_length):
    inputs = Input(shape=(time_steps,))
    x = Embedding(output_dim=embed_size, input_dim=vocab_size, weights=[embedding_matrix],input_length=time_steps, trainable=trainable)(inputs)
    #default merge_mode is concatenate others are mul,ave,sum
    y= Bidirectional(LSTM(unit_length, return_sequences=True), input_shape=(time_steps, 1), merge_mode='mul')(x)
    #print("y.shape",y.shape)
    a= Bidirectional(LSTM(unit_length, return_sequences=False), input_shape=(time_steps, 1), merge_mode='sum')(y)
    model = Model(inputs=inputs, outputs=a)
    return model

def vqaModel(embedding_matrix, trainable=False, num_classes=1000,embed_size=100, vocab_size=10000, time_steps=20, unit_length=256):
    encoded_question=vgg16Model(num_classes)
    encoded_image=lstmModel(embedding_matrix, trainable, embed_size, vocab_size, time_steps, unit_length)

    merged = keras.layers.concatenate([encoded_question.output, encoded_image.output])
    output = Dense(num_classes, activation='softmax')(merged)
    vqa_model = Model(inputs=[encoded_question.input, encoded_image.input], outputs=output)
    return vqa_model