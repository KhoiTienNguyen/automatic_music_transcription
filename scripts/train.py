import os, sys
import fnmatch
import cv2
import numpy as np
import string
import time
import json
import math

from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import Dense, LSTM, Reshape, BatchNormalization, Input, Conv2D, MaxPool2D, Lambda, Bidirectional
from tensorflow.keras.models import Model
from tensorflow.keras.activations import relu, sigmoid, softmax
import tensorflow.keras.backend as K
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras import layers

instrument = 'instruments/marimba'
max_label_len = 51

def batch_gen():
    path_x = f'{instrument}/x'
    path_y = f'{instrument}/y'
    for idx, dirname in enumerate(os.listdir(path_x)):
        training_img = []
        training_txt = []
        train_input_length = []
        train_label_length = []
        for filename in os.listdir(f'{path_x}/{dirname}'):
            # Images
            img = cv2.imread(f'{path_x}/{dirname}/{filename}', cv2.IMREAD_GRAYSCALE)
            # img = img.reshape(img.shape[1], img.shape[0])
            # (W, H) --> (W, H, 1)
            img = np.expand_dims(img, axis=2)
            # Normalize image
            img = img / 255.
            training_img.append(img)

            # Text Targets
            text = np.load(f'{path_y}/{dirname}/{filename}'.split('.')[0] + '.npy')
            # if len(text) > max_label_len:
            #     max_label_len = len(text)
            training_txt.append(text)
            train_label_length.append(len(text))
            train_input_length.append(img.shape[1]-1)
        train_padded_txt = pad_sequences(training_txt, padding='post', maxlen=50)

        yield np.array(training_img), np.array(train_padded_txt), np.array(train_input_length), np.array(train_label_length)

def val_gen():
    path_x = f'{instrument}/validate_x'
    path_y = f'{instrument}/validate_y'
    for idx, dirname in enumerate(os.listdir(path_x)):
        validate_img = []
        validate_txt = []
        validate_input_length = []
        validate_label_length = []
        # max_label_len = 0
        for filename in os.listdir(f'{path_x}/{dirname}'):
            # Images
            img = cv2.imread(f'{path_x}/{dirname}/{filename}', cv2.IMREAD_GRAYSCALE)
            # img = img.reshape(img.shape[1], img.shape[0])
            # (W, H) --> (W, H, 1)
            img = np.expand_dims(img, axis=2)
            # Normalize image
            img = img / 255.
            validate_img.append(img)

            # Text Targets
            text = np.load(f'{path_y}/{dirname}/{filename}'.split('.')[0] + '.npy')
            # if len(text) > max_label_len:
            #     max_label_len = len(text)
            validate_txt.append(text)
            validate_label_length.append(len(text))
            validate_input_length.append(img.shape[1]-1)
        validate_padded_txt = pad_sequences(validate_txt, padding='post', maxlen=max_label_len)

        yield np.array(validate_img), np.array(validate_padded_txt), np.array(validate_input_length), np.array(validate_label_length)

def main():
    # Load word to index mapping
    w2i = json.load(open('w2i_all.json', 'r'))

    ###### MODEL

    inputs = Input(shape=(128,None,1))
    # convolution layer with kernel size (3,3)
    conv_1 = Conv2D(64, (3,3), activation = 'relu', padding='same')(inputs)
    # poolig layer with kernel size (2,2)
    pool_1 = MaxPool2D(pool_size=(2, 1))(conv_1)
    conv_2 = Conv2D(128, (3,3), activation = 'relu', padding='same')(pool_1)
    pool_2 = MaxPool2D(pool_size=(2, 1))(conv_2)
    conv_3 = Conv2D(256, (3,3), activation = 'relu', padding='same')(pool_2)
    conv_4 = Conv2D(256, (3,3), activation = 'relu', padding='same')(conv_3)
    # poolig layer with kernel size (2,1)
    pool_4 = MaxPool2D(pool_size=(2, 1))(conv_4)
    conv_5 = Conv2D(512, (3,3), activation = 'relu', padding='same')(pool_4)
    pool_5 = MaxPool2D(pool_size=(2, 1))(conv_5)
    conv_6 = Conv2D(512, (3,3), activation = 'relu', padding='same')(pool_5)
    pool_6 = MaxPool2D(pool_size=(2, 1))(conv_6)
    conv_7 = Conv2D(512, (3,3), activation = 'relu', padding='same')(pool_6)
    # Batch normalization layer
    batch_norm_5 = BatchNormalization()(conv_7)
    conv_8 = Conv2D(512, (3,3), activation = 'relu', padding='same')(batch_norm_5)
    batch_norm_6 = BatchNormalization()(conv_8)
    pool_7 = MaxPool2D(pool_size=(2, 1))(batch_norm_6)
    conv_9 = Conv2D(512, (2,2), activation = 'relu')(pool_7)
    squeezed = Lambda(lambda x: K.squeeze(x, 1))(conv_9)
    # bidirectional LSTM layers with units=128
    blstm_1 = Bidirectional(LSTM(128, return_sequences=True, dropout = 0.2))(squeezed)
    blstm_2 = Bidirectional(LSTM(128, return_sequences=True, dropout = 0.2))(blstm_1)
    outputs = Dense(len(w2i)+1, activation = 'softmax')(blstm_2)
    # model to be used at test time
    act_model = Model(inputs, outputs)

    print(act_model.summary())

    ###### CTC LOSS

    labels = Input(name='the_labels', shape=[max_label_len], dtype='float32')
    input_length = Input(name='input_length', shape=[1], dtype='int64')
    label_length = Input(name='label_length', shape=[1], dtype='int64')
    

    def ctc_lambda_func(args):
        y_pred, labels, input_length, label_length = args
    
        return K.ctc_batch_cost(labels, y_pred, input_length, label_length, )
    
    
    loss_out = Lambda(ctc_lambda_func, output_shape=(1,), name='ctc')([outputs, labels, input_length, label_length])

    ###### INITIALIZE MODEL

    #model to be used at training time
    model = Model(inputs=[inputs, labels, input_length, label_length], outputs=loss_out)

    model.compile(loss={'ctc': lambda y_true, y_pred: y_pred}, optimizer = 'adam')
 
    filepath="best_model.hdf5"
    checkpoint = ModelCheckpoint(filepath=filepath, monitor='val_loss', verbose=1, save_best_only=True, mode='auto')
    callbacks_list = [checkpoint]

    ###### TRAINING MODEL

    batch_size = 16
    epochs = 10
    current_loss = 99999999
    for i in range(epochs):
        gen = batch_gen()
        # next(gen, None)
        index = 1
        while True:
            print(f'Epoch: {i+1} | Batch: {index}')
            # print(model.weights)
            index +=1
            next_batch = next(gen, None)
            if not next_batch:
                break
            training_img, train_padded_txt, train_input_length, train_label_length = next_batch
            # model.fit(x=[training_img, train_padded_txt, train_input_length, train_label_length], y=np.zeros(len(training_img)), batch_size=batch_size, epochs = 1, validation_data = ([valid_img, valid_padded_txt, valid_input_length, valid_label_length], [np.zeros(len(valid_img))]), verbose = 1, callbacks = callbacks_list)
            model.fit(x=[training_img, train_padded_txt, train_input_length, train_label_length], y=np.zeros(len(training_img)), batch_size=batch_size, epochs = 1,  verbose = 2, callbacks = callbacks_list)
            val = val_gen()
            for _ in range(4):
                loss = []
                valid_img, valid_padded_txt, valid_input_length, valid_label_length = next(val)
                # print(K.ctc_batch_cost(valid_img, valid_padded_txt, valid_input_length, valid_label_length))
                # print(type(K.ctc_batch_cost(valid_img, valid_padded_txt, valid_input_length, valid_label_length)))
                loss.append(model.predict([valid_img, valid_padded_txt, valid_input_length, valid_label_length]))
            mean_loss = np.mean(loss)
            print(f'Lowest loss: {current_loss} | Validation loss: {mean_loss}')
            if mean_loss < current_loss:
                print(f'Validation loss {mean_loss} lower than Lowest loss {current_loss} ===> Saving model and weights ')
                current_loss = mean_loss
                model.save_weights('my_model_weights.h5')
                model.save('best_model.hdf5')

if __name__ == '__main__':
    main()