import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


import tensorflow as tf
import math
from keras.models import Sequential
from keras.layers import Dense, Input
from keras.utils import np_utils
from keras.layers import Flatten, Dropout
from keras.utils.vis_utils import plot_model

#================================================================================================
# model creation
#================================================================================================

def emg_model():

    # Autoencoder Reconstruction 
    import tensorflow as tf
    import datetime

    # EMG Data

    sequence_length = 128
    num_features0 = 2

    input0 = Input(shape=(sequence_length, num_features0))
    encoded = Conv1D(filters=10, kernel_size=5, strides=1, padding='same')(input0)
    encoded = LeakyReLU(alpha=0.05)(encoded)
    encoded = MaxPooling1D(2)(encoded)
    encoded = Conv1D(filters=50, kernel_size=5, strides=1, padding='same')(encoded)
    encoded = LeakyReLU(alpha=0.05)(encoded)
    encoded = MaxPooling1D(2)(encoded)
    encoded = Conv1D(filters=50, kernel_size=5, strides=1, padding='same')(encoded)
    encoded = LeakyReLU(alpha=0.05)(encoded)
    encoded = Flatten()(encoded)
    encoded = Dense((5000), )(encoded)
    encoded = LeakyReLU(alpha=0.05)(encoded)
    encoded = Dense((1000))(encoded)
    encoded_emg = LeakyReLU(alpha=0.05)(encoded)

    decoded = Dense((5000))(encoded_emg)
    decoded = LeakyReLU(alpha=0.05)(decoded)
    decoded = Dense((1600))(decoded)
    decoded = LeakyReLU(alpha=0.05)(decoded)
    decoded = Reshape((32, 50))(decoded)
    decoded = Conv1DTranspose(filters=50, kernel_size=5, strides=1, padding='same')(decoded)
    decoded = LeakyReLU(alpha=0.05)(decoded)
    decoded = UpSampling1D(2)(decoded)
    decoded = Conv1DTranspose(filters=50, kernel_size=5 ,strides=1, padding='same')(decoded)
    decoded = LeakyReLU(alpha=0.05)(decoded)
    decoded = UpSampling1D(2)(decoded)
    decoded = Conv1DTranspose(filters=10, kernel_size=5, strides=1,padding='same')(decoded)
    decoded = LeakyReLU(alpha=0.05)(decoded)
    decoded = Conv1DTranspose(filters=2, kernel_size=5, strides=1,padding='same')(decoded)
    decoded = LeakyReLU(alpha=0.05)(decoded)

    cnn_autoencoder_emg_all = Model(input0, decoded)
    
    return cnn_autoencoder_emg_all


# import keras

# cnn_autoencoder_gsr_all.compile(optimizer=Adam(learning_rate=0.0001), loss = 'mse')
# early_stopper = EarlyStopping(patience=10, restore_best_weights=True)

# history5 = cnn_autoencoder_gsr_all.fit(x_train_gsr, x_train_gsr,
#                 epochs=100,
#                 batch_size=16,
#                 shuffle=True,
#                 validation_data=(x_test_gsr, x_test_gsr),
#                 callbacks=[early_stopper])
# encoder_gsr = Model(input3, encoded_gsr)

#================================================================================================
# model creation
#================================================================================================
def eog_model():
    import keras

    cnn_autoencoder_emg_all.compile(optimizer=Adam(learning_rate=0.001), loss = 'mse')
    early_stopper = EarlyStopping(patience=10, restore_best_weights=True)

    history1 = cnn_autoencoder_emg_all.fit(x_train_emg, x_train_emg,
                    epochs=100,
                    batch_size=16,
                    shuffle=True,
                    validation_data=(x_test_emg, x_test_emg),
                    callbacks=[early_stopper])
    encoder_emg = Model(input0, encoded_emg)

    #EOG Data

    sequence_length = 128
    num_features2 = 2

    input2 = Input(shape=(sequence_length, num_features2))
    encoded2 = Conv1D(filters=10, kernel_size=5, strides=1, padding='same')(input2)
    encoded2 = LeakyReLU(alpha=0.05)(encoded2)
    encoded2 = MaxPooling1D(2)(encoded2)
    encoded2 = Conv1D(filters=50, kernel_size=5, strides=1, padding='same')(encoded2)
    encoded2 = LeakyReLU(alpha=0.05)(encoded2)
    encoded2 = MaxPooling1D(2)(encoded2)
    encoded2 = Conv1D(filters=50, kernel_size=5, strides=1, padding='same')(encoded2)
    encoded2 = LeakyReLU(alpha=0.05)(encoded2)
    encoded2 = Flatten()(encoded2)
    encoded2 = Dense((5000))(encoded2)
    encoded2 = LeakyReLU(alpha=0.05)(encoded2)
    encoded2 = Dense((1000))(encoded2)
    encoded_eog = LeakyReLU(alpha=0.05)(encoded2)

    decoded2 = Dense((5000))(encoded_eog)
    decoded2 = LeakyReLU(alpha=0.05)(decoded2)
    decoded2 = Dense((1600))(decoded2)
    decoded2 = LeakyReLU(alpha=0.05)(decoded2)
    decoded2 = Reshape((32, 50))(decoded2)
    decoded2 = Conv1DTranspose(filters=50, kernel_size=5, strides=1, padding='same')(decoded2)
    decoded2 = LeakyReLU(alpha=0.05)(decoded2)
    decoded2 = UpSampling1D(2)(decoded2)
    decoded2 = Conv1DTranspose(filters=50, kernel_size=5 ,strides=1, padding='same')(decoded2)
    decoded2 = LeakyReLU(alpha=0.05)(decoded2)
    decoded2 = UpSampling1D(2)(decoded2)
    decoded2 = Conv1DTranspose(filters=10, kernel_size=5, strides=1,padding='same')(decoded2)
    decoded2 = LeakyReLU(alpha=0.05)(decoded2)
    decoded2 = Conv1DTranspose(filters=2, kernel_size=5, strides=1,padding='same')(decoded2)
    decoded2 = LeakyReLU(alpha=0.05)(decoded2)

    cnn_autoencoder_eog_all = Model(input2, decoded2)
#     import keras

#     cnn_autoencoder_eog_all.compile(optimizer=Adam(learning_rate=0.0001), loss = 'mse')
#     early_stopper = EarlyStopping(patience=10, restore_best_weights=True)

#     history3 = cnn_autoencoder_eog_all.fit(x_train_eog, x_train_eog,
#                     epochs=100,
#                     batch_size=16,
#                     shuffle=True,
#                     validation_data=(x_test_eog, x_test_eog),
#                     callbacks=[early_stopper])
#     encoder_eog = Model(input2, encoded_eog)
    
    return cnn_autoencoder_eog_all
#================================================================================================
# model creation
#================================================================================================
def gsr_model():

    #GSR Data

    sequence_length = 128
    num_features2 = 1

    input3 = Input(shape=(sequence_length, num_features2))
    encoded3 = Conv1D(filters=10, kernel_size=5, strides=1, padding='same')(input3)
    encoded3 = LeakyReLU(alpha=0.05)(encoded3)
    encoded3 = MaxPooling1D(2)(encoded3)
    encoded3 = Conv1D(filters=50, kernel_size=5, strides=1, padding='same')(encoded3)
    encoded3 = LeakyReLU(alpha=0.05)(encoded3)
    encoded3 = MaxPooling1D(2)(encoded3)
    encoded3 = Conv1D(filters=50, kernel_size=5, strides=1, padding='same')(encoded3)
    encoded3 = LeakyReLU(alpha=0.05)(encoded3)
    encoded3 = Flatten()(encoded3)
    encoded3 = Dense((1600))(encoded3)
    encoded3 = LeakyReLU(alpha=0.05)(encoded3)

    encoded3 = Dense((1000))(encoded3)
    encoded_gsr = LeakyReLU(alpha=0.05)(encoded3)

    decoded3 = Dense((5000))(encoded_gsr)
    decoded3 = LeakyReLU(alpha=0.05)(decoded3)
    decoded3 = Dense((1600))(encoded_gsr)
    decoded3 = LeakyReLU(alpha=0.05)(decoded3)
    decoded3 = Reshape((32, 50))(decoded3)
    decoded3 = Conv1DTranspose(filters=50, kernel_size=5, strides=1, padding='same')(decoded3)
    decoded3 = LeakyReLU(alpha=0.05)(decoded3)
    decoded3 = UpSampling1D(2)(decoded3)
    decoded3 = Conv1DTranspose(filters=50, kernel_size=5 ,strides=1, padding='same')(decoded3)
    decoded3 = LeakyReLU(alpha=0.05)(decoded3)
    decoded3 = UpSampling1D(2)(decoded3)
    decoded3 = Conv1DTranspose(filters=10, kernel_size=5, strides=1,padding='same')(decoded3)
    decoded3 = LeakyReLU(alpha=0.05)(decoded3)
    decoded3 = Conv1DTranspose(filters=1, kernel_size=5, strides=1,padding='same')(decoded3)
    decoded3 = LeakyReLU(alpha=0.05)(decoded3)

    cnn_autoencoder_gsr_all = Model(input3, decoded3)
    
    return cnn_autoencoder_gsr_all
#================================================================================================
# model creation
#================================================================================================
def valence_classification():
    sequence_length = 128
    num_features0 = 2

    input0 = Input(shape=(sequence_length, num_features0))
    encoded = Conv1D(filters=10, kernel_size=5, strides=1, padding='same')(input0)
    encoded = LeakyReLU(alpha=0.05)(encoded)
    encoded = MaxPooling1D(2)(encoded)
    encoded = Conv1D(filters=50, kernel_size=5, strides=1, padding='same')(encoded)
    encoded = LeakyReLU(alpha=0.05)(encoded)
    encoded = MaxPooling1D(2)(encoded)
    encoded = Conv1D(filters=50, kernel_size=5, strides=1, padding='same')(encoded)
    encoded = LeakyReLU(alpha=0.05)(encoded)
    encoded = Flatten()(encoded)
    encoded = Dense((5000), )(encoded)
    encoded = LeakyReLU(alpha=0.05)(encoded)
    encoded = Dense((1000))(encoded)
    encoded_emg = LeakyReLU(alpha=0.05)(encoded)

    decoded = Dense((5000))(encoded_emg)
    decoded = LeakyReLU(alpha=0.05)(decoded)
    decoded = Dense((1600))(decoded)
    decoded = LeakyReLU(alpha=0.05)(decoded)
    decoded = Reshape((32, 50))(decoded)
    decoded = Conv1DTranspose(filters=50, kernel_size=5, strides=1, padding='same')(decoded)
    decoded = LeakyReLU(alpha=0.05)(decoded)
    decoded = UpSampling1D(2)(decoded)
    decoded = Conv1DTranspose(filters=50, kernel_size=5 ,strides=1, padding='same')(decoded)
    decoded = LeakyReLU(alpha=0.05)(decoded)
    decoded = UpSampling1D(2)(decoded)
    decoded = Conv1DTranspose(filters=10, kernel_size=5, strides=1,padding='same')(decoded)
    decoded = LeakyReLU(alpha=0.05)(decoded)
    decoded = Conv1DTranspose(filters=2, kernel_size=5, strides=1,padding='same')(decoded)
    decoded = LeakyReLU(alpha=0.05)(decoded)

  # cnn_autoencoder_emg_s1 = Model(input0, decoded)

  # cnn_autoencoder_emg_s1.set_weights(global_weight_emg)

    encoder_emg = Model(input0, encoded_emg)
    encoder_emg.set_weights(global_weight_emg[:10])
  # encoder_emg.set_weights(global_weight_emg)

    sequence_length = 128
    num_features2 = 2

    input2 = Input(shape=(sequence_length, num_features2))
    encoded2 = Conv1D(filters=10, kernel_size=5, strides=1, padding='same')(input2)
    encoded2 = LeakyReLU(alpha=0.05)(encoded2)
    encoded2 = MaxPooling1D(2)(encoded2)
    encoded2 = Conv1D(filters=50, kernel_size=5, strides=1, padding='same')(encoded2)
    encoded2 = LeakyReLU(alpha=0.05)(encoded2)
    encoded2 = MaxPooling1D(2)(encoded2)
    encoded2 = Conv1D(filters=50, kernel_size=5, strides=1, padding='same')(encoded2)
    encoded2 = LeakyReLU(alpha=0.05)(encoded2)
    encoded2 = Flatten()(encoded2)
    encoded2 = Dense((5000))(encoded2)
    encoded2 = LeakyReLU(alpha=0.05)(encoded2)
    encoded2 = Dense((1000))(encoded2)
    encoded_eog = LeakyReLU(alpha=0.05)(encoded2)

    decoded2 = Dense((5000))(encoded_eog)
    decoded2 = LeakyReLU(alpha=0.05)(decoded2)
    decoded2 = Dense((1600))(decoded2)
    decoded2 = LeakyReLU(alpha=0.05)(decoded2)
    decoded2 = Reshape((32, 50))(decoded2)
    decoded2 = Conv1DTranspose(filters=50, kernel_size=5, strides=1, padding='same')(decoded2)
    decoded2 = LeakyReLU(alpha=0.05)(decoded2)
    decoded2 = UpSampling1D(2)(decoded2)
    decoded2 = Conv1DTranspose(filters=50, kernel_size=5 ,strides=1, padding='same')(decoded2)
    decoded2 = LeakyReLU(alpha=0.05)(decoded2)
    decoded2 = UpSampling1D(2)(decoded2)
    decoded2 = Conv1DTranspose(filters=10, kernel_size=5, strides=1,padding='same')(decoded2)
    decoded2 = LeakyReLU(alpha=0.05)(decoded2)
    decoded2 = Conv1DTranspose(filters=2, kernel_size=5, strides=1,padding='same')(decoded2)
    decoded2 = LeakyReLU(alpha=0.05)(decoded2)

  # cnn_autoencoder_eog_s2 = Model(input2, decoded2)
  # cnn_autoencoder_eog_s2.set_weights(global_weight_eog)

    encoder_eog = Model(input2, encoded_eog)
    encoder_eog.set_weights(global_weight_eog[:10])


    sequence_length = 128
    num_features2 = 1

    input3 = Input(shape=(sequence_length, num_features2))
    encoded3 = Conv1D(filters=10, kernel_size=5, strides=1, padding='same')(input3)
    encoded3 = LeakyReLU(alpha=0.05)(encoded3)
    encoded3 = MaxPooling1D(2)(encoded3)
    encoded3 = LeakyReLU(alpha=0.05)(encoded3)
    encoded3 = MaxPooling1D(2)(encoded3)
    encoded3 = Conv1D(filters=50, kernel_size=5, strides=1, padding='same')(encoded3)
    encoded3 = LeakyReLU(alpha=0.05)(encoded3)
    encoded3 = Flatten()(encoded3)
    encoded3 = Dense((1600))(encoded3)
    encoded3 = LeakyReLU(alpha=0.05)(encoded3)

    encoded3 = Dense((1000))(encoded3)
    encoded_gsr = LeakyReLU(alpha=0.05)(encoded3)

  # decoded3 = Dense((5000))(encoded_gsr)
  # decoded3 = LeakyReLU(alpha=0.05)(decoded3)      #NEED TO BE CHANGED / CHECKED
    decoded3 = Dense((1600))(encoded_gsr)
  # decoded3 = Dense((1600))(decoded3)
    decoded3 = LeakyReLU(alpha=0.05)(decoded3)
    decoded3 = Reshape((32, 50))(decoded3)
    decoded3 = Conv1DTranspose(filters=50, kernel_size=5, strides=1, padding='same')(decoded3)
    decoded3 = LeakyReLU(alpha=0.05)(decoded3)
    decoded3 = UpSampling1D(2)(decoded3)
    decoded3 = Conv1DTranspose(filters=50, kernel_size=5 ,strides=1, padding='same')(decoded3)
    decoded3 = LeakyReLU(alpha=0.05)(decoded3)
    decoded3 = UpSampling1D(2)(decoded3)
    decoded3 = Conv1DTranspose(filters=10, kernel_size=5, strides=1,padding='same')(decoded3)
    decoded3 = LeakyReLU(alpha=0.05)(decoded3)
    decoded3 = Conv1DTranspose(filters=1, kernel_size=5, strides=1,padding='same')(decoded3)
    decoded3 = LeakyReLU(alpha=0.05)(decoded3)

  # cnn_autoencoder_gsr_s1 = Model(input3, decoded3)

  # cnn_autoencoder_gsr_s1.set_weights(global_weight_gsr)

    encoder_gsr = Model(input3, encoded_gsr)
    encoder_gsr.set_weights(global_weight_gsr[:10])

    encoder_emg.trainable=False
    encoder_eog.trainable=False
    encoder_gsr.trainable=False

  # Concatenating the Models
    concatenated = concatenate([encoder_eog.output, encoder_emg.output, encoder_gsr.output], axis=-1)
    out = Dense(1024, activation='relu')(concatenated)
  # out = Dropout(0.5)(out)
    out = Dense(512, activation='relu')(out)
    out = Dropout(0.25)(out)
  # out = Dense(256, activation='relu')(out)
  # out = Dropout(0.25)(out)
    out = Dense(2, activation='softmax')(out)
    MDCARE_all_v = Model(inputs=[encoder_eog.input, encoder_emg.input, encoder_gsr.input], outputs=out)

    MDCARE_all_v.compile(optimizer=Adam(learning_rate=0.0001, decay=1e-6), 
                loss='categorical_crossentropy', 
                metrics=['accuracy'])
  
    return MDCARE_all_v
