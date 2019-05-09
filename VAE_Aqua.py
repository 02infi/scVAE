# "ssh -L 16006:127.0.0.1:6006 gjumde@141.80.186.196"
# "source activate Frontal_lobe"
# "tensorboard --logdir=/local/gaurav/data/t_flow_files"
# open "localhost:16006/" in  browser



from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import keras
from keras.layers import Lambda, Input, Dense
from keras.models import Model
from keras.datasets import mnist
from keras.losses import mse, binary_crossentropy
from keras.utils import plot_model
from keras import backend as K
from keras.layers import Input, Dense, BatchNormalization, LeakyReLU, Dropout, Lambda
from keras.callbacks import TensorBoard
import numpy as np
import matplotlib.pyplot as plt
import argparse
import os

import csv
import scipy.io

# reparameterization trick
# instead of sampling from Q(z|X), sample epsilon = N(0,I)
# z = z_mean + sqrt(var) * epsilon
def sampling(args):
    """Reparameterization trick by sampling from an isotropic unit Gaussian.
    # Arguments
        args (tensor): mean and log of variance of Q(z|X)
    # Returns
        z (tensor): sampled latent vector
    """
    z_mean, z_log_var = args
    batch = K.shape(z_mean)[0]
    dim = K.int_shape(z_mean)[1]
    # by default, random_normal has mean = 0 and std = 1.0
    epsilon = K.random_normal(shape=(batch, dim))
    return z_mean + K.exp(0.5 * z_log_var) * epsilon



#
matrix_dir= "/local/gaurav/data/matrix.mtx"
L2 = scipy.io.mmread(matrix_dir)
Genes = scipy.genfromtxt('/local/gaurav/data/genes.tsv',delimiter="\t",dtype= str )
Cells = scipy.genfromtxt('/local/gaurav/data/barcodes.tsv',delimiter="\t",dtype= str)
#

training_data = L2.toarray().T


## PARAMETER
original_dim = training_data.shape[1]
latent_dim = 100

# network parameters
input_shape = (original_dim, )
intermediate_dim = 512
batch_size = 1000
epochs = 100
dropout_rate = 0.05








## ENCODER NETWORK 

# Build Encoder Sub-Network 
inputs = Input(shape=input_shape, name='encoder_input')
Hidden_1 = Dense(800, kernel_initializer= keras.initializers.glorot_normal() , use_bias=False)(inputs)

Hidden_1 = BatchNormalization(axis=1)(Hidden_1)
Hidden_1 = LeakyReLU()(Hidden_1)
Hidden_1 = Dropout(dropout_rate)(Hidden_1)

Hidden_1 = Dense(intermediate_dim, kernel_initializer= keras.initializers.glorot_normal(), activation='relu')(Hidden_1)

Hidden_1 = BatchNormalization(axis=1)(Hidden_1)
Hidden_1 = LeakyReLU()(Hidden_1)
Hidden_1 = Dropout(dropout_rate)(Hidden_1)

z_mean = Dense(latent_dim, name='z_mean')(Hidden_1)
z_log_var = Dense(latent_dim, name='z_log_var')(Hidden_1)

# use reparameterization trick to push the sampling out as input
# note that "output_shape" isn't necessary with the TensorFlow backend
z = Lambda(sampling, output_shape=(latent_dim,), name='z')([z_mean, z_log_var])


latent_inputs = Input(shape=(latent_dim,), name='z_sampling')


## Encoder Model Instance 
encoder = Model(inputs, [z_mean, z_log_var, z], name='encoder')
encoder.summary()
# plot_model(encoder, to_file='vae_mlp_encoder.png', show_shapes=True)

## # Build Dencoder Sub-Network 


Hidden_2 = BatchNormalization(axis=1)(latent_inputs)
Hidden_2 = LeakyReLU()(Hidden_2)
Hidden_2 = Dropout(dropout_rate)(Hidden_2)

Hidden_2 = Dense(intermediate_dim, activation='relu')(Hidden_2)

Hidden_2 = BatchNormalization(axis=1)(Hidden_2)
Hidden_2 = LeakyReLU()(Hidden_2)
Hidden_2 = Dropout(dropout_rate)(Hidden_2)

Hidden_2 = Dense(original_dim, kernel_initializer= keras.initializers.glorot_normal(), activation='relu')(Hidden_2)

decoder = Model(latent_inputs, Hidden_2, name="decoder")





# instantiate VAE model
outputs = decoder(encoder(inputs)[2])
vae = Model(inputs, outputs, name='vae_mlp')



reconstruction_loss = mse(inputs, outputs)
reconstruction_loss = binary_crossentropy(inputs,outputs)

reconstruction_loss *= original_dim
kl_loss = 1 + z_log_var - K.square(z_mean) - K.exp(z_log_var)
kl_loss = K.sum(kl_loss, axis=-1)
kl_loss *= -0.5
vae_loss = K.mean(reconstruction_loss + kl_loss)
vae.add_loss(vae_loss)
vae.compile(optimizer='adam')
vae.summary()
#plot_model(vae,to_file='vae_mlp.png',show_shapes=True)

vae.fit(training_data,epochs=epochs,batch_size=batch_size,callbacks=[TensorBoard(log_dir='/local/gaurav/data/t_flow_files')])
vae.save_weights('/local/gaurav/data/vae_mlp_mnist.h5')







    if args.weights:
        vae.load_weights(args.weights)
    else:
        # train the autoencoder
        vae.fit(training_data,
                epochs=epochs,
                batch_size=batch_size,
                validation_data=(x_test, None))
        vae.save_weights('vae_mlp_mnist.h5')

plot_results(models,data,batch_size=batch_size,model_name="vae_mlp")







if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    help_ = "Load h5 model trained weights"
    parser.add_argument("-w", "--weights", help=help_)
    help_ = "Use mse loss instead of binary cross entropy (default)"
    parser.add_argument("-m",
                        "--mse",
                        help=help_, action='store_true')
    args = parser.parse_args()
    models = (encoder, decoder)
    data = (x_test, y_test)

    # VAE loss = mse_loss or xent_loss + kl_loss
    if args.mse:
        reconstruction_loss = mse(inputs, outputs)
    else:
        reconstruction_loss = binary_crossentropy(inputs,
                                                  outputs)

    reconstruction_loss *= original_dim
    kl_loss = 1 + z_log_var - K.square(z_mean) - K.exp(z_log_var)
    kl_loss = K.sum(kl_loss, axis=-1)
    kl_loss *= -0.5
    vae_loss = K.mean(reconstruction_loss + kl_loss)
    vae.add_loss(vae_loss)
    vae.compile(optimizer='adam')
    vae.summary()
    plot_model(vae,
               to_file='vae_mlp.png',
               show_shapes=True)

    if args.weights:
        vae.load_weights(args.weights)
    else:
        # train the autoencoder
        vae.fit(training_data,
                epochs=epochs,
                batch_size=batch_size,
                validation_data=(x_test, None))
        vae.save_weights('vae_mlp_mnist.h5')

    plot_results(models,
                 data,
                 batch_size=batch_size,
model_name="vae_mlp")

