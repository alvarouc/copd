from keras.layers import Input, Dense
from keras.models import Model
from keras import regularizers
# from keras.callbacks import TensorBoard


def build_autoencoder(input_dim, layers_dim=[100, 10, 10],
                      activations=['relu', 'sigmoid'],
                      inits=['glorot_uniform', 'glorot_normal'],
                      optimizer='adadelta',
                      l2=1e-5,
                      loss='mse'):

    input_row = Input(shape=(input_dim,))

    for n, layer_dim in enumerate(layers_dim):
        if n == 0:
            encoded = Dense(layer_dim, activation=activations[0],
                            kernel_initializer=inits[0])(input_row)
        elif n < (len(layer_dim) - 1):
            encoded = Dense(layer_dim, activation=activations[0],
                            kernel_initializer=inits[0])(encoded)
        else:
            encoded = Dense(layer_dim, activation=activations[0],
                            activity_regularizer=regularizers.l2(l2),
                            kernel_initializer=inits[0])(encoded)

    encoder = Model(input_row, encoded)

    for n, layer_dim in enumerate(reversed(layers_dim[:-1])):
        if n == 0:
            decoded = Dense(layer_dim, activation=activations[0],
                            kernel_initializer=inits[0])(encoded)
        else:
            decoded = Dense(layer_dim, activation=activations[0],
                            kernel_regularizer=regularizers.l2(l2),
                            kernel_initializer=inits[0])(decoded)

    decoded = Dense(input_dim, activation=activations[1],
                    kernel_regularizer=regularizers.l2(l2),
                    kernel_initializer=inits[1])(decoded)

    autoencoder = Model(input_row, decoded)
    autoencoder.compile(optimizer=optimizer, loss=loss)
    return autoencoder, encoder
