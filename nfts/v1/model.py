import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Conv2D, BatchNormalization, Reshape, Dropout, LeakyReLU, Flatten, Add, UpSampling2D, Activation, PReLU
from tensorflow.keras.optimizers import RMSprop


def clamp_weights(w):
    return tf.keras.backend.clip(w, -0.01, 0.01)


def wasserstein_loss(y_true, y_pred):
  return tf.reduce_mean(y_true * y_pred)


def residual_block(x, filters, kernel_size=(4, 4), strides=1, bn=False):
    x = Conv2D(filters, kernel_size, strides=strides, padding='same')(x)
    x = PReLU()(x)

    inner = Conv2D(filters, kernel_size, strides=1, padding='same')(x)
    inner = PReLU()(inner)

    x = Add()([x, inner])

    if bn:
        x = BatchNormalization()(x)

    return x


def create_generator(seed_dim=128, momentum=0.8, load_path=False):
    inputs = Input(shape=(seed_dim))
    inner = inputs

    inner = Dense(seed_dim * 8 * 8)(inner)
    inner = PReLU()(inner)
    inner = Reshape((8, 8, seed_dim))(inner)

    #inner = UpSampling2D()(inner)

    #inner = Conv2D(128, (4, 4), strides=1, padding='same')(inner)
    #inner = PReLU()(inner)
    #inner = BatchNormalization(momentum=momentum)(inner)

    inner = UpSampling2D()(inner)

    inner = Conv2D(128, (4, 4), strides=1, padding='same')(inner)
    inner = BatchNormalization(momentum=momentum)(inner)
    inner = PReLU()(inner)

    inner = UpSampling2D()(inner)

    inner = Conv2D(64, (4, 4), strides=1, padding='same')(inner)
    inner = BatchNormalization(momentum=momentum)(inner)
    inner = PReLU()(inner)

    inner = UpSampling2D()(inner)

    inner = Conv2D(32, (4, 4), strides=1, activation='relu', padding='same')(inner)
    inner = BatchNormalization(momentum=momentum)(inner)
    inner = PReLU()(inner)

    inner = Conv2D(3, (4, 4), strides=1, padding='same')(inner)

    outputs = Activation('sigmoid')(inner)

    generator = Model(inputs=inputs, outputs=outputs)

    if load_path:
        try:
            generator.load_weights(load_path)
        
        except:
            print('failed to load generator')

    return generator


def create_discriminator(input_shape=(64, 64, 3), momentum=0.8, load_path=False):
    inputs = Input(shape=input_shape)
    inner = inputs

    inner = Conv2D(16, (3, 3), strides=2, padding='same', kernel_constraint=clamp_weights)(inner)
    inner = BatchNormalization(momentum=momentum)(inner)
    inner = PReLU()(inner)

    inner = Dropout(0.25)(inner)

    inner = Conv2D(32, (3, 3), strides=2, padding='same', kernel_constraint=clamp_weights)(inner)
    inner = BatchNormalization(momentum=momentum)(inner)
    inner = PReLU()(inner)

    inner = Dropout(0.25)(inner)

    inner = Conv2D(64, (3, 3), strides=2, padding='same', kernel_constraint=clamp_weights)(inner)
    inner = BatchNormalization(momentum=momentum)(inner)
    inner = PReLU()(inner)

    inner = Dropout(0.25)(inner)

    inner = Conv2D(128, (3, 3), strides=1, padding='same', kernel_constraint=clamp_weights)(inner)
    inner = BatchNormalization(momentum=momentum)(inner)
    inner = PReLU()(inner)

    inner = Dropout(0.25)(inner)

    inner = Flatten()(inner)

    outputs = Dense(1, kernel_constraint=clamp_weights)(inner)

    discriminator = Model(inputs=inputs, outputs=outputs)

    if load_path:
        try:
            discriminator.load_weights(load_path)

        except: 
            print('failed to load discriminator')

    return discriminator


def create_combined(generator, discriminator, seed_dim=128,):
    inputs = Input(shape=(seed_dim,))
    inner = inputs

    inner = generator(inner)

    discriminator.trainable = False
    inner = discriminator(inner)

    outputs = inner

    return Model(inputs=inputs, outputs=outputs)

