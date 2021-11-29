import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Conv2D, BatchNormalization, Reshape, Dropout, LeakyReLU, Flatten, Add, UpSampling2D, Activation, Conv2DTranspose, PReLU
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


def generator_layer(x, filters, kernel_size=(4, 4), momentum=0.8):
    x = UpSampling2D()(x)
    x = Conv2D(filters, kernel_size, strides=1, padding='same')(x)
    x = PReLU()(x)
    x = BatchNormalization(momentum=momentum)(x)

    return x


def discriminator_layer(x, filters, kernel_size=(4, 4), momentum=0.8):
    x = Conv2D(filters, kernel_size, strides=2, padding='same', kernel_constraint=clamp_weights)(x)
    x = PReLU()(x)
    x = BatchNormalization(momentum=momentum)(x)
    x = Dropout(0.25)(x)

    return x


def generator_base(latent_dim):
    inputs = Input(shape=(latent_dim, ))
    inner = inputs

    inner = Dense(latent_dim * 4 * 4)(inner)
    inner = PReLU()(inner)
    inner = BatchNormalization()(inner)

    inner = Reshape((4, 4, latent_dim))(inner)

    base = Model(inputs=inputs, outputs=inner)
    base.compile(loss='mse', optimizer='adam')

    return base


def generator_head(generator):
    inputs = Input(shape=generator.input_shape[1:])
    inner = inputs

    inner = generator(inner)

    inner = Conv2D(3, (4,4), strides=1, padding='same')(inner)
    inner = Activation('sigmoid')(inner)

    head = Model(inputs=inputs, outputs=inner)
    head.compile(loss='mse', optimizer='adam')

    return head


def discriminator_head(learning_rate=5e-5):
    inputs = Input(shape=(4, 4, 3))
    inner = inputs

    inner = Flatten()(inner)

    inner = Dense(1)(inner)

    head = Model(inputs=inputs, outputs=inner)

    optimizer = RMSprop(learning_rate=learning_rate)
    head.compile(loss=wasserstein_loss, optimizer=optimizer)

    return head


def create_combined(generator, discriminator, learning_rate=5e-5):
    inputs = Input(shape=generator.input_shape[1:])
    inner = inputs

    inner = generator(inner)

    discriminator.trainable = False
    inner = discriminator(inner)

    outputs = inner

    combined = Model(inputs=inputs, outputs=outputs)

    optimizer = RMSprop(learning_rate=learning_rate)
    combined.compile(loss=wasserstein_loss, optimizer=optimizer)

    return combined


def upscale_generator(generator, filters=128):
    inputs = Input(shape=generator.input_shape[1:])
    inner = inputs

    inner = generator(inner)

    inner = generator_layer(inner, filters, kernel_size=(4,4))

    base = Model(inputs=inputs, outputs=inner)
    base.compile(loss='mse', optimizer='adam')

    return base


def upscale_discriminator(discriminator, learning_rate=5e-5):
    inputs = Input(shape=(
        discriminator.input_shape[1] * 2,
        discriminator.input_shape[2] * 2,
        3
    ))
    inner = inputs

    inner = discriminator_layer(inner, 3, kernel_size=(4,4))

    inner = discriminator(inner)

    head = Model(inputs=inputs, outputs=inner)

    optimizer = RMSprop(learning_rate=learning_rate)
    head.compile(loss=wasserstein_loss, optimizer=optimizer)

    return head
