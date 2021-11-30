import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Conv2D, BatchNormalization, Reshape, Multiply, LeakyReLU, Flatten, Add, UpSampling2D, Activation, Layer, PReLU
from tensorflow.keras.optimizers import RMSprop


def clamp_weights(w):
    return tf.keras.backend.clip(w, -0.01, 0.01)


def wasserstein_loss(y_true, y_pred):
  return -tf.reduce_mean(y_true * y_pred)


def generator_base(n_noise=256, n_channels=256, momentum=0.8):
    inputs = Input(shape=(n_noise))
    inner = inputs

    inner = Dense(4 * 4 * n_channels)(inputs)
    inner = Reshape((4, 4, n_channels))(inner)
    inner = PReLU()(inner)

    inner = Conv2D(n_channels, (4, 4), padding='same')(inner)
    # noise
    inner = PReLU()(inner)
    inner = BatchNormalization(momentum=momentum)(inner)

    return Model(inputs=inputs, outputs=inner)


def generator_block(resolution, n_channels=256, momentum=0.8):
    inputs = Input(shape=resolution)
    inner = inputs

    inner = UpSampling2D()(inner)

    inner = Conv2D(n_channels, (4, 4), strides=1, padding='same')(inner)
    # noise
    inner = PReLU()(inner)
    #inner = BatchNormalization(momentum=momentum)(inner)

    inner = Conv2D(n_channels, (4, 4), strides=1, padding='same')(inner)
    # noise
    inner = PReLU()(inner)
    inner = BatchNormalization(momentum=momentum)(inner)

    return Model(inputs=[inputs], outputs=inner)


def generator_head(n_channels=256):
    inputs = Input(shape=(None, None, n_channels))
    inner = inputs

    inner = Conv2D(3, (1, 1), strides=1, padding='same')(inner)
    inner = Activation('sigmoid')(inner)

    return Model(inputs=inputs, outputs=inner)


def create_generator(n_noise=256, n_channels=256, momentum=0.8):
    inputs = Input(shape=(n_noise,))
    inner = inputs

    base = generator_base(n_noise=n_noise, n_channels=n_channels, momentum=momentum)
    inner = base(inner)

    head = generator_head(n_channels=n_channels)
    inner = head(inner)

    return base, head, Model(inputs=inputs, outputs=inner)


def grow_generator_base(base, n_noise=256, n_channels=256, momentum=0.8):
    inputs = Input(shape=(n_noise,))
    inner = inputs

    inner = base(inner)

    new_base = generator_block(base.output_shape[1:], n_channels=n_channels, momentum=momentum)
    inner = new_base(inner)

    return Model(inputs=inputs, outputs=inner)


def grow_generator(base, head, n_noise=256, n_channels=256, momentum=0.8):
    inputs = Input(shape=(n_noise,))
    inner = inputs

    base = grow_generator_base(base, n_noise=n_noise, n_channels=n_noise, momentum=momentum)
    inner = base(inner)

    inner = head(inner)

    return base, head, Model(inputs=inputs, outputs=inner)


def discriminator_base(n_channels=256):
    inputs = Input(shape=(None, None, 3))
    inner = inputs

    inner = Conv2D(n_channels, (1, 1), strides=1, padding='same', kernel_constraint=clamp_weights)(inner)

    return Model(inputs=inputs, outputs=inner)


def discriminator_block(resolution, n_channels=256, momentum=0.8):
    inputs = Input(shape=resolution)
    inner = inputs

    inner = Conv2D(n_channels, (4, 4), strides=2, padding='same', kernel_constraint=clamp_weights)(inner)
    inner = PReLU()(inner)
    inner = BatchNormalization(momentum=momentum)(inner)

    return Model(inputs=inputs, outputs=inner)


def discriminator_head(n_channels=256):
    inputs = Input(shape=(4, 4, n_channels))
    inner = inputs

    inner = Flatten()(inner)

    inner = Dense(1)(inner)
    inner = Activation('sigmoid')(inner)

    return Model(inputs=inputs, outputs=inner)


def create_discriminator(n_channels=256):
    inputs = Input(shape=(4, 4, 3))
    inner = inputs

    base = discriminator_base(n_channels=n_channels)
    inner = base(inner)

    head = discriminator_head(n_channels=n_channels)
    inner = head(inner)

    return base, head, Model(inputs=inputs, outputs=inner)


def grow_discriminator_head(head, n_channels=256, momentum=0.8):
    resolution = (head.input_shape[1]*2, head.input_shape[1]*2, n_channels)

    inputs = Input(shape=resolution)
    inner = inputs

    new_head = discriminator_block(resolution, n_channels=n_channels, momentum=momentum)
    inner = new_head(inner)

    inner = head(inner)

    return Model(inputs=inputs, outputs=inner)


def grow_discriminator(base, head, n_channels=256, momentum=0.8):
    resolution = (head.input_shape[1]*2, head.input_shape[1]*2, 3)

    inputs = Input(shape=resolution)
    inner = inputs

    inner = base(inner)

    head = grow_discriminator_head(head, n_channels=n_channels, momentum=momentum)
    inner = head(inner)

    return base, head, Model(inputs=inputs, outputs=inner)


def create_combined(generator, discriminator, n_noise=256):
    inputs = Input(shape=(n_noise,))
    inner = inputs

    inner = generator(inner)
    inner = discriminator(inner)

    return Model(inputs=inputs, outputs=inner)
