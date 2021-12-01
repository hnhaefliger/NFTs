import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Conv2D, BatchNormalization, Reshape, Dropout, GaussianNoise, Flatten, Add, UpSampling2D, Activation, PReLU


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


def create_generator(n_noise=256, seed_depth=1024, kernel_size=5, noise_reshape=8, n_upscales=3, momentum=0.8, load_path=0.8, bn=True, noise=False):
    assert kernel_size <= noise_reshape
    assert seed_depth // (2 ** n_upscales) == seed_depth / (2 ** n_upscales)

    inputs = Input(shape=(n_noise,))
    inner = inputs

    inner = Dense(seed_depth * noise_reshape * noise_reshape)(inner)
    inner = PReLU()(inner)
    inner = Reshape((noise_reshape, noise_reshape, seed_depth))(inner)

    for _ in range(n_upscales):
        noise_reshape = noise_reshape / 2

        inner = UpSampling2D()(inner)
        inner = Conv2D(noise_reshape, (kernel_size, kernel_size), strides=1, padding='same')(inner)
        inner = PReLU()(inner)
        
        if bn:
            inner = BatchNormalization(momentum=momentum)(inner)

        if noise:
            inner = GaussianNoise(1)(inner, training=True)

    inner = Conv2D(3, (kernel_size, kernel_size), strides=1, padding='same')
    inner = Activation('tanh')(inner)
    inner = (inner + tf.constant(1)) / tf.constant(2)

    generator = Model(inputs=inputs, outputs=inner)

    if load_path:
        try:
            generator.load_weights(load_path)

        except:
            print('failed to load generator')

    return generator


def create_discriminator(input_shape=(64, 64, 3), kernel_size=5, n_filters=32, n_downscales=4, momentum=0.8, load_path=False, bn=True, dropout=0.2):
    inputs = Input(shape=input_shape)
    inner = inputs

    for _ in range(n_downscales):
        inner = Conv2D(n_filters, (kernel_size, kernel_size), strides=2, padding='same', kernel_constraint=clamp_weights)(inner)
        inner = PReLU()(inner)

        if bn:
            inner = BatchNormalization(momentum=momentum)(inner)

        inner = Dropout(dropout)(inner)

        n_filters = n_filters * 2

    inner = Flatten()(inner)
    inner = Dense(1, kernel_constraint=clamp_weights)(inner)

    discriminator = Model(inputs=inputs, outputs=inner)

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

