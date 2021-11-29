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


def create_generator(seed_dim=128, load_path=False):
    inputs = Input(shape=(seed_dim))
    inner = inputs

    inner = Dense(seed_dim * 8 * 8, activation='relu')(inner)
    inner = Reshape((8, 8, seed_dim))(inner)

    inner = UpSampling2D()(inner)

    inner = Conv2D(128, (4, 4), activation='relu', padding="same")(inner)
    inner = BatchNormalization(momentum=0.8)(inner)

    inner = UpSampling2D()(inner)

    inner = Conv2D(64, (4, 4), activation='relu', padding="same")(inner)
    inner = BatchNormalization(momentum=0.8)(inner)

    inner = UpSampling2D()(inner)

    inner = Conv2D(32, (4, 4), activation='relu', padding="same")(inner)
    inner = BatchNormalization(momentum=0.8)(inner)

    inner = Conv2D(3, kernel_size=4, padding="same")(inner)

    outputs = Activation('sigmoid')(inner)

    generator = Model(inputs=inputs, outputs=outputs)
    generator.compile(loss='mse', optimizer=RMSprop(learning_rate=0.00005))

    if load_path:
        try:
            generator.load_weights(load_path)
        
        except:
            print('failed to load generator')

    return generator


def create_discriminator(input_shape=(64, 64, 3), load_path=False):
    inputs = Input(shape=input_shape)
    inner = inputs

    inner = Conv2D(16, kernel_size=3, strides=2, padding="same", kernel_constraint=clamp_weights)(inner)
    inner = LeakyReLU(alpha=0.2)(inner)

    inner = Dropout(0.25)(inner)

    inner = Conv2D(32, kernel_size=3, strides=2, padding="same", kernel_constraint=clamp_weights)(inner)
    inner = LeakyReLU(alpha=0.2)(inner)
    inner = BatchNormalization(momentum=0.8)(inner)

    inner = Dropout(0.25)(inner)

    inner = Conv2D(64, kernel_size=3, strides=2, padding="same", kernel_constraint=clamp_weights)(inner)
    inner = LeakyReLU(alpha=0.2)(inner)
    inner = BatchNormalization(momentum=0.8)(inner)

    inner = Dropout(0.25)(inner)

    inner = Conv2D(128, kernel_size=3, strides=1, padding="same", kernel_constraint=clamp_weights)(inner)
    inner = LeakyReLU(alpha=0.2)(inner)
    inner = BatchNormalization(momentum=0.8)(inner)

    inner = Dropout(0.25)(inner)

    inner = Flatten()(inner)

    outputs = Dense(1, kernel_constraint=clamp_weights)(inner)

    discriminator = Model(inputs=inputs, outputs=outputs)
    discriminator.compile(loss=wasserstein_loss, optimizer=RMSprop(learning_rate=0.00005))

    if load_path:
        try:
            discriminator.load_weights(load_path)

        except: 
            print('failed to load generator')

    return discriminator


def create_models(seed_dim=128, generator_load_path=False, discriminator_load_path=False):
    generator = create_generator(seed_dim=seed_dim, load_path=generator_load_path)
    discriminator = create_discriminator(input_shape=(64, 64, 3), load_path=discriminator_load_path)

    inputs = Input(shape=(seed_dim))
    inner = inputs

    inner = generator(inner)

    discriminator.trainable = False
    inner = discriminator(inner)

    outputs = inner

    combined = Model(inputs=inputs, outputs=outputs)
    combined.compile(loss=wasserstein_loss, optimizer=RMSprop(learning_rate=0.00005))

    return generator, discriminator, combined
