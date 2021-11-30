import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Conv2D, BatchNormalization, Reshape, Multiply, LeakyReLU, Flatten, Add, UpSampling2D, Activation, Layer, PReLU
from tensorflow.keras.optimizers import RMSprop


def clamp_weights(w):
    return tf.keras.backend.clip(w, -0.01, 0.01)


def wasserstein_loss(y_true, y_pred):
  return tf.reduce_mean(y_true * y_pred)


def AdaIN(styles, inner, n_styles=256, n_channels=256):
    styles1 = Dense(n_channels)(styles)
    styles1 = Reshape((1, 1, n_channels))(styles1)

    styles2 = Dense(n_channels)(styles)
    styles2 = Reshape((1, 1, n_channels))(styles2)

    inner = Multiply()([inner, styles1 + 1])
    inner = Add()([inner, styles2])

    return inner


def generator_base(n_styles=256, n_channels=256, momentum=0.8):
    styles = Input(shape=(n_styles))
    inputs = Input(shape=(1,))  # constant
    inner = inputs

    inner = Dense(4 * 4 * n_channels)(inputs)
    inner = Reshape((4, 4, n_channels))(inner)
    # noise
    inner = PReLU()(inner)
    inner = BatchNormalization(axis=3, momentum=momentum)(inner)

    inner = AdaIN(styles, inner, n_styles=n_styles, n_channels=n_channels)

    inner = Conv2D(n_channels, (3, 3), padding='same')(inner)
    # noise
    inner = PReLU()(inner)
    inner = BatchNormalization(axis=3, momentum=momentum)(inner)
    
    inner = AdaIN(styles, inner, n_styles=n_styles, n_channels=n_channels)

    return Model(inputs=[styles, inputs], outputs=inner)


def generator_block(n_styles=256, n_channels=256, momentum=0.8):
    styles = Input(shape=(n_styles))
    inputs = Input(shape=(1,))
    inner = inputs

    inner = UpSampling2D()(inner)

    inner = Conv2D(n_channels, (3, 3), strides=1, padding='same')(inner)
    # noise
    inner = PReLU()(inner)
    inner = BatchNormalization(axis=3, momentum=momentum)(inner)

    inner = AdaIN(styles, inner, n_styles=n_styles, n_channels=n_channels)

    inner = Conv2D(n_channels, (3, 3), strides=1, padding='same')(inner)
    # noise
    inner = PReLU()(inner)
    inner = BatchNormalization(axis=3, momentum=momentum)(inner)

    inner = AdaIN(styles, inner, n_styles=n_styles, n_channels=n_channels)

    return Model(inputs=[styles, inputs], outputs=inner)


def generator_head(n_channels=256):
    inputs = Input(shape=(None, None, n_channels))
    inner = inputs

    inner = Conv2D(3, (1, 1), strides=1, padding='same')(inner)
    inner = Activation('sigmoid')(inner)

    return Model(inputs=inputs, outputs=inner)


def create_generator(n_styles=256, n_channels=256, momentum=0.8):
    styles = Input(shape=(n_styles))
    inputs = Input(shape=(1,))
    inner = inputs

    base = generator_base(n_styles=n_styles, n_channels=n_channels, momentum=momentum)
    inner = base([styles, inner])

    head = generator_head(n_channels=n_channels)
    inner = head(inner)

    return base, head, Model(inputs=[styles, inputs], outputs=inner)


def grow_generator(base, head, n_styles=256, n_channels=256, momentum=0.8):
    styles = Input(shape=(n_styles))
    inputs = Input(shape=(1,))
    inner = inputs
    
    inner = base(inner)

    base = generator_block(n_styles=n_styles, n_channels=n_channels, momentum=momentum)
    inner = base([styles, inner])

    inner = head(inner)

    return base, head, Model(inputs=[styles, inputs], outputs=inner)


def discriminator_block(resolution, n_channels):
    pass


base, head, generator = create_generator()
base, head, generator = grow_generator(base, head)
