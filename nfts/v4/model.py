import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Conv2D, BatchNormalization, Reshape, Dropout, LeakyReLU, Flatten, Add, UpSampling2D, Activation, Layer, PReLU
from tensorflow.keras.optimizers import RMSprop


def clamp_weights(w):
    return tf.keras.backend.clip(w, -0.01, 0.01)


def wasserstein_loss(y_true, y_pred):
  return tf.reduce_mean(y_true * y_pred)


class AdaIN(Layer):
    def build(self, input_shapes):
        self.styles_shape = input_shapes[0]
        self.x_shape = input_shapes[1]

        self.dense = Dense(self.x_shape[-1])
        self.reshape = Reshape((self.x_shape[-1], 2))

    def call(self, x):
        x, styles = x
        styles = self.dense(styles)
        styles = self.reshape(styles)

        return x * (styles[:, 0] + 1) + styles[:, 1]


def generator_base(n_styles=256, n_channels=256, momentum=0.8):
    styles = Input(shape=(n_styles))
    inputs = Input(shape=(1,))  # constant
    inner = inputs

    inner = Dense(4 * 4 * n_channels)(inputs)
    inner = Reshape((4, 4, n_channels))(inputs)
    # noise
    inner = PReLU()(inner)
    inner = BatchNormalization(axis=3, momentum=momentum)(inner)

    inner = AdaIN()(styles, inner)

    inner = Conv2D(n_channels, (3, 3), padding='same')(inner)
    # noise
    inner = PReLU()(inner)
    inner = BatchNormalization(axis=3, momentum=momentum)(inner)
    
    inner = AdaIN()(styles, inner)

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

    inner = AdaIN()(styles, inner)

    inner = Conv2D(n_channels, (3, 3), strides=1, padding='same')(inner)
    # noise
    inner = PReLU()(inner)
    inner = BatchNormalization(axis=3, momentum=momentum)(inner)

    inner = AdaIN()(styles, inner)

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

    base = generator_block(n_styles=n_styles, n_channels=n_channels, momentum=momentum)(base)
    inner = base([styles, inner])

    inner = head(inner)

    return base, head, Model(inputs=[styles, inputs], outputs=inner)


base, head, generator = create_generator()
base, head, generator = grow_generator(base, head)
