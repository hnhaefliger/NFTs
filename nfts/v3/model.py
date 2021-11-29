import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Conv2D, Lambda, Reshape, Layer, LeakyReLU, Flatten, Add, UpSampling2D, Activation, RepeatVector, PReLU, Multiply
import numpy as np
from keras import backend
from keras.utils import tf_utils


def clamp_weights(w):
    return tf.keras.backend.clip(w, -0.01, 0.01)


def wasserstein_loss(y_true, y_pred):
  return tf.reduce_mean(y_true * y_pred)

'''
class Gaussian(Layer):
    def build(self, input_shape):
        self.n, self.h, self.w, self.c = input_shape
        self.channel_weights = self.add_weight(shape=[1, 1, 1, self.c], trainable=True)

    def call(self, x):
        noise = tf.random.random_normal(shape=(x.shape[0], x.shape[1], 1), mean=0., stddev=1., dtype=x.dtype)()
        outputs = x + self.b * noise
        return outputs
'''


class Gaussian(Layer):
    def __init__(self, stddev, seed=None, **kwargs):
        super(Gaussian, self).__init__(**kwargs)
        self.supports_masking = True
        self.stddev = stddev
        self.seed = seed
        self._random_generator = backend.RandomGenerator(seed)

    def call(self, inputs, training=None):
        def noised():
            return inputs + self._random_generator.random_normal(
                shape=tf.shape(inputs),
                mean=0.,
                stddev=self.stddev,
                dtype=inputs.dtype)

        return noised()

    def get_config(self):
        config = {'stddev': self.stddev, 'seed': self.seed}
        base_config = super(Gaussian, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    @tf_utils.shape_type_conversion
    def compute_output_shape(self, input_shape):
        return input_shape


class AdaIN(Layer):
    def build(self, input_shapes):
        self.x_shape = input_shapes[0]
        self.styles_shape = input_shapes[1]

        self.dense = Dense(self.self.x_shape[-1])
        self.reshape = Reshape((self.x_shape[-1], 2))

    def call(self, x):
        x, styles = x
        styles = self.dense(x.shape[1] * 2)(styles)
        styles = self.reshape(styles)

        return x * (styles[:, 0] + 1) + styles[:, 1]


def StyleGanMapping(style_dimensions=256, mapping_dimensions=256, mapping_layers=8):
    inputs = Input(shape=(style_dimensions,))
    inner = inputs

    for _ in range(mapping_layers):
        inner = Dense(mapping_dimensions)(inner)

    mapping = Model(inputs=inputs, outputs=inner)
    mapping.compile(loss='mse', optimizer='adam')

    return mapping


def StyleGanBase(mapping_dimensions=256, n_channels=256):
    def expand_inputs(inputs):
        inner = tf.repeat(inputs, 4, axis=0)
        inner = tf.repeat(inner, 4, axis=1)
        inner = tf.repeat(inner, n_channels, axis=2)

        return inner

    styles = Input(shape=(mapping_dimensions,))

    inputs = Input(shape=(1,)) # constant
    inner = inputs
    inner = Reshape((1, 1, 1))(inner)

    inner = Lambda(expand_inputs, output_shape=(4, 4, n_channels))(inner)

    inner = Gaussian()(inner)
    inner = AdaIN()(inner, styles)
    inner = Conv2D(n_channels, (3, 3), padding='same')(inner)
    inner = PReLU()(inner)
    inner = Gaussian()(inner)
    inner = AdaIN()(inner, styles)

    base = Model(inputs=[styles, inputs], outputs=inner)
    base.compile(loss='mse', optimizer='adam')

    return base


def StyleGanLayer(inner, styles, n_channels=512):
    inner = UpSampling2D()(inner)

    inner = Conv2D(n_channels, (3, 3), padding='same')(inner)
    inner = PReLU()(inner)
    inner = Gaussian()(inner)
    inner = AdaIN()(inner, styles)

    inner = Conv2D(n_channels, (3, 3), padding='same')(inner)
    inner = PReLU()(inner)
    inner = Gaussian()(inner)
    inner = AdaIN()(inner, styles)

    return inner


def StyleGanHead(dimensions=(4, 4, 256)):
    inputs = Input(shape=dimensions)
    inner = inputs

    inner = Conv2D(3, (1, 1), padding='same')(inner)
    inner = Activation('sigmoid')(inner)

    head = Model(inputs=inputs, outputs=inner)
    head.compile(loss='mse', optimizer='adam')

    return head


def StyleGan(style_dimensions=256, mapping_dimensions=256, mapping_layers=8, n_channels=256, style_layers=4):
    styles = StyleGanMapping(style_dimensions=style_dimensions, mapping_dimensions=mapping_dimensions, mapping_layers=mapping_layers)

    base = StyleGanBase(mapping_dimensions=mapping_dimensions, n_channels=n_channels)

    head = StyleGanHead(dimensions=base.shape)

    style_inputs = Input(shape=(style_dimensions,))

    constant_inputs = Input(shape=(1,))

    inner = styles(style_inputs)
    inner = base([inner, constant_inputs])
    inner = head(base)

    generator = Model(inputs=[style_inputs, constant_inputs], outputs=inner)
    generator.compile(loss='mse', optimizer='adam')

    return styles, base, generator


def grow_style_gan(styles, base, n_channels=256):
    base = StyleGanLayer(base, styles, n_channels=n_channels)

    style_inputs = Input(shape=styles.input_shape)

    constant_inputs = Input(shape=(1,))

    head = StyleGanHead(dimensions=base.shape)

    inner = styles(style_inputs)
    inner = base([inner, constant_inputs])
    inner = head(base)

    generator = Model(inputs=[style_inputs, constant_inputs], outputs=inner)
    generator.compile(loss='mse', optimizer='adam')

    return styles, base, generator


styles, base, generator = StyleGan()
styles, base, generator = grow_style_gan(styles, base)
