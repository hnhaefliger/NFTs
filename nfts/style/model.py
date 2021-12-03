from .discriminator import PhoramaDiscriminator
from tensorflow.keras.layers import Input, Dense, Conv2D, BatchNormalization, LeakyReLU, Flatten
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Activation, Concatenate, Conv2DTranspose


def u_net_layer(x, filters):
    for _ in range(2):
        x = Conv2D(filters, (3, 3), strides=1, padding='same')(x)
        x = Activation('relu')(x)

    return x


def down_conv(x, filters):
    x = MaxPooling2D((2, 2))(x)

    return x


def up_conv(x, filters):
    x = Conv2DTranspose(filters, (2, 2), strides=1, padding='same')


def concat(x, y):
    inner = Concatenate()(x, y)

    return inner


def u_net():
    inputs = Input(shape=(None, None, 3))
    inner = inputs

    inner1 = u_net_layer(inner, 64)

    inner2 = down_conv(inner1, 64)
    inner2 = u_net_layer(inner2, 128)
    
    inner3 = down_conv(inner2, 128)
    inner3 = u_net_layer(inner3, 256)

    inner4 = down_conv(inner3, 256)
    inner4 = u_net_layer(inner4, 512)

    inner5 = down_conv(inner4, 512)
    inner5 = u_net_layer(inner5, 1024)
    inner5 = up_conv(inner5, 512)

    inner4 = concat(inner5, inner4)
    inner4 = u_net_layer(inner4, 512)
    inner4 = up_conv(inner4, 256)

    inner3 = concat(inner4, inner3)
    inner3 = u_net_layer(inner3, 512)
    inner3 = up_conv(inner3, 128)

    inner2 = concat(inner3, inner2)
    inner2 = u_net_layer(inner2, 128)
    inner2 = up_conv(inner2, 64)

    inner1 = concat(inner2, inner1)
    inner1 = u_net_layer(inner1, 64)

    inner = Conv2D(3, (3, 3), strides=1, padding='same')(inner1)
    inner = Activation('sigmoid')(inner)

    return Model(inputs=inputs, outputs=inner)


def vgg():
    return tf.keras.applications.VGG19(include_top=False, weights='imagenet')


def d_block(inputs, filters, strides=1, bn=True):
    x = Conv2D(filters, (3, 3), strides=strides, padding='same')(inputs)
    x = LeakyReLU(alpha=0.2)(x)

    if bn:
        x = BatchNormalization(momentum=0.8)(x)

    return x


def discriminator(resolution=16):
    inputs = Input(shape=(resolution, resolution, 3))
    inner = inputs

    inner = d_block(inner, 64, bn=False)
    inner = d_block(inner, 64, strides=2)

    filters = 128
    for _ in range(3):
        inner = d_block(inner, filters)
        inner = d_block(inner, filters, strides=2)
        filters = filters * 2

    inner = Flatten()(inner)

    inner = Dense(1024)(inner)
    inner = LeakyReLU(alpha=0.2)(inner)

    outputs = Dense(1, activation='sigmoid')(inner)

    return Model(inputs=inputs, outputs=outputs)


def combined(generator, discriminator, vgg):
    inputs = Input(shape=(None, None, 3))
    inner = inputs

    inner = generator(inner)

    style = discriminator(inner)
    content = vgg(inner)

    return Model(inputs=inputs, outputs=[style, content])
