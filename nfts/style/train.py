import tensorflow as tf
from tensorflow.keras.preprocessing.image import save_img
import os
import numpy as np
import sys
import time
import tqdm
import model
import dataset


def train(
        data,
        batches,
        debug=False,
        debug_path=False,
        save=False,
        generator_save_path=False,
        discriminator_save_path=False,
        generator_learning_rate=1e-3,
        discriminator_learning_rate=1e-3,
        resolution=16,
        style_weight=1e-3,
        content_weight=1e0,
    ):
    if debug:
        if not os.path.isdir(debug_path):
            os.mkdir(debug_path)

    fake_y = np.zeros((data.batch_size, 1))
    real_y = np.ones((data.batch_size, 1))
    disc_y = np.concatenate((real_y, fake_y), axis=0)


    generator = model.u_net()
    vgg = model.vgg()
    discriminator = model.discriminator(resolution=resolution)
    combined = model.combined(generator, discriminator, vgg)

    discriminator.compile(loss='binary_crossentropy', optimizer=tf.keras.optimizers.Adam(learning_rate=discriminator_learning_rate))
    combined.compile(loss=['binary_crossentropy', 'mse'], loss_weights=[style_weight, content_weight], optimizer=tf.keras.optimizers.Adam(learning_rate=generator_learning_rate))

    bar = tqdm.tqdm(range(batches), desc=f'training', leave=True, unit='B')

    for batch in bar:
        content_images, style_images = data[batch]

        content_vgg = vgg.predict(content_images)
        fake_images = generator.predict(content_images)

        disc_x = np.concatenate((style_images, fake_images), axis=0)

        discriminator_loss = discriminator.train_on_batch(disc_x, disc_y)

        discriminator.trainable = False
        vgg.trainable = False

        generator_loss = combined.train_on_batch(content_images, [real_y, content_vgg])

        # utils
        bar.set_postfix(disc_loss=discriminator_loss, gen_loss=generator_loss)

        if debug:
            if batch % debug == 0:
                save_img(f'{debug_path}/debug_{int(time.time())}.png', tf.concat([*fake_images], axis=1))

        if save:
            if batch % save == 0:
                generator.save(generator_save_path)
                discriminator.save(discriminator_save_path)


if __name__ == '__main__':
    content_path = sys.argv[1]
    style_path = sys.argv[2]

    data = dataset.Dataset(
        [f'{content_path}/{image}' for image in os.listdir(content_path)],
        [f'{style_path}/{image}' for image in os.listdir(style_path)],
        resolution=(32, 32),
    )

    train(
        data,
        int(sys.argv[3]),
        debug=250,
        debug_path='nft_debug',
        save=250,
        generator_save_path='nft_generator.h5',
        discriminator_save_path='nft_discriminator.h5',
        generator_learning_rate=1e-3,
        discriminator_learning_rate=1e-3,
        resolution=32,
        style_weight=float(sys.argv[4]),
        content_weight=float(sys.argv[5]),
    )
