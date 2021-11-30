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
        n_upscales=4,
        batches_per_res=10000,
        disc_batches=1,
        gen_batches=1,
        debug=False,
        debug_path=False,
        save=False,
        generator_save_path=False,
        discriminator_save_path=False,
        n_styles=256,
        n_channels=256,
        momentum=0.8,
        generator_learning_rate=1e-3,
        discriminator_learning_rate=1e-3,
    ):
    if debug:
        if not os.path.isdir(debug_path):
            os.mkdir(debug_path)


    resolution = 4
    fake_y = np.ones((data.batch_size, 1))
    real_y = -1 * np.ones((data.batch_size, 1))
    disc_y = np.concatenate((real_y, fake_y), axis=0)
    constant = np.ones((data.batch_size, 1))


    generator_base, generator_head, generator = model.create_generator(n_styles=n_styles, n_channels=n_channels, momentum=momentum)

    discriminator_base, discriminator_head, discriminator = model.create_discriminator(n_channels=n_channels)
    discriminator.compile(loss=model.wasserstein_loss, optimizer=tf.keras.optimizers.Adam(learning_rate=discriminator_learning_rate))

    combined = model.create_combined(generator, discriminator, n_styles=n_styles)
    combined.compile(loss=model.wasserstein_loss, optimizer=tf.keras.optimizers.Adam(learning_rate=generator_learning_rate))


    for upscale in range(n_upscales):
        # grow models
        generator_base, generator_head, generator = model.grow_generator(generator_base, generator_head, n_styles=n_styles, n_channels=n_channels, momentum=momentum)

        discriminator_base, discriminator_head, discriminator = model.grow_discriminator(discriminator_base, discriminator_head, n_channels=n_channels, momentum=momentum)
        discriminator.compile(loss=model.wasserstein_loss, optimizer=tf.keras.optimizers.Adam(learning_rate=discriminator_learning_rate))

        combined = model.create_combined(generator, discriminator, n_styles=n_styles)
        combined.compile(loss=model.wasserstein_loss, optimizer=tf.keras.optimizers.Adam(learning_rate=generator_learning_rate))

        resolution = resolution * 2

        # resolution epoch
        bar = tqdm.tqdm(range(batches_per_res), desc=f'resolution {resolution}', leave=True, unit='B')

        for batch in bar:
            # discriminator training
            for _ in range(disc_batches):
                real_x = data.get_batch((resolution, resolution))
                gen_x = noise = np.random.normal(0, 1, (data.batch_size, n_styles))

                fake_x = generator.predict([gen_x, constant])

                disc_x = np.concatenate((real_x, fake_x), axis=0)

                discriminator_loss = discriminator.train_on_batch(disc_x, disc_y)


            # generator training
            discriminator.trainable = False

            for _ in range(gen_batches):
                real_x = data.get_batch((resolution, resolution))
                gen_x = noise = np.random.normal(0, 1, (data.batch_size, n_styles))

                generator_loss = combined.train_on_batch([gen_x, constant], real_y)


            # utils
            bar.set_postfix(disc_loss=discriminator_loss, gen_loss=generator_loss)

            if debug:
                if batch % debug == 0:
                    save_img(f'{debug_path}/debug_{int(time.time())}.png', tf.concat([*fake_x], axis=1))

            if save:
                if batch % save == 0:
                    generator.save(generator_save_path)
                    discriminator.save(discriminator_save_path)



if __name__ == '__main__':
    dataset_path = sys.argv[1]

    data = dataset.Dataset([dataset_path + '/' + image for image in os.listdir(dataset_path)])

    train(
        data,
        n_upscales=4,
        batches_per_res=10000,
        disc_batches=1,
        gen_batches=1,
        debug=250,
        debug_path='nft_debug',
        save=250,
        generator_save_path='nft_generator.h5',
        discriminator_save_path='nft_discriminator.h5',
        n_styles=256,
        n_channels=256,
        momentum=0.8,
        generator_learning_rate=1e-3,
        discriminator_learning_rate=1e-3,
    )
