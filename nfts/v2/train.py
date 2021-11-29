import tensorflow as tf
from tensorflow.keras.preprocessing.image import save_img
import os
import numpy as np
import sys
import time
import tqdm
import model
import dataset


def train(data, n_upscales=4, batches_per_res=10000, batches_per_gen=5, debug=False, debug_path=False, save=False, generator_save_path=False, discriminator_save_path=False):
    fake_y = np.ones((data.batch_size, 1))
    real_y = -1 * np.ones((data.batch_size, 1))

    if debug:
        if not os.path.isdir(debug_path):
            os.mkdir(debug_path)


    generator_base = model.generator_base()
    generator = model.generator_head(generator_base)

    discriminator = model.discriminator_head(learning_rate=5e-5)

    combined = model.create_combined(generator, discriminator, learning_rate=5e-5)

    resolution = 4


    for upscale in n_upscales:
        generator_base = model.upscale_generator(generator_base, filters=128)
        generator = model.generator_head(generator_base)

        discriminator = model.upscale_discriminator(discriminator, learning_rate=5e-5)

        combined = model.create_combined(generator, discriminator, learning_rate=5e-5)

        resolution = resolution * 4

        bar = tqdm.tqdm(range(batches_per_res), desc=resolution, leave=True, unit='B')

        for batch in bar:
            for disc_batch in range(batches_per_gen):
                real_x, gen_x = data.get_batch((resolution, resolution))

                fake_x = generator.predict(gen_x)

                disc_x = np.concatenate((real_x, fake_x), axis=0)
                disc_y = np.concatenate((real_y, fake_y), axis=0)

                discriminator_loss = discriminator.train_on_batch(disc_x, disc_y)

            discriminator.trainable = False
            generator_loss = combined.train_on_batch(gen_x, real_y)

            bar.set_postfix(disc_loss=discriminator_loss, gen_loss=generator_loss)

            if debug:
                if batch % debug == 0:
                    save_img(f'debug_path/debug_{time.time()}.png', tf.concat([*fake_x], axis=1))

            if save:
                if batch % save == 0:
                    generator.save(generator_save_path)
                    discriminator.save(discriminator_save_path)


if __name__ == '__main__':
    dataset_path = sys.argv[1]

    data = dataset.Dataset([dataset_path + '/' + image for image in os.listdir(dataset_path)])

    train(data, debug=250, debug_path='nft_debug', save=250, generator_save_path='nft_generator.h5', discriminator_save_path='nft_discriminator.h5')
