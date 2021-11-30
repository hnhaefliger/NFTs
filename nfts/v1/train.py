import tensorflow as tf
from tensorflow.keras.preprocessing.image import save_img
import time
import numpy as np
import tqdm


def train(
        data,
        n_noise=128,
        n_batches=100000,
        gen_batches=1,
        disc_batches=3,
        debug=False,
        debug_path=False,
        save=False,
        generator_save_path=False,
        discriminator_save_path=False
    ):

    fake_y = np.ones((data.batch_size, 1))
    real_y = -1 * np.ones((data.batch_size, 1))
    disc_y = np.concatenate((real_y, fake_y), axis=0)

    generator = model.create_generator(seed_dim=n_noise, load_path='nft_generator.h5')
    
    discriminator = model.create_discriminator(input_shape=(64, 64, 3), load_path='nft_discriminator.h5')
    discriminator.compile(loss=model.wasserstein_loss, optimizer=tf.keras.optimizers.RMSprop(learning_rate=0.00005))
    
    combined = model.create_combined(generator, discriminator, seed_dim=n_noise)
    combined.compile(loss=model.wasserstein_loss, optimizer=tf.keras.optimizers.RMSprop(learning_rate=0.00005))

    bar = tqdm.tqdm(range(n_batches), desc='training', leave=True, unit='B')

    for batch in bar:
        for _ in range(disc_batches):
            real_x = data[0]
            gen_x = np.random.normal(0, 1, (data.batch_size, n_noise))

            fake_x = generator.predict(gen_x)
            disc_x = np.concatenate((real_x, fake_x), axis=0)

            discriminator_loss = discriminator.train_on_batch(disc_x, disc_y)

        discriminator.trainable = False

        for _ in range(gen_batches):
            gen_x = np.random.normal(0, 1, (data.batch_size, n_noise))

            generator_loss = combined.train_on_batch(gen_x, real_y)

        bar.set_postfix(disc_loss=discriminator_loss, gen_loss=generator_loss)

        if debug:
            if batch % debug == 0:
                save_img(f'{debug_path}/debug_{int(time.time())}.png', tf.concat([*fake_x], axis=1))

        if save:
            if batch % save == 0:
                generator.save(generator_save_path)
                discriminator.save(discriminator_save_path)


if __name__ == '__main__':
    import os
    import sys
    import model
    import dataset

    dataset_path = sys.argv[1]

    data = dataset.Dataset([dataset_path + '/' + image for image in os.listdir(dataset_path)], 16)

    train(
        data,
        n_noise=256,
        n_batches=100000,
        gen_batches=1,
        disc_batches=3,
        debug=250,
        debug_path='nft_debug',
        save=250,
        generator_save_path='nft_generator.h5',
        discriminator_save_path='nft_discriminator.h5'
    )
