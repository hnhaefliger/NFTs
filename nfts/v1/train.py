import tensorflow as tf
from tensorflow.keras.preprocessing.image import save_img
import matplotlib.pyplot as plt
import numpy as np
import tqdm


def train(generator, discriminator, combined, data, n_batches, disc_batches=5, debug=False, debug_path=False, save=False, generator_save_path=False, discriminator_save_path=False):
    fake_y = np.ones((data.batch_size, 1))
    real_y = -1 * np.ones((data.batch_size, 1))

    bar = tqdm.tqdm(range(n_batches), desc='training', leave=True, unit='B')

    for batch in bar:
        for disc_batch in range(disc_batches):
            gen_x, real_x = data[np.random.randint(0, high=len(data)-1, size=(16,))]

            fake_x = generator.predict(gen_x)

            disc_x = np.concatenate((real_x, fake_x), axis=0)
            disc_y = np.concatenate((real_y, fake_y), axis=0)

            discriminator_loss = discriminator.train_on_batch(disc_x, disc_y)

        discriminator.trainable = False
        generator_loss = combined.train_on_batch(gen_x, real_y)

        bar.set_postfix(disc_loss=discriminator_loss, gen_loss=generator_loss)

        if debug:
            if batch % debug == 0:
                save_img(debug_path, tf.concat([*fake_x], axis=1))

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

    generator, discriminator, combined = model.create_models(generator_load_path='nft_generator.h5', discriminator_load_path='nft_discriminator.h5')

    train(generator, discriminator, combined, data, 100000, disc_batches=3, debug=250, debug_path='nft_debug.png', save=250, generator_save_path='nft_generator.h5', discriminator_save_path='nft_discriminator.h5')
