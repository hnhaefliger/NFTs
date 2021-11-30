import numpy as np
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.utils import Sequence


class Dataset(Sequence):
    def __init__(self, images, batch_size=16):
        self.images = images
        self.batch_size = batch_size

    def __getitem__(self, resolution=(4,4)):
        images = np.random.choice(self.images, size=self.batch_size)
        images = [img_to_array(load_img(image, target_size=resolution)) for image in images]
        images = np.array(images) / 255

        return images
