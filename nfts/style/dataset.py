import numpy as np
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.utils import Sequence
import random


def random_crop(image, resolution):
    x = random.randint(0, len(image[1]) - resolution[0])
    y = random.randint(0, len(image[0]) - resolution[1])

    return image[y:y+resolution[1], x:x+resolution[0]]


class Dataset(Sequence):
    def __init__(self, content_images, style_images, resolution=(32, 32), batch_size=16):
        self.content_images = content_images
        self.style_images = style_images
        self.batch_size = batch_size
        self.resolution = resolution

    def __getitem__(self, idx):
        content_images = np.random.choice(self.content_images, size=self.batch_size)
        content_images = [img_to_array(load_img(image)) for image in content_images]

        content_images = [random_crop(image, self.resolution) for image in content_images]
        content_images = np.array(content_images) / 255

        style_images = np.random.choice(self.style_images, size=self.batch_size)
        style_images = [img_to_array(load_img(image)) for image in style_images]

        style_images = [random_crop(image, self.resolution) for image in style_images]
        style_images = np.array(style_images) / 255

        return content_images, style_images
