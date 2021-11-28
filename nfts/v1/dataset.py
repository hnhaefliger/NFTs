import numpy as np
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.utils import Sequence


class Dataset(Sequence):
    def __init__(self, images, batch_size):
        self.real_x = np.array(images)
        np.random.shuffle(self.real_x)
        self.batch_size = batch_size

    def __len__(self):
        return len(self.real_x)

    def __getitem__(self, idx):
        real_x = self.real_x[idx]
        real_x = [img_to_array(load_img(image, target_size=(64, 64))) for image in real_x]

        real_x = np.array(real_x) / 255
        gen_x = np.random.normal(0, 1, (self.batch_size, 128))

        return gen_x, real_x
