import numpy as np
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.utils import Sequence


class Dataset(Sequence):
    def __init__(self, images, batch_size):
        self.real_x = np.array(images)
        self.batch_size = batch_size

    def __len__(self):
        return len(self.real_x)

    def __getitem__(self, idx):
        real_x = self.real_x[np.random.randint(0, high=len(self.real_x)-1, size=(self.batch_size,))]
        real_x = [img_to_array(load_img(image, target_size=(64, 64))) for image in real_x]
        real_x = np.array(real_x) / 255

        return real_x
