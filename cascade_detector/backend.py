import cv2
import numpy as np
from PIL import Image


class OpenCVBackend:
    def __init__(self):
        pass

    def __repr__(self):
        return "OpenCV Backend"

    @classmethod
    def read_image(cls, filename, *args):
        return cv2.imread(filename, *args)

    @classmethod
    def get_image_size(cls, image):
        if image.ndim == 2:
            height, width = image.shape
        elif image.ndim == 3:
            height, width, _ = image.shape
        else:
            raise NotImplementedError
        return width, height

    @classmethod
    def to_array(cls, image):
        return image

    @classmethod
    def to_rgb_array(cls, image):
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return image

    @classmethod
    def from_array(cls, array):
        return array

    @classmethod
    def resize(cls,
               image,
               dst_width,
               dst_height,
               interpolation=cv2.INTER_LINEAR):
        return cv2.resize(image, (dst_width, dst_height),
                          interpolation=interpolation)


class PillowBackend:
    def __init__(self):
        pass

    def __repr__(self):
        return "Pillow(PIL) Backend"

    @classmethod
    def read_image(cls, filename):
        return Image.open(filename)

    @classmethod
    def get_image_size(cls, image):
        width, height = image.size
        return width, height

    @classmethod
    def to_array(cls, image):
        return np.asarray(image)

    @classmethod
    def to_rgb_array(cls, image):
        return cls.to_array(image)

    @classmethod
    def from_array(cls, array):
        return Image.fromarray(array)

    @classmethod
    def resize(cls,
               image,
               dst_width,
               dst_height,
               interpolation=Image.BILINEAR):
        return image.resize((dst_width, dst_height), interpolation)
