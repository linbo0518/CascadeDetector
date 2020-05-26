import cv2
import numpy as np
from PIL import ImageDraw, JpegImagePlugin


def show_bboxes(image, bboxes, landmarks=[]):
    if isinstance(image, JpegImagePlugin.JpegImageFile):
        image_copy = image.copy()
        draw = ImageDraw.Draw(image_copy)

        for bbox in bboxes:
            draw.rectangle([(bbox[0], bbox[1]), (bbox[2], bbox[3])],
                           outline='white')

        for p in landmarks:
            for i in range(5):
                r = 1
                draw.ellipse(
                    ((p[i] - r, p[i + 5] - r, p[i] + r, p[i + 5] + r)),
                    fill='blue')
        return image_copy

    if isinstance(image, np.ndarray):
        image_copy = image.copy()

        for bbox in bboxes:
            image_copy = cv2.rectangle(image_copy, (bbox[0], bbox[1]),
                                       (bbox[2], bbox[3]), (255, 255, 255))
        for p in landmarks:
            for i in range(5):
                r = 1
                image_copy = cv2.circle(image_copy, (p[i], p[i + 5]), r,
                                        (255, 0, 0), -1)
        return image_copy