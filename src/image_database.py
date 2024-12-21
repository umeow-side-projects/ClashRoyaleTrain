import cv2

from os import path

class ImageDatabase:
    @staticmethod
    def get_image(filename, flags=cv2.IMREAD_COLOR):
        return cv2.imread(f'src/image_database/{filename}', flags)

    @staticmethod
    def save_image(filename, img):
        cv2.imwrite(f'src/image_database/{filename}', img)