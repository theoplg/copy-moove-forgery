import cv2 as cv
import os


def ensure_exists(path):
    d = os.path.dirname(path)
    if d and not os.path.exists(d):
        os.makedirs(d, exist_ok=True)


def load_image(path, flags=cv.IMREAD_COLOR):
    img = cv.imread(path, flags)
    if img is None:
        raise FileNotFoundError(path)
    return img


def save_image(path, img):
    ensure_exists(path)
    cv.imwrite(path, img)
