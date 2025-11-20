import cv2 as cv


def gaussian_blur(img, ksize):
    return cv.GaussianBlur(img, (ksize, ksize), 0)


def bilateral_filter(img, d, sigma_color, sigma_space):
    return cv.bilateralFilter(img, d, sigma_color, sigma_space)


def normalize_img(img):
    return cv.normalize(img, None, 0, 255, cv.NORM_MINMAX)
