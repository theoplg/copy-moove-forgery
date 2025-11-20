import matplotlib.pyplot as plt
import cv2 as cv


def show(img, title=None):
    if img.ndim == 3:
        img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
    plt.imshow(img, cmap=None if img.ndim == 3 else "gray")
    if title:
        plt.title(title)
    plt.axis("off")
    plt.show()


def show_gray(img, title=None):
    plt.imshow(img, cmap="gray")
    if title:
        plt.title(title)
    plt.axis("off")
    plt.show()


def show_side_by_side(img1, img2, title=None):
    if img1.ndim == 3:
        img1 = cv.cvtColor(img1, cv.COLOR_BGR2RGB)
    if img2.ndim == 3:
        img2 = cv.cvtColor(img2, cv.COLOR_BGR2RGB)

    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.imshow(img1, cmap=None if img1.ndim == 3 else "gray")
    plt.axis("off")

    plt.subplot(1, 2, 2)
    plt.imshow(img2, cmap=None if img2.ndim == 3 else "gray")
    plt.axis("off")

    if title:
        plt.suptitle(title)

    plt.show()
