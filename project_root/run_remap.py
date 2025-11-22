import cv2 as cv
from patchmatch import *
from utils import *


def main():
    scale = 0.1
    # remplacer par les images souhaitées
    img1 = cv.imread("input/trainspotting_2.png", cv.IMREAD_COLOR)
    img2 = cv.imread("input/trainspotting_1.png", cv.IMREAD_COLOR)
    img2 = cv.resize(img2, None, fx=scale, fy=scale, interpolation=cv.INTER_LINEAR)
    img1 = cv.resize(img1, None, fx=scale, fy=scale, interpolation=cv.INTER_LINEAR)

    r = 4
    offsets = init_off(img1)
    # if B is the source image and A the image to remap, we compute the offsets from A to B
    offsets, _ = patchmatch(img1, img2, r, offsets, nb_iters=5)
    
    new_img = remap(img2, offsets, 3)
    cv.imshow("Reconstructed Image", new_img)
    cv.waitKey(0)
    cv.destroyAllWindows()


if __name__ == "__main__":
    main()
