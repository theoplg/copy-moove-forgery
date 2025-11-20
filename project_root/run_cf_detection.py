import cv2 as cv
from patchmatch import *
from utils import *


def main():
    img1 = load_image("input/lena_modif_2.png")
    img2 = load_image("input/lena_modif_2.png")

    r = 4
    offsets = init_off(img1)
    offsets, _ = patchmatch(img1, img2, r, offsets, nb_iters=5)

    disp = displacement_map(offsets)
    mask, err = detection_mask_from_offsets(offsets)

    over = overlay_mask_on_image(img1, mask)

    save_image("output/disp.png", disp)
    save_image("output/mask.png", mask)
    save_image("output/overlay.png", over)


if __name__ == "__main__":
    main()
