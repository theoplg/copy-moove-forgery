import cv2 as cv
from patchmatch import *
from utils import *
import sys

def main():
    # vérifier les arguments
    if len(sys.argv) < 2:
        print("Usage: python3 run_cf_detection.py <image> [iters]")
        return

    image_path = sys.argv[1]
    iters = int(sys.argv[2]) if len(sys.argv) > 2 else 5

    img1 = load_image(image_path)
    img2 = load_image(image_path)

    r = 4
    offsets = init_off(img1)
    offsets, _ = patchmatch(img1, img2, r, offsets, nb_iters=iters)

    disp = displacement_map(offsets)
    mask, err = detection_mask_from_offsets(offsets)
    over = overlay_mask_on_image(img1, mask)

    save_image("output/disp.png", disp)
    save_image("output/mask.png", mask)
    save_image("output/overlay.png", over)

    print(f"Terminé ! iters={iters} — fichiers dans output/")


if __name__ == "__main__":
    main()
