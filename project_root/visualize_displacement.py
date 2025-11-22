import os
print("CWD =", os.getcwd())
from patchmatch import *
from utils import *


def main():
    # remplacer par les images souhaitées
    img1 = load_image("input/lena_modif.png")
    img2 = load_image("input/lena_modif.png")

    r = 4
    offsets = init_off(img1)
    offsets, _ = patchmatch(img1, img2, r, offsets, nb_iters=5)

    disp = displacement_map(offsets)
    show(disp, "Displacement Map")


if __name__ == "__main__":
    main()
