import numpy as np


def overlay_mask_on_image(img, mask, alpha=0.5):
    # s'assurer que mask est binaire {0,1}
    m = (mask > 0).astype(np.uint8)

    # créer un calque rouge
    overlay = np.zeros_like(img, dtype=np.uint8)

    # si l'image est BGR (OpenCV), on met rouge = (0,0,255)
    # si elle est RGB (matplotlib), c'est de toute façon proche visuellement
    overlay[..., 2] = 255  # canal rouge

    # appliquer alpha blending uniquement où m == 1
    img_float = img.astype(np.float32)
    overlay_float = overlay.astype(np.float32)

    img_float[m == 1] = (
        (1 - alpha) * img_float[m == 1] +
         alpha      * overlay_float[m == 1]
    )

    return img_float.astype(np.uint8)
