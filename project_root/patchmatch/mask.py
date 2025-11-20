import cv2 as cv
import numpy as np

from .displacement import median_filter_offsets, compute_error_map_translation


def size_filter(mask, min_size=500):
    num_labels, labels, stats, _ = cv.connectedComponentsWithStats(
        mask.astype(np.uint8), connectivity=4
    )
    filtered = mask.copy()
    for lab in range(1, num_labels):
        area = stats[lab, cv.CC_STAT_AREA]
        if area < min_size:
            filtered[labels == lab] = 0
    return filtered


def detection_mask_from_offsets(
    offsets,
    rho_m=4,
    rho_e=6,
    tau_error=1e-3,
    tau_disp=5,
    min_size=1000,
):
    offsets_f = median_filter_offsets(offsets, radius=rho_m)
    error_map = compute_error_map_translation(offsets_f, rho_e=rho_e)

    if tau_error is None:
        finite = np.isfinite(error_map)
        if np.any(finite):
            tau_error = np.percentile(error_map[finite], 90.0)
        else:
            tau_error = 0.0

    dx = offsets_f[..., 0].astype(np.float32)
    dy = offsets_f[..., 1].astype(np.float32)
    mag = np.sqrt(dx * dx + dy * dy)

    # mask_err = (error_map >= tau_error).astype(np.uint8)
    # mask_disp = (disp_norm >= tau_disp).astype(np.uint8)

    # mask = mask_err * mask_disp
    # mask = size_filter(mask, min_size=min_size)
    # mask = (mask > 0).astype(np.uint8) * 255

    mask = (error_map < tau_error) & (mag > tau_disp)

    mask = mask.astype(np.uint8) * 255

    # 5) filtre de taille
    mask = size_filter(mask, min_size=min_size)

    return mask, error_map
