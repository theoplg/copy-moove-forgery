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


def filter_by_global_frequency(offsets, min_count=500):
    H, W = offsets.shape[:2]
    # on aplatit le tableau pour avoir une liste de vecteurs (N, 2)
    vecs = offsets.reshape(-1, 2)
    
    # pour compter les occurrences de chaque vecteur (dx, dy)
    unique_vecs, inverse_indices, counts = np.unique(vecs, axis=0, return_counts=True, return_inverse=True)
    
    # on crée une carte où chaque pixel contient le nombre de fois que son vecteur déplacement apparaît dans toute l'image
    freq_map = counts[inverse_indices].reshape(H, W)
    
    # on ne garde que les pixels dont le vecteur est très fréquent
    mask_freq = (freq_map > min_count)
    
    print(f"Vecteurs uniques trouvés : {len(unique_vecs)}")
    print(f"Pixels conservés par fréquence globale : {np.count_nonzero(mask_freq)}")
    
    return mask_freq


def detection_mask_from_offsets(offsets, dist_map, patch_r,
                                rho_m=4,    
                                rho_e=6,    
                                tau_error=200,
                                tau_disp=10,     
                                min_size=500,
                                max_rmse=15,
                                min_global_count=800):

    # filtrage médian
    offsets_f = median_filter_offsets(offsets, radius=rho_m)

    # filtre de fréquence globale
    freq_mask = filter_by_global_frequency(offsets_f, min_count=min_global_count)

    # error map
    err = compute_error_map_translation(offsets_f, rho_e=rho_e)
    consistency_mask = (err < tau_error)

    # RMSE
    patch_area = ((2 * patch_r + 1) ** 2) * 3
    rmse_map = np.sqrt(dist_map / patch_area)
    quality_mask = (rmse_map < max_rmse)
    
    # minimum deplacement
    mag = np.sqrt(offsets_f[..., 0]**2 + offsets_f[..., 1]**2)
    displacement_mask = (mag > tau_disp)

    mask = freq_mask & consistency_mask & displacement_mask & quality_mask

    mask = mask.astype(np.uint8) * 255
    
    # nettoyage des petites régions
    mask = size_filter(mask, min_size=min_size)

    return mask, err
