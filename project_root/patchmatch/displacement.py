import numpy as np
from scipy.ndimage import median_filter


def displacement_map(offsets):
    dx = offsets[..., 0].astype(np.float32)
    dy = offsets[..., 1].astype(np.float32)

    def norm01(a):
        amin = a.min()
        amax = a.max()
        if amax == amin:
            return np.zeros_like(a, dtype=np.float32)
        return (a - amin) / (amax - amin)

    dx_n = norm01(dx)
    dy_n = norm01(dy)
    mix = 1.0 - 0.5 * (dx_n + dy_n)

    H, W = dx.shape
    img = np.zeros((H, W, 3), dtype=np.float32)
    img[..., 0] = mix
    img[..., 1] = dy_n
    img[..., 2] = dx_n

    return (img * 255.0).clip(0, 255).astype(np.uint8)


def median_filter_offsets(offsets, radius=4):
    k = 2 * radius + 1
    dx = offsets[..., 0]
    dy = offsets[..., 1]

    dx_f = median_filter(dx, size=k)
    dy_f = median_filter(dy, size=k)

    return np.stack((dx_f, dy_f), axis=-1)


def compute_error_map_translation(offsets, rho_e=6):
    H, W, _ = offsets.shape
    dx = offsets[..., 0].astype(np.float32)
    dy = offsets[..., 1].astype(np.float32)

    error = np.full((H, W), np.nan, dtype=np.float32)

    for i in range(rho_e, H - rho_e):
        for j in range(rho_e, W - rho_e):
            dx_win = dx[i - rho_e:i + rho_e + 1, j - rho_e:j + rho_e + 1]
            dy_win = dy[i - rho_e:i + rho_e + 1, j - rho_e:j + rho_e + 1]

            mdx = dx_win.mean()
            mdy = dy_win.mean()

            dev2 = (dx_win - mdx) ** 2 + (dy_win - mdy) ** 2
            error[i, j] = dev2.mean()

    # pour les bords (où on n'a pas de fenêtre complète), on met une grosse erreur
    finite = np.isfinite(error)
    if np.any(finite):
        max_e = np.nanmax(error[finite])
        error = np.nan_to_num(error, nan=max_e)
    else:
        error[:] = 0.0

    return error
