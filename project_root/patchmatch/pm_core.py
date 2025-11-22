import cv2 as cv
import numpy as np
import math
import matplotlib as plt


def copy_im(img, r):
    return cv.copyMakeBorder(img, r, r, r, r, borderType=cv.BORDER_CONSTANT, value=0) #on crée un copie de l'image de base pour éviter les problèmes de bord

def patch(copy_img, pos, r):
    # on utilise copy image pour ne pas avoir de problème de bord
    cx = pos[0]
    cy = pos[1]  # pos c'est les coordonnées du centre du patch
    patch = copy_img[cx - r:cx + r + 1, cy - r:cy + r +1] # on fait juste un bloc de pixel de la taille de notre choix
    return patch


def init_off(img):
    H, W = img.shape[:2]  
    offsets = np.zeros((H, W, 2), dtype=int)
    for i in range(H):
        for j in range(W): 
            offsets[i, j] = (np.random.randint(0, W) - j, np.random.randint(0, H) - i) # position aléatoire dans l'image B
    return offsets


def is_forbidden_offset(dx, dy, R2):
    # dx et dy sont les composantes de la translation
    dist_carre = dx**2 + dy**2
    return dist_carre <= R2  # True si l'offset est dans le disque interdit (distance <= R)


def distance(patch1, patch2):
    h, w, _ = patch1.shape
    diff = patch1.astype(np.float32) - patch2.astype(np.float32)
    d = np.sum(np.abs(diff)) / (h*w*3)
    return d


def random_search(im1, im2, r, offsets, dist, alpha=0.5):
    h, w = im1.shape[:2]
    im2_h, im2_w = im2.shape[:2]

    for i in range(h):
        for j in range(w):
            best_dx, best_dy = offsets[i, j]
            best_dist = dist[i, j]

            R = max(im2_h, im2_w)
            while R >= 1:
                rand_dx = int(best_dx + np.random.uniform(-R, R))
                rand_dy = int(best_dy + np.random.uniform(-R, R))

                cand_x = j + rand_dx
                cand_y = i + rand_dy

                if 0 <= cand_x < im2_w and 0 <= cand_y < im2_h:
                    p1 = patch(im1, (i + r, j + r), r)
                    p2 = patch(im2, (cand_y + r, cand_x + r), r)
                    if p1.shape == p2.shape:
                        d = distance(p1, p2)
                        if d < best_dist:
                            best_dist = d
                            best_dx, best_dy = rand_dx, rand_dy

                R = int(R * alpha)

            offsets[i, j] = [best_dx, best_dy]
            dist[i, j] = best_dist

    return offsets, dist


def propag(im1, im2, r, offsets, direction='forward', R2=16):
    H, W = im1.shape[:2]  
    dist = np.full((H, W), np.inf, dtype=np.float32)
    copy1 = copy_im(im1, r)
    copy2 = copy_im(im2, r)
    if direction == 'forward':
        for i in range(H):
            for j in range(W):
                dx, dy = offsets[i, j]
                
                p1 = patch(copy1, (i + r, j + r), r)

                # verif offset interdit
                if not is_forbidden_offset(dx, dy, R2):
                    p2 = patch(copy2, (i + dy + r, j + dx + r), r)
                    if p1.shape[:2] == (2*r + 1, 2*r + 1) and p2.shape[:2] == (2*r + 1, 2*r + 1):
                        d_cur = distance(p1, p2)
                        dist[i, j] = d_cur

                # voisin du haut
                if i > 0 and patch(copy2, (i + offsets[i-1, j][1] + r, j + offsets[i-1, j][0] + r), r).shape[:2] == (2*r+1, 2*r+1):
                    cand_dx, cand_dy = offsets[i-1, j]
                    if not is_forbidden_offset(cand_dx, cand_dy, R2):
                        d1 = distance(patch(copy1, (i + r, j + r), r), patch(copy2, (i + offsets[i-1, j][1] + r, j + offsets[i-1, j][0] + r), r))
                        if d1 < dist[i, j]:
                            offsets[i, j] = offsets[i-1, j]
                            dist[i, j] = d1

                # voisin de gauche
                if j > 0 and patch(copy2, (i + offsets[i, j-1][1] + r, j + offsets[i, j-1][0] + r), r).shape[:2] == (2*r+1, 2*r+1):
                    cand_dx, cand_dy = offsets[i, j-1]
                    if not is_forbidden_offset(cand_dx, cand_dy, R2):    
                        d2 = distance(patch(copy1, (i + r, j + r), r), patch(copy2, (i + offsets[i, j-1][1] + r, j + offsets[i, j-1][0] + r), r))
                        if d2 < dist[i, j]:
                            offsets[i, j] = offsets[i, j-1]
                            dist[i, j] = d2

    else:
        for i in range(H-1, -1, -1):
            for j in range(W-1, -1, -1):
                dx, dy = offsets[i, j]
                
                p1 = patch(copy1, (i + r, j + r), r)

                # verif offset interdit
                if not is_forbidden_offset(dx, dy, R2):
                    p2 = patch(copy2, (i + dy + r, j + dx + r), r)
                    if p1.shape[:2] == (2*r + 1, 2*r + 1) and p2.shape[:2] == (2*r + 1, 2*r + 1):
                        d_cur = distance(p1, p2)
                        dist[i, j] = d_cur

                # voisin du bas
                if i < H-1 and patch(copy2, (i + offsets[i+1, j][1] + r, j + offsets[i+1, j][0] + r), r).shape[:2] == (2*r+1, 2*r+1):
                    cand_dx, cand_dy = offsets[i+1, j]
                    if not is_forbidden_offset(cand_dx, cand_dy, R2):
                        d1 = distance(patch(copy1, (i + r, j + r), r), patch(copy2, (i + offsets[i+1, j][1] + r, j + offsets[i+1, j][0] + r), r))
                        if d1 < dist[i, j]:
                            offsets[i, j] = offsets[i+1, j]
                            dist[i, j] = d1

                # voisin de droite
                if j < W-1 and patch(copy2, (i + offsets[i, j+1][1] + r, j + offsets[i, j+1][0] + r), r).shape[:2] == (2*r+1, 2*r+1):
                    cand_dx, cand_dy = offsets[i, j+1]
                    if not is_forbidden_offset(cand_dx, cand_dy, R2):
                        d2 = distance(patch(copy1, (i + r, j + r), r), patch(copy2, (i + offsets[i, j+1][1] + r, j + offsets[i, j+1][0] + r), r))
                        if d2 < dist[i, j]:
                            offsets[i, j] = offsets[i, j+1]
                            dist[i, j] = d2

    return offsets, dist


def patchmatch(im1, im2, r, offsets, nb_iters = 5): # l'argument offset est le dictionnaire des offsets
    for _ in range(1, nb_iters+1):
        print("IT NUMBER: ", _)

        print("FORWARD PROPAG...")

        # on parcourt d'abord de haut en bas et de gauche à droi...uis on compare avec les voisins de gauche et les voisins du haut
        offsets, dist = propag(im1, im2, r, offsets, direction='forward')

        print("END FORWARD PROPAG")

        print("BACKWARD PROPAG...")
        
        # on parcourt ensuite de bas en haut et de droite, puis on compare avec les voisins de droite et du bas
        offsets, dist = propag(im1, im2, r, offsets, direction='backward')

        print("END BACKWARD PROPAG")

        print("RANDOM SEARCH...")

        offsets, dist = random_search(im1, im2, r, offsets, dist, alpha=0.5)

        print("END RANDOM SEARCH")

    return offsets, dist


# reconstruction de l'image 2 à partir de l'image 1 et des offsets
def remap(source, offsets, r):
    H, W = offsets.shape[:2]
    C = 1 if source.ndim == 2 else source.shape[2]

    out = np.zeros((H, W, C), np.float32) if C > 1 else np.zeros((H, W), np.float32)
    wgt = np.zeros((H, W), np.float32)

    for i in range(H):
        for j in range(W):
            dx, dy = offsets[i, j]
            cx_src = j + dx
            cy_src = i + dy

            for u in range(-r, r + 1):
                for v in range(-r, r + 1):
                    y  = i  + v
                    x  = j  + u
                    ys = cy_src + v
                    xs = cx_src + u

                    if 0 <= y < H and 0 <= x < W and 0 <= ys < H and 0 <= xs < W:
                        if C == 1:
                            out[y, x] += source[ys, xs]
                        else:
                            out[y, x, :] += source[ys, xs, :]
                        wgt[y, x] += 1.0

    # normalisation
    wgt = np.maximum(wgt, 1e-8)
    if C == 1:
        out /= wgt
    else:
        out /= wgt[:, :, None]

    out = np.clip(out, 0, 255).astype(np.uint8)
    return out
