import numpy as np
import cv2 as cv

def add_impulse_noise(img, ratio):
    """
    Ajoute un bruit impulsionnel (Poivre et Sel) à l'image.
    
    Args:
        img (numpy array): L'image source chargée.
        ratio (float): Pourcentage de l'image à bruiter (ex: 0.05 pour 5%).
                       La moitié sera du poivre (noir), l'autre du sel (blanc).
    """
    # On travaille sur une copie pour ne pas modifier l'original
    out = np.copy(img)
    
    # On récupère les dimensions (Hauteur, Largeur)
    # On ignore le nombre de canaux pour le masque aléatoire
    h, w = out.shape[:2]
    
    # On génère une matrice de nombres aléatoires entre 0 et 1
    rng = np.random.rand(h, w)
    
    # --- SEL (Blanc / 255) ---
    # Si le nombre aléatoire est inférieur à la moitié du ratio
    # Exemple: pour ratio 0.05, les valeurs < 0.025 deviennent blanches
    out[rng < ratio / 2] = 255

    # --- POIVRE (Noir / 0) ---
    # Si le nombre aléatoire est supérieur à (1 - moitié du ratio)
    # Exemple: pour ratio 0.05, les valeurs > 0.975 deviennent noires
    out[rng > 1 - (ratio / 2)] = 0
    
    return out

# --- EXEMPLE D'UTILISATION ---

input_filename = "images/martinique2.jpeg" # Mets ton image ici

# 1. Chargement
img_source = cv.imread(input_filename, cv.IMREAD_COLOR)

if img_source is None:
    print("Erreur de chargement de l'image.")
else:
    # 2. Ajout du bruit (Ici 5% de l'image sera corrompue)
    img_impulse = add_impulse_noise(img_source, ratio=0.05)

    # 3. Affichage
    cv.imshow("Image Originale", img_source)
    cv.imshow("Image Poivre et Sel (5%)", img_impulse)
    
    cv.waitKey(0)
    cv.destroyAllWindows()
    
    # Sauvegarde si besoin
    # cv.imwrite("images/martinique_poivre_sel.jpg", img_impulse)