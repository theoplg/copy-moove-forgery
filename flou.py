import cv2 as cv
import numpy as np

def add_blur(img, ksize_val):
    """
    Applique un flou gaussien à l'image.
    
    Args:
        img: L'image source.
        ksize_val (int): La force du flou. Doit être un nombre IMPAIR (1, 3, 5, 7, 9...).
                         Plus il est grand, plus c'est flou.
    """
    # Vérification de sécurité : le noyau doit être impair
    if ksize_val % 2 == 0:
        ksize_val += 1
        
    # (ksize_val, ksize_val) est la taille du noyau (kernel)
    # 0 signifie que l'écart-type est calculé automatiquement
    blurred = cv.GaussianBlur(img, (ksize_val, ksize_val), 0)
    
    return blurred

# --- EXEMPLE D'UTILISATION ---

input_filename = "images/eleph.jpeg" 

# 1. Chargement
img_source = cv.imread(input_filename, cv.IMREAD_COLOR)

if img_source is None:
    print("Erreur chargement")
else:
    # 2. Ajout du flou
    # ksize=3 : flou très léger
    # ksize=5 : flou moyen
    # ksize=15 : très flou (l'algo va probablement échouer)
    img_floue = add_blur(img_source, ksize_val=20)

    # 3. Affichage
    cv.imshow("Originale", img_source)
    cv.imshow("Flou Gaussien (k=5)", img_floue)
    
    cv.waitKey(0)
    cv.destroyAllWindows()