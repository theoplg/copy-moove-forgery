import numpy as np
import cv2 as cv

def add_noise_and_save(input_path, output_path, br):
    """
    Charge une image, ajoute un bruit blanc gaussien d'écart type 'br',
    sauvegarde le résultat et renvoie l'image bruitée.
    
    Args:
        input_path (str): Chemin de l'image source.
        output_path (str): Chemin pour sauvegarder l'image bruitée.
        br (float): Écart type du bruit (l'intensité).
    """
    # 1. Chargement de l'image
    im = cv.imread(input_path, cv.IMREAD_COLOR)
    if im is None:
        print(f"Erreur: Impossible de lire l'image {input_path}")
        return None

    # 2. Conversion en float pour les calculs mathématiques
    imt = np.float32(im.copy())
    sh = imt.shape

    # 3. Génération du bruit gaussien
    # np.random.randn génère des valeurs selon une loi normale centrée en 0
    bruit = br * np.random.randn(*sh)

    # 4. Ajout du bruit
    imt = imt + bruit

    # 5. IMPORTANT : On limite les valeurs entre 0 et 255
    # Sinon 255 + 10 devient 265, ce qui créerait des bugs visuels
    imt = np.clip(imt, 0, 255)

    # 6. Re-conversion en entiers 8 bits (0-255) pour l'enregistrement
    res_img = imt.astype(np.uint8)

    # 7. Enregistrement
    cv.imwrite(output_path, res_img)
    print(f"Image bruitée enregistrée sous : {output_path}")

    return res_img

def jpeg_compression(img, quality):
    """
    Simule une compression JPEG.
    quality : de 0 (très compressé/moche) à 100 (parfait)
    """
    # On encode l'image en mémoire tampon (buffer) au format JPEG
    encode_param = [int(cv.IMWRITE_JPEG_QUALITY), quality]
    result, encimg = cv.imencode('.jpg', img, encode_param)
    
    # On décode immédiatement pour récupérer l'image avec les défauts
    decimg = cv.imdecode(encimg, 1)
    return decimg

# --- Test ---


# Test avec qualité 30 (assez destructeur, typique du web)


# Ensuite, lancez votre detection sur 'im_jpeg' !
# --- Exemple d'utilisation ---

# --- Exemple d'utilisation ---

input_filename = "images/eleph.jpeg" 

# 1. D'ABORD, on charge l'image en mémoire (transforme le fichier en matrice numpy)
img_source = cv.imread(input_filename, cv.IMREAD_COLOR)

# Vérification de sécurité (toujours utile)
if img_source is None:
    print(f"Erreur : Impossible de trouver ou lire l'image '{input_filename}'")
    exit()

# 2. Ensuite, on passe l'image chargée à la fonction (pas le nom du fichier)
im_jpeg = jpeg_compression(img_source, quality=30) # Qualité 10 pour bien voir les dégâts

# 3. Affichage
# Attention : cv.imshow attend aussi une image (img_source), pas le nom du fichier
cv.imshow("Image Originale", img_source)
cv.imshow("Image Compressee (Q=10)", im_jpeg)

cv.waitKey(0)
cv.destroyAllWindows()

# Si vous voulez sauvegarder le résultat compressé :
cv.imwrite("images/eleph_compressee.jpg", im_jpeg)

# if img_bruitee is not None:
    # Affichage pour vérifier
#     cv.imshow("Image Bruitee", img_bruitee)
#     cv.waitKey(0)
#     cv.destroyAllWindows()

