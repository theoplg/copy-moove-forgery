import cv2 as cv
import numpy as np
import test  # Votre module test.py

def nothing(x):
    pass

def interactive_tuner(image_path, scale=1):
    # ... (Début identique : chargement, redimensionnement, PatchMatch) ...
    print(f"--- Préparation de {image_path} ---")
    img = cv.imread(image_path)
    if img is None:
        print("Erreur: Image introuvable.")
        return
    
    img = cv.resize(img, None, fx=scale, fy=scale)
    print("Calcul de PatchMatch en cours...")
    
    offsets_init = test.init_off(img)
    offsets, dist_map = test.patchmatch(img, img, 4, offsets_init, nb_iters=5)
    
    print("PatchMatch terminé ! Ouverture de la fenêtre...")

    window_name = "Tuner"
    cv.namedWindow(window_name)
    
    # --- C'EST ICI QU'ON CHANGE LES VALEURS PAR DÉFAUT ---
    # Syntaxe : createTrackbar("Nom", "Fenêtre", VALEUR_INITIALE, VALEUR_MAX, callback)
    
    # Tau Error : 800 (plus tolérant car l'inpainting est approximatif)
    cv.createTrackbar("Tau Error", window_name, 800, 2000, nothing)
    
    # Tau Disp : 5 (accepte les copies locales très proches)
    cv.createTrackbar("Tau Disp", window_name, 5, 100, nothing)
    
    # Min Size : 50 (détecte les petites miettes/poussières)
    cv.createTrackbar("Min Size", window_name, 50, 5000, nothing)
    
    # Global Freq : 20 (détecte les vecteurs rares/chaotiques)
    cv.createTrackbar("Global Freq", window_name, 20, 2000, nothing)
    
    # Max RMSE : 25 (tolérance couleur un peu plus large)
    cv.createTrackbar("Max RMSE", window_name, 25, 100, nothing)
    
    # Median Rad : 1 (préserve le bruit fin, ne lisse pas)
    cv.createTrackbar("Median Rad", window_name, 1, 10, nothing)

    last_params = None
    vis = img.copy()
    mask_vis = np.zeros(img.shape[:2], dtype=np.uint8)

    while True:
        # ... (Le reste de la boucle while reste IDENTIQUE au code précédent) ...
        # ... (Récupération des trackbars, appel de detection_mask_from_offsets, affichage, etc.) ...
        
        # Copiez-collez simplement le contenu de la boucle `while` de ma réponse précédente ici.
        # Le changement important est juste au-dessus dans les `createTrackbar`.

        # Pour rappel de la boucle :
        t_err = cv.getTrackbarPos("Tau Error", window_name)
        t_disp = cv.getTrackbarPos("Tau Disp", window_name)
        min_sz = cv.getTrackbarPos("Min Size", window_name)
        min_glob = cv.getTrackbarPos("Global Freq", window_name)
        max_rmse = cv.getTrackbarPos("Max RMSE", window_name)
        rho_m = cv.getTrackbarPos("Median Rad", window_name)
        if rho_m < 1: rho_m = 1

        current_params = (t_err, t_disp, min_sz, min_glob, max_rmse, rho_m)

        if current_params != last_params:
            print(f"Paramètres : {current_params}")
            mask, _ = test.detection_mask_from_offsets(
                offsets, dist_map, 
                patch_r=4, 
                rho_m=rho_m, rho_e=rho_m+2, 
                tau_error=t_err, tau_disp=t_disp, 
                min_size=min_sz, max_rmse=max_rmse, 
                min_global_count=min_glob
            )
            
            mask = cv.dilate(mask, np.ones((3,3), np.uint8), iterations=1) # Dilatation légère
            mask_vis = mask
            
            overlay = img.copy()
            overlay[mask > 0] = [0, 0, 255]
            vis = cv.addWeighted(img, 0.7, overlay, 0.3, 0)
            
            cv.imshow(window_name, vis)
            cv.imshow("Masque Binaire", mask_vis)
            last_params = current_params

        if (cv.waitKey(10) & 0xFF) in [ord('q'), 27]:
            break

    cv.destroyAllWindows()

if __name__ == "__main__":
    # ATTENTION : Si votre image est très grande (ex: 4000px), mettez scale=0.2
    # Sinon le calcul sera trop lent pour être interactif.
    interactive_tuner("images/inpaint_mart.png", scale=1)