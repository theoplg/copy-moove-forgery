import numpy as np
import cv2 as cv

def noise(im,br):
    """ Cette fonction ajoute un bruit blanc gaussier d'ecart type br
       a l'image im et renvoie le resultat"""
    imt=np.float32(im.copy())
    sh=imt.shape
    bruit=br*np.random.randn(*sh)
    imt=imt+bruit
    return imt

im1 = cv.imread("images/lena_color.tiff", cv.IMREAD_COLOR)
im2 = noise(im1, 5)
cv.imshow("im1", im1.astype(np.uint8))
cv.imshow("im2", im2.astype(np.uint8))
cv.waitKey(0)
cv.destroyAllWindows()
 
