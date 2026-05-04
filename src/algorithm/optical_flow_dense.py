import sys
import numpy as np
import cv2 as cv
import os
import argparse
from analyst.analystFourier import *
from enums import Algorithm, Mask

def main(cap, video_name, mask):

    # Initialisation
    ret, frame = cap.read()
    if not ret:
        print("Erreur : Impossible de lire la première image.")
        return

    previous_image = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    hsv = np.zeros_like(frame)
    hsv[..., 1] = 255

    hauteur, largeur, _ = frame.shape
    tracker = AnalystDense(hauteur, largeur)
    fps = cap.get(cv.CAP_PROP_FPS)
        

    # Variables pour le lissage
    smooth_tx, smooth_ty = 0, 0
    alpha = 0.1 # Facteur de lissage (entre 0 et 1). 0.1 = lent/fluide, 0.5 = réactif
    
    # Boucle de traitement
    while True:
        ret, frame = cap.read()
        if not ret: break 
        if mask is not None:
            fg_mask = mask.apply(frame)
        else:
            fg_mask = np.ones(frame.shape[:2], dtype=np.uint8) * 255
        kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (5,5))
        fg_mask = cv.morphologyEx(fg_mask, cv.MORPH_OPEN, kernel)
        h, w = frame.shape[:2]

        # On calcule la surface totale de l'image
        total_area = h * w
  
        seuil_pourcentage = 0.0001
        
        contours, _ = cv.findContours(fg_mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
        
        # On cherche le plus gros contour
        if contours:
            c = max(contours, key=cv.contourArea)
            if cv.contourArea(c) > (total_area * seuil_pourcentage):
                x, y, w_obj, h_obj = cv.boundingRect(c)
                cx = x + w_obj // 2
                cy = y + h_obj // 2
                
                target_tx = (w // 2) - cx
                target_ty = (h // 2) - cy
                
                smooth_tx = (1 - alpha) * smooth_tx + alpha * target_tx
                smooth_ty = (1 - alpha) * smooth_ty + alpha * target_ty
                                
        M_transform = np.float32([[1, 0, smooth_tx], [0, 1, smooth_ty]])
        
        # Translation de l'image pour centrer le mouvement
        frame_centered = cv.warpAffine(frame, M_transform, (w, h))
        frame_gray = cv.cvtColor(frame_centered, cv.COLOR_BGR2GRAY)
        
        # Calcul du flux optique avec Farneback
        flow = cv.calcOpticalFlowFarneback(previous_image, frame_gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
        
        # On masque le flux optique pour ne garder que les zones de mouvement détectées par le background subtraction
        mask_centered = cv.warpAffine(fg_mask, M_transform, (w, h))
        mask_bool = mask_centered > 0 
        flow[~mask_bool] = 0
        
        # Calcul des magnitudes et angles du flux optique
        mag, ang = cv.cartToPolar(flow[..., 0], flow[..., 1], angleInDegrees=True)
        tracker.update(mag, ang)
        
        hsv[..., 0] = ang * 180 / np.pi / 2
        hsv[..., 2] = cv.normalize(mag, None, 0, 255, cv.NORM_MINMAX)
        bgr = cv.cvtColor(hsv, cv.COLOR_HSV2BGR)
        overlay = cv.addWeighted(frame_centered, 1, bgr, 1, 0)
        
        cv.imshow('Frame Centered', overlay)
        
        previous_image = frame_gray
        
        # Sortie clavier (Echap)
        if cv.waitKey(1) == 27: 
            print("Arrêt par l'utilisateur.")
            break

    # Nettoyage et fin de traitement
    cap.release()
    cv.destroyAllWindows()
    
    # Calcul final
    print("Calcul des mouvements finaux...")
    tracker.detectMovements(fps, video_name)
    print("Terminé.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Farneback Dense Optical Flow')
    parser.add_argument('video', type=str, help='chemin vers le fichier vidéo')
    args = parser.parse_args()
    
    main(args)