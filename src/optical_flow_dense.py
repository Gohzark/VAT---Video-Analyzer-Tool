import sys
import numpy as np
import cv2 as cv
import os
import argparse
from analystDense import *

def main(args):
    # 1. Vérification du fichier
    if not os.path.exists(args.video):
        print(f"Erreur : Le fichier vidéo '{args.video}' est introuvable.")
        sys.exit(1)

    cap = cv.VideoCapture(args.video)
    video_filename = os.path.basename(args.video)
    video_name = os.path.splitext(video_filename)[0]

    if not cap.isOpened():
        print(f"Erreur : Impossible d'ouvrir la source vidéo : {args.video}")
        sys.exit(1)

    # Initialisation
    ret, frame = cap.read()
    if not ret:
        print("Erreur : Impossible de lire la première image.")
        return

    prvs = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    hsv = np.zeros_like(frame)
    hsv[..., 1] = 255

    hauteur, largeur, _ = frame.shape
    tracker = AnalystDense(hauteur, largeur)
    fps = cap.get(cv.CAP_PROP_FPS)
        
    # Background Subtraction avec mixture de gaussiennes (MOG2)
    duree_warmup = 10
    backSub = cv.createBackgroundSubtractorMOG2(history=duree_warmup, varThreshold=30, detectShadows=False)
    #Monter le seuil de détection (varThreshold) pour éviter les petits mouvements parasites, et réduire l'historique pour être plus réactif aux changements rapides

    # Phase de Warm-up du background subtraction pour stabiliser le modèle avant de commencer à traiter les mouvements
    print("Initialisation du fond (merci de patienter)...")
    for i in range(duree_warmup):
        ret, frame = cap.read()
        if ret:
            backSub.apply(frame)
    print("Initialisation terminée, démarrage du traitement.")
    
    print("Traitement en cours... Appuyez sur 'Echap' pour quitter.")

    # Variables pour le lissage
    smooth_tx, smooth_ty = 0, 0
    alpha = 0.1 # Facteur de lissage (entre 0 et 1). 0.1 = lent/fluide, 0.5 = réactif
    
    # Boucle de traitement
    while True:
        ret, frame = cap.read()
        if not ret: break 
        
        fg_mask = backSub.apply(frame)
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
        
        # Trasnlation de l'image pour centrer le mouvement
        frame_centered = cv.warpAffine(frame, M_transform, (w, h))
        frame_gray = cv.cvtColor(frame_centered, cv.COLOR_BGR2GRAY)
        
        # Calcul du flux optique avec Farneback
        flow = cv.calcOpticalFlowFarneback(prvs, frame_gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
        
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
        
        prvs = frame_gray
        
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
    parser = argparse.ArgumentParser(description='Lucas-Kanade Optical Flow')
    parser.add_argument('video', type=str, help='chemin vers le fichier vidéo')
    args = parser.parse_args()
    
    main(args)