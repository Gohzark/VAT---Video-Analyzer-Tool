import os
import numpy as np
import argparse            
import sys
import cv2 as cv
from utils.enums import Mask, Centering, Algorithm, Analyze
from signal_processing.analyzer import Analyzer
import optical_flow_estimation.optical_flow_Farneback as optical_flow_Farneback
import optical_flow_estimation.optical_flow_LK as optical_flow_LK

def openVideo(video_path):
    if not os.path.exists(video_path):
        print(f"Erreur : Le fichier vidéo '{video_path}' est introuvable.")
        sys.exit(1)

    cap = cv.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Erreur : Impossible d'ouvrir la source vidéo : {video_path}")
        sys.exit(1)
    
    return cap

def createMask(cap, mask_type):
    mask = None
    match mask_type:
        case Mask.MOG2:
            print("Masque MOG2 (Mixture de gaussiennes) sélectionné")
            #Monter le seuil de détection (varThreshold) pour éviter les petits mouvements parasites, et réduire l'historique pour être plus réactif aux changements rapides
            warmup_duration = 30
            background_threshold = 40
            mask = cv.createBackgroundSubtractorMOG2(history=warmup_duration, varThreshold=background_threshold, detectShadows=False)
            # Phase de Warm-up du background subtraction pour stabiliser le modèle avant de commencer à traiter les mouvements
            print("Initialisation du fond (merci de patienter)...")
            for i in range(warmup_duration):
                ret, frame = cap.read()
                if ret:
                    mask.apply(frame)
            print("Initialisation terminée, démarrage du traitement.")
            cap.set(cv.CAP_PROP_POS_FRAMES, 0)
        case Mask.NoMask:
            print("Aucun masque de mouvement sélectionné, le flux optique sera calculé sur toute l'image.")
    return mask

def initAnalyse(analyze, video_name, height, width, algorithm, mask, centering):
    match analyze:
        case Analyze.FastFourierTransformation:
            print("Analyse par FFT sélectionnée")
        case Analyze.StartStop:
            print("Analyse StartStop sélectionnée")
        case Analyze.Sliding:
            print("Analyse par décalage du signal sélectionnée")
    return Analyze(video_name, height, width, algorithm, mask, analyze, centering)
        
def getSignal(cap, algorithm, mask, centering):
    match algorithm:
        case Algorithm.LucasKanade:
            print("Algorithme Lucas-Kanade (sparse) sélectionné")
            optical_flow_LK.run_LK(cap, mask, cap.get(cv.CAP_PROP_FRAME_WIDTH), cap.get(cv.CAP_PROP_FRAME_HEIGHT))
        case Algorithm.Farneback:
            print("Algorithme Farneback (dense) sélectionné")
            optical_flow_Farneback.run_Farneback(cap, mask, centering)
        case Algorithm.Megaflow:
            #nécessite d'avoir généré les flux optiques avec le notebook "megaflow.ipynb" avant de lancer l'analyse
            print("Algorithme Megaflow (dense) sélectionné")
                          
def flow_to_magnitudes(video: str, algo: str) -> None:
    flow_path = os.path.join("outputs", video, algo, "optical_flow.npy")
    output_path = os.path.join(os.path.dirname(flow_path), "magnitudes.npy")

    if os.path.exists(output_path):
        answer = input(f"[flow_to_magnitudes] {output_path} existe déjà. Écraser ? (o/n) : ").strip().lower()
        if answer != 'o':
            print("Fichier conservé, conversion annulée.")
            return

    flows = np.load(flow_path)
    result = []
    for flow in flows:
        u = flow[..., 0]
        v = flow[..., 1]
        mag = np.sqrt(u**2 + v**2)
        score = float(np.sum(mag)) / (mag.shape[0] * mag.shape[1])
        result.append((score))

    np.save(output_path, np.array(result))
    print(f"Magnitudes sauvegardées dans : {output_path}")
    