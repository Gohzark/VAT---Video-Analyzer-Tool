import sys
import cv2 as cv
import os
import argparse
from tracker.analyzer import Analyzer
import algorithm.optical_flow_dense as optical_flow_dense
import algorithm.optical_flow_sparse as optical_flow_sparse
from utils.enums import Algorithm, Mask, Analyze

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
            background_threshold = 20
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
        case Analyze.FFT:
            print("Analyse par FFT sélectionnée")
        case Analyze.SS:
            print("Analyse StartStop sélectionnée")
        case Analyze.Sliding:
            print("Analyse par décalage du signal sélectionnée")
    return Analyzer(video_name, height, width, algorithm, mask, analyze, centering)
        
def useAlgorithm(cap, algorithm, mask, tracker, centering):
    match algorithm:
        case Algorithm.LK:
            print("Algorithme Lucas-Kanade (sparse) sélectionné")
            optical_flow_sparse.run_sparse(cap, mask, tracker)
        case Algorithm.Farneback:
            print("Algorithme Farneback (dense) sélectionné")
            optical_flow_dense.run_dense(cap, mask, tracker, centering)
            
def main(args):
    cap = openVideo("resources/" + args.video)
    mask = createMask(cap, args.mask)
    analyse = initAnalyse(args.analyse, args.video, cap.get(cv.CAP_PROP_FRAME_HEIGHT), cap.get(cv.CAP_PROP_FRAME_WIDTH), args.algorithm, args.mask, args.centering)
    useAlgorithm(cap, args.algorithm, mask, analyse, args.centering)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Video Analyzer Tool')
    parser.add_argument('video', type=str)
    parser.add_argument(
        'algorithm',
        type=Algorithm,
        choices=list(Algorithm)
    )
    parser.add_argument(
        'mask',
        type=Mask,
        choices=list(Mask)
    )
    parser.add_argument(
        'analyse',
        type=Analyze,
        choices=list(Analyze)
    )
    parser.add_argument(
        '--centering',
        action='store_true',
        default=False,
    )
    args = parser.parse_args()
    main(args)
    
#exemple : /usr/bin/python3 /home/tino/Bureau/stage/Video_Analyzer_Tool/src/main.py ./resources/pu.mp4 Farneback MOG2 StartStop



