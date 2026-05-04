import sys
import cv2 as cv
import os
import argparse
from tracker.analyzerFourier import AnalyzerFourier
from tracker.analyzerStartStop import AnalyzerStartStop
import algorithm.optical_flow_dense as optical_flow_dense
import algorithm.optical_flow_sparse as optical_flow_sparse

from utils.enums import Algorithm, Mask, Tracker

def openVideo(video_path):
    if not os.path.exists(video_path):
        print(f"Erreur : Le fichier vidéo '{video_path}' est introuvable.")
        sys.exit(1)

    cap = cv.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Erreur : Impossible d'ouvrir la source vidéo : {video_path}")
        sys.exit(1)
    
    return cap

def getVideoName(video_path):
    video_filename = os.path.basename(video_path)
    video_name = os.path.splitext(video_filename)[0]
    return video_name

def createMask(cap, mask_type):
    mask = None
    match mask_type:
        case Mask.MO2:
            print("Masque MO2 (Mixture de gaussiennes) sélectionné")
            #Monter le seuil de détection (varThreshold) pour éviter les petits mouvements parasites, et réduire l'historique pour être plus réactif aux changements rapides
            warmup_duration = 10
            background_threshold = 30
            mask = cv.createBackgroundSubtractorMOG2(history=warmup_duration, varThreshold=background_threshold, detectShadows=False)
            # Phase de Warm-up du background subtraction pour stabiliser le modèle avant de commencer à traiter les mouvements
            print("Initialisation du fond (merci de patienter)...")
            for i in range(warmup_duration):
                ret, frame = cap.read()
                if ret:
                    mask.apply(frame)
            print("Initialisation terminée, démarrage du traitement.")
            cap.set(cv.CAP_PROP_POS_FRAMES, 0)
        case Mask.No:
            print("Aucun masque de mouvement sélectionné, le flux optique sera calculé sur toute l'image.")
    return mask

def useAlgorithm(cap, algorithm, video_name, mask, tracker):
    match algorithm:
        case Algorithm.LK:
            print("Algorithme Lucas-Kanade (sparse) sélectionné")
            optical_flow_sparse.run_sparse(cap, video_name, mask, tracker)
        case Algorithm.FARNEBACK:
            print("Algorithme Farneback (dense) sélectionné")
            optical_flow_dense.run_dense(cap, video_name, mask, tracker)

def initTracker(tracker_type, height, width):
    match tracker_type:
        case Tracker.Fourier:
            print("Tracker Fourier sélectionné")
            return AnalyzerFourier(height, width)
        case Tracker.Sparse:
            print("Tracker StartAndStop sélectionné")
            return AnalyzerStartStop(height, width)
        
def main(args):
    cap = openVideo(args.video)
    video_name = getVideoName(args.video)
    mask = createMask(cap, args.mask)
    tracker = initTracker(args.tracker, cap.get(cv.CAP_PROP_FRAME_HEIGHT), cap.get(cv.CAP_PROP_FRAME_WIDTH))
    useAlgorithm(cap, args.algorithm, video_name, mask, tracker)

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
        'tracker',
        type=Tracker,
        choices=list(Tracker)
    )
    args = parser.parse_args()
    main(args)