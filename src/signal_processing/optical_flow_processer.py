import os
from matplotlib import pyplot as plt
import numpy as np
import cv2 as cv
from signal_processing.analyzer import Analyzer
from utils.enums import Mask, Centering, Algorithm, Analyze
import optical_flow_estimation.optical_flow_Farneback as optical_flow_Farneback
import optical_flow_estimation.optical_flow_LK as optical_flow_LK
from optical_flow_estimation.megaflow.run_megaflow import run_megaflow


def openVideo(video_path):
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Erreur : Le fichier vidéo '{video_path}' est introuvable.")

    cap = cv.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Erreur : Impossible d'ouvrir la source vidéo : {video_path}")
    
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
        case Mask.NoMask:
            print("Aucun masque de mouvement sélectionné, le flux optique sera calculé sur toute l'image.")
    return mask

def getOpticalFlow(chemin_video, algorithm, mask_name, centering, callback_progress=None, callback_image=None):
    # 1. Si c'est Megaflow, pas besoin d'ouvrir de flux vidéo local complet ici
    if algorithm == Algorithm.Megaflow or getattr(algorithm, 'name', None) == "Megaflow":
        print("Algorithme Megaflow (dense) sélectionné")
        cap_info = openVideo(chemin_video)
        fps = cap_info.get(cv.CAP_PROP_FPS)
        cap_info.release()
        
        run_megaflow(chemin_video, centering)
        return None, fps

    # 2. Pour les algos locaux (LK et Farneback) : UN SEUL flux ouvert du début à la fin
    cap = openVideo(chemin_video)
    fps = cap.get(cv.CAP_PROP_FPS)

    mask = None
    if mask_name != Mask.NoMask:
        # On passe le VRAI flux de travail au créateur de masque
        mask = createMask(cap, mask_name)

    # 3. Exécution de l'algorithme local
    # Grâce à notre modification, LK génère maintenant le même format 4D que Farneback !
    try:
        if algorithm == Algorithm.LucasKanade or getattr(algorithm, 'name', None) == "LucasKanade" or getattr(algorithm, 'value', None) == "LK":
            print("Algorithme Lucas-Kanade (sparse -> dense format) sélectionné")
            optical_flow = optical_flow_LK.run_LK(cap, mask, centering, callback_progress, callback_image)
            
        elif algorithm == Algorithm.Farneback or getattr(algorithm, 'name', None) == "Farneback":
            print("Algorithme Farneback (dense) sélectionné")
            optical_flow = optical_flow_Farneback.run_Farneback(cap, mask, centering, callback_progress, callback_image)
            
        else:
            raise ValueError(f"Algorithme non reconnu : {algorithm!r} (type: {type(algorithm).__name__})")
            
    finally:
        # On s'assure que le fichier vidéo soit TOUJOURS libéré, même si le calcul crash au milieu
        cap.release()

    # 4. Validation du résultat
    if optical_flow is None or (hasattr(optical_flow, '__len__') and len(optical_flow) == 0):
        raise RuntimeError(f"Le calcul du flux optique n'a produit aucun résultat pour l'algorithme {getattr(algorithm, 'name', 'Inconnu')}.")

    return optical_flow, fps

def initAnalyse(video_name, algorithm, mask, centering, analyze):
    match analyze:
        case Analyze.FastFourierTransformation:
            print("Analyse par FFT sélectionnée")
        case Analyze.StartStop:
            print("Analyse StartStop sélectionnée")
        case Analyze.Sliding:
            print("Analyse par décalage du signal sélectionnée")
    return Analyzer(video_name, algorithm, mask, centering, analyze)

def _plotEvolution(plot_dir: str, magnitudes: list, fps: float) -> None:
    fig, ax = plt.subplots(figsize=(10, 4))
    t = np.linspace(0, len(magnitudes) / fps, len(magnitudes))
    ax.plot(t, magnitudes, color='blue', label='Magnitude du mouvement')
    ax.set_title("Évolution du mouvement")
    ax.set_xlabel("Temps (s)")
    ax.set_ylabel("Magnitude")
    ax.legend()
    plt.tight_layout()
    plt.savefig(plot_dir + "/plot_evolution.png", dpi=150)
    plt.close()
    
def detectMovements(analyzer, fps: float) -> None:
    magnitudes = np.load(os.path.join("outputs", analyzer.video_name, analyzer.algorithm.value, analyzer.mask.value, analyzer.centering.value, "magnitudes.npy"))
    _plotEvolution(analyzer._plot_dir(), magnitudes, fps)
    match analyzer.analyze:
        case Analyze.FastFourierTransformation:
            analyzer._detectFFT(magnitudes, fps)
        case Analyze.Sliding:
            analyzer._detectBySliding(magnitudes, fps)
        case Analyze.StartStop:
            analyzer._detectStartStop(magnitudes, fps)

def flowToMagnitudes(flow_path: str) -> None:
    output_path = os.path.join(os.path.dirname(flow_path), "magnitudes.npy")

    flows = np.load(flow_path)
    result = []
    for flow in flows:
        u = flow[..., 0]
        v = flow[..., 1]
        mag = np.sqrt(u**2 + v**2)
        score = float(np.sum(mag)) / (mag.shape[0] * mag.shape[1])
        result.append((score))
    
    return np.array(result)
    
