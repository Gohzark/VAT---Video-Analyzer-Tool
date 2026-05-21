import cv2 as cv
import argparse
from signal_processing.analyzer import Analyzer
import optical_flow_estimation.optical_flow_Farneback as optical_flow_Farneback
import optical_flow_estimation.optical_flow_LK as optical_flow_LK
from signal_processing.optical_flow_processing import flow_to_magnitudes
from utils.enums import Algorithm, Mask, Analyze, Centering
from utils.helpers import openVideo, createMask


def initAnalyse(analyze, video_name, height, width, algorithm, mask, centering):
    match analyze:
        case Analyze.FastFourierTransformation:
            print("Analyse par FFT sélectionnée")
        case Analyze.StartStop:
            print("Analyse StartStop sélectionnée")
        case Analyze.Sliding:
            print("Analyse par décalage du signal sélectionnée")
    return Analyzer(video_name, height, width, algorithm, mask, analyze, centering)
        
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
            
def main(args):
    cap = openVideo("resources/" + args.video)
    mask = createMask(cap, args.mask)
    getSignal(cap, args.algorithm, mask, args.centering)
    flow_to_magnitudes(args.video, args.algorithm.value)
    analyse = initAnalyse(args.analyse, args.video, cap.get(cv.CAP_PROP_FRAME_HEIGHT), cap.get(cv.CAP_PROP_FRAME_WIDTH), args.algorithm, args.mask, args.centering)
    analyse.detectMovements(cap.get(cv.CAP_PROP_FPS))

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
        'centering',
        type=Centering,
        choices=list(Centering),
    )
    parser.add_argument(
        'analyse',
        type=Analyze,
        choices=list(Analyze)
    )
    args = parser.parse_args()
    main(args)
    
#exemple : /usr/bin/python3 /home/tino/Bureau/stage/Video_Analyzer_Tool/src/main.py ./resources/pu.mp4 Farneback MOG2 StartStop



