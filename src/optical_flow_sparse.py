import warnings
warnings.filterwarnings("ignore")

import cv2 as cv
import numpy as np
import argparse
from analystSparse import *

#Ce code permet de détecter des mouvements distincts (avec une pause entre les mouvements).

parser = argparse.ArgumentParser(description='This sample demonstrates Lucas-Kanade Optical Flow calculation. \
                                              The example file can be downloaded from: \
                                              https://www.bogotobogo.com/python/OpenCV_Python/images/mean_shift_tracking/slow_traffic_small.mp4')
parser.add_argument('image', type=str, help='path to image file')
args = parser.parse_args()

cap = cv.VideoCapture(args.image)

if not cap.isOpened():
    print(f"Erreur : Impossible d'ouvrir la source vidéo : {args.image}")
    exit()

ret, old_frame = cap.read()

if not ret or old_frame is None:
    print("Erreur : La première image est vide ou la fin du fichier est atteinte.")
    exit()
else:
    print("Succès : Fichier vidéo ouvert et première image lue.")

# params for ShiTomasi corner detection
feature_params = dict( maxCorners = 50,
                       qualityLevel = 10e-3,
                       minDistance = 30,
                       blockSize = 7 )

# Parameters for lucas kanade optical flow
lk_params = dict( winSize  = (15, 15),
                  maxLevel = 2,
                  criteria = (cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 10, 0.03))

# Create some random colors
color = np.random.randint(0, 255, (100, 3))

# Take first frame and find corners in it
ret, old_frame = cap.read()
old_gray = cv.cvtColor(old_frame, cv.COLOR_BGR2GRAY)
p0 = cv.goodFeaturesToTrack(old_gray, mask = None, **feature_params)

# Create a mask image for drawing purposes
mask = np.zeros_like(old_frame)

# Resize window
cv.namedWindow('frame', cv.WINDOW_NORMAL)
cv.resizeWindow('frame', 1280, 720)

# Init analyst
hauteur, largeur, canaux = old_frame.shape
tracker = AnalystSparse(hauteur, largeur)
fps = cap.get(cv.CAP_PROP_FPS)
frame_count = cap.get(cv.CAP_PROP_FRAME_COUNT)
duration = frame_count / fps
of_count_calculated = 1
while(1):
    ret, frame = cap.read()
    if not ret:
        print('Fin de la vidéo ou erreur de lecture')
        break

    frame_gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

    if p0 is None or len(p0) == 0:
        p0 = cv.goodFeaturesToTrack(frame_gray, mask=None, **feature_params)
        
        if p0 is None:
            of_count_calculated += 1
            tracker.update(np.empty((0, 2)), np.empty((0, 2)))
            cv.imshow('frame', frame) 
            old_gray = frame_gray.copy()
            if cv.waitKey(30) & 0xff == 27: 
                break
            continue 

    p1, st, err = cv.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **lk_params)
    if p1 is not None:
        good_new = p1[st==1]
        good_old = p0[st==1]

        if len(good_new) > 0:
            of_count_calculated += 1
            tracker.update(good_new, good_old)
            for i, (new, old) in enumerate(zip(good_new, good_old)):
                a, b = new.ravel()
                c, d = old.ravel()
                mask = cv.line(mask, (int(a), int(b)), (int(c), int(d)), color[i].tolist(), 2)
                frame = cv.circle(frame, (int(a), int(b)), 5, color[i].tolist(), -1)
            
            p0 = good_new.reshape(-1, 1, 2)
        else:
            of_count_calculated += 1
            tracker.update(np.empty((0, 2)), np.empty((0, 2)))  # ← frame vide
            p0 = None 
            
    img = cv.add(frame, mask)
    cv.imshow('frame', img)
    
    old_gray = frame_gray.copy()
    
    k = cv.waitKey(15) & 0xff
    if k == 27:
        break
    
cv.destroyAllWindows()

movement_count_serie, frame_start_serie, frame_end_serie = tracker.detectMovements()
frame_count_serie = frame_end_serie - frame_start_serie
duration_serie = frame_count_serie / fps
timer_start_serie = frame_start_serie / fps
timer_end_serie = frame_end_serie / fps
rythm_serie = tracker.getRythm(movement_count_serie, frame_count_serie, fps)

print("--INFOS DE LA VIDEO --")
print(f"fps de la video = {fps}")
print(f"nombre de frames de la video = {frame_count}")
print(f"durée de la video = {duration} sec")
print(f"nombre de flux optiques traités = {of_count_calculated}\n")

print("--INFOS DE LA SERIE --")
print(f"nombre de frames de la série de mouvements = {frame_count_serie}")
print(f"temps de début de la série de mouvements = {timer_start_serie} sec")
print(f"temps de fin de la série de mouvements = {timer_end_serie} sec")
print(f"durée totale de la série de mouvements = {timer_end_serie-timer_start_serie} sec")
print(f"mouvements détectés = {movement_count_serie}")
print(f"mouvements par seconde = {rythm_serie}")
print(f"mouvements par minute = {rythm_serie*60}")

#TODO debugger le graph
#tracker.printGraph()