import cv2 as cv
import numpy as np
from tracker.analyzerStartStop import *

def get_fg_mask(frame, mask_option):
        if mask_option is not None:  
            fg = mask_option.apply(frame)
            return fg
        return None 
    
def run_sparse(cap, video_name, mask_option, tracker):

    ret, old_frame = cap.read()

    if not ret or old_frame is None:
        print("Erreur : La première image est vide ou la fin du fichier est atteinte.")
        exit()
    else:
        print("Succès : Fichier vidéo ouvert et première image lue.")

    # params for ShiTomasi corner detection
    # maxCorners : nombre maximum de coins à détecter
    # qualityLevel : seuil de qualité pour la détection des coins (plus bas = plus de coins détectés, plus haut = moins de coins détectés)
    # minDistance : distance minimale entre les coins détectés (en pixels)
    # blockSize : taille du bloc utilisé pour calculer les coins (en pixels)
    feature_params = dict( maxCorners = 50,
                        qualityLevel = 1e-4,
                        minDistance = 20,
                        blockSize = 7 )

    # Parameters for lucas kanade optical flow
    # winSize : taille de la fenêtre de recherche (en pixels)
    # maxLevel : nombre de niveaux de la pyramide d'images
    # criteria : critère d'arrêt pour l'algorithme de Lucas-Kanade (combinaison de nombre maximum d'itérations et de précision minimale)
    lk_params = dict( winSize  = (15, 15),
                    maxLevel = 2,
                    criteria = (cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 10, 0.03))

    # Create some random colors
    color = np.random.randint(0, 255, (100, 3))

    # Take first frame and find corners in it
    old_gray = cv.cvtColor(old_frame, cv.COLOR_BGR2GRAY)
    
    fg_mask = get_fg_mask(old_frame, mask_option)
    
    p0 = cv.goodFeaturesToTrack(old_gray, mask=fg_mask, **feature_params)

    # Create a mask image for drawing purposes
    drawingMask = np.zeros_like(old_frame)

    # Resize window
    cv.namedWindow('frame', cv.WINDOW_NORMAL)
    cv.resizeWindow('frame', 1280, 720)

   
    fps = cap.get(cv.CAP_PROP_FPS)
    frame_count = cap.get(cv.CAP_PROP_FRAME_COUNT)
    while(1):
        ret, frame = cap.read()
        if not ret:
            print('Fin de la vidéo ou erreur de lecture')
            break

        frame_gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

        if p0 is None or len(p0) == 0:
            fg_mask = get_fg_mask(frame, mask_option)
            p0 = cv.goodFeaturesToTrack(frame_gray, mask=fg_mask, **feature_params)
            
            if p0 is None:
                tracker.update(FlowData(
                    current_points=np.empty((0, 2)),
                    old_points=np.empty((0, 2)),
                ))
                cv.imshow('frame', frame) 
                old_gray = frame_gray.copy()
                if cv.waitKey(30) & 0xff == 27: 
                    break
                continue 

        p1, st, err = cv.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **lk_params)
        if p1 is not None:
            good_new = p1[st == 1]
            good_old = p0[st == 1]

            if len(good_new) > 0:
                tracker.update(FlowData(
                    current_points=good_new,
                    old_points=good_old,
                ))
                for i, (new, old) in enumerate(zip(good_new, good_old)):
                    a, b = new.ravel()
                    c, d = old.ravel()
                    drawingMask = cv.line(drawingMask, (int(a), int(b)), (int(c), int(d)),
                                         color[i % len(color)].tolist(), 2)
                    frame = cv.circle(frame, (int(a), int(b)), 5,
                                      color[i % len(color)].tolist(), -1)
                p0 = good_new.reshape(-1, 1, 2)
            else:
                tracker.update(FlowData(
                    current_points=np.empty((0, 2)),
                    old_points=np.empty((0, 2)),
                ))
                p0 = None
                
        img = cv.add(frame, drawingMask)
        cv.imshow('frame', img)
        
        old_gray = frame_gray.copy()
        
        k = cv.waitKey(15) & 0xff
        if k == 27:
            break
        
    cv.destroyAllWindows()
    # Calcul final
    print("Calcul des mouvements finaux...")
    tracker.detectMovements(fps)
    print("Terminé.")