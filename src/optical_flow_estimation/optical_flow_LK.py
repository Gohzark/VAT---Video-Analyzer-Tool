import cv2 as cv
import numpy as np
from signal_processing.analyzer import *
from .centering_processor import Centerer

def get_fg_mask(frame, mask_option):
    if mask_option is not None:  
        fg = mask_option.apply(frame)
        return fg
    return None 

def run_LK(cap, mask_option, centering, callback_progress=None, callback_image=None):
    image_width = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
    image_height = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))
    ret, old_frame = cap.read()
    if not ret or old_frame is None:
        print("Erreur : La première image est vide ou la fin du fichier est atteinte.")
        exit()
        
    total_frames = int(cap.get(cv.CAP_PROP_FRAME_COUNT))
    centerer = Centerer(centering, has_mask=(mask_option is not None))
    feature_params = dict(maxCorners=50, qualityLevel=1e-4, minDistance=20, blockSize=7)
    lk_params = dict(winSize=(15, 15), maxLevel=5,
                     criteria=(cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 10, 0.03))

    color = np.random.randint(0, 255, (100, 3))
    
    # Centrage de la première frame
    if mask_option is not None:
        fg_mask_init = mask_option.apply(old_frame)
    else:
        fg_mask_init = np.ones(old_frame.shape[:2], dtype=np.uint8) * 255

    old_frame_c, fg_mask_init_c = centerer.apply(old_frame, fg_mask_init)
    old_gray = cv.cvtColor(old_frame_c, cv.COLOR_BGR2GRAY)
    p0 = cv.goodFeaturesToTrack(old_gray, mask=fg_mask_init_c, **feature_params)
    drawingMask = np.zeros_like(old_frame_c)

    frames_data = []

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Fin de la vidéo ou erreur de lecture.")
            break

        if mask_option is not None:
            fg_mask = mask_option.apply(frame)
        else:
            fg_mask = np.ones(frame.shape[:2], dtype=np.uint8) * 255

        frame_c, fg_mask_c = centerer.apply(frame, fg_mask)
        frame_gray = cv.cvtColor(frame_c, cv.COLOR_BGR2GRAY)

        good_new, good_old = np.empty((0, 2)), np.empty((0, 2))

        if p0 is None or len(p0) == 0:
            fg_mask = get_fg_mask(frame, mask_option)
            p0 = cv.goodFeaturesToTrack(frame_gray, mask=fg_mask_c, **feature_params)
            if p0 is None:
                frames_data.append((0.0, 0.0))
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
                for i, (new, old) in enumerate(zip(good_new, good_old)):
                    a, b = new.ravel()
                    c, d = old.ravel()
                    drawingMask = cv.line(drawingMask, (int(a), int(b)), (int(c), int(d)),
                                         color[i % len(color)].tolist(), 2)
                    frame = cv.circle(frame, (int(a), int(b)), 5,
                                      color[i % len(color)].tolist(), -1)
                p0 = good_new.reshape(-1, 1, 2)
            else:
                p0 = None

        if len(good_new) > 0 and len(good_old) > 0:
            diff = good_new - good_old
            magn = np.linalg.norm(diff, axis=1)
            angle = float(np.mean(np.degrees(np.arctan2(diff[:, 1], diff[:, 0]))))
            score = float(np.sum(magn)) / (image_width * image_height)
            frames_data.append((score, angle))
        else:
            frames_data.append((0.0, 0.0))

        img = cv.add(frame, drawingMask)
        old_gray = frame_gray.copy()
        
        
        if callback_progress:
            callback_progress(len(frames_data), total_frames)
            
        if callback_image:
            callback_image(img)

    cv.destroyAllWindows()
    print("Terminé.")
    return np.array(frames_data)