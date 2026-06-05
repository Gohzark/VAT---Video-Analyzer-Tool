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
    
    if not cap.isOpened():
        print("Erreur : Impossible d'ouvrir le flux vidéo.")
        return np.empty((0, image_height, image_width, 2))

    ret, old_frame = cap.read()
    if not ret or old_frame is None:
        print("Erreur : La première image est vide.")
        return np.empty((0, image_height, image_width, 2))
        
    total_frames = int(cap.get(cv.CAP_PROP_FRAME_COUNT))
    centerer = Centerer(centering, has_mask=(mask_option is not None))
    feature_params = dict(maxCorners=50, qualityLevel=1e-4, minDistance=20, blockSize=7)
    lk_params = dict(winSize=(15, 15), maxLevel=5,
                     criteria=(cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 10, 0.03))

    color = np.random.randint(0, 255, (100, 3))
    
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
            break

        if mask_option is not None:
            fg_mask = mask_option.apply(frame)
        else:
            fg_mask = np.ones(frame.shape[:2], dtype=np.uint8) * 255

        frame_c, fg_mask_c = centerer.apply(frame, fg_mask)
        frame_gray = cv.cvtColor(frame_c, cv.COLOR_BGR2GRAY)

        good_new, good_old = np.empty((0, 2)), np.empty((0, 2))

        current_flow = np.zeros((image_height, image_width, 2), dtype=np.float32)

        if p0 is None or len(p0) == 0:
            fg_mask_c = get_fg_mask(frame_c, mask_option)
            p0 = cv.goodFeaturesToTrack(frame_gray, mask=fg_mask_c, **feature_params)
            if p0 is None:
                frames_data.append(current_flow) 
                old_gray = frame_gray.copy()
                continue

        p1, st, err = cv.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **lk_params)
        if p1 is not None:
            good_new = p1[st == 1]
            good_old = p0[st == 1]
            if len(good_new) > 0:
                for i, (new, old) in enumerate(zip(good_new, good_old)):
                    a, b = new.ravel()
                    c, d = old.ravel()
                    
                    # Dessin des repères visuels
                    drawingMask = cv.line(drawingMask, (int(a), int(b)), (int(c), int(d)), color[i % len(color)].tolist(), 2)
                    frame_c = cv.circle(frame_c, (int(a), int(b)), 5, color[i % len(color)].tolist(), -1)

                    dx = a - c
                    dy = b - d
                    
                    posX, posY = int(c), int(d)
                    if 0 <= posY < image_height and 0 <= posX < image_width:
                        current_flow[posY, posX, 0] = dx
                        current_flow[posY, posX, 1] = dy
                
                p0 = good_new.reshape(-1, 1, 2)
            else:
                p0 = None

        frames_data.append(current_flow)

        img = cv.add(frame_c, drawingMask)
        old_gray = frame_gray.copy()
        
        if callback_progress:
            callback_progress(len(frames_data), total_frames)
            
        if callback_image:
            callback_image(img)

    print("Terminé.")
    return np.array(frames_data)