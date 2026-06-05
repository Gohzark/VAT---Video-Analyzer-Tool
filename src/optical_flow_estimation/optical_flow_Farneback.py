from turtle import st

import numpy as np
import cv2 as cv
from signal_processing.analyzer import *
from .centering_processor import Centerer

def run_Farneback(cap, mask, centering, callback_progress=None, callback_image=None):
    # Initialisation
    ret, frame = cap.read()
    if not ret:
        print("Erreur : Impossible de lire la première image.")
        return
    centerer = Centerer(centering, has_mask=(mask is not None))
    total_frames = int(cap.get(cv.CAP_PROP_FRAME_COUNT))
    previous_image = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    hsv = np.zeros_like(frame)
    hsv[..., 1] = 255
    frames_data = []
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if mask is not None:
            fg_mask = mask.apply(frame)
        else:
            fg_mask = np.ones(frame.shape[:2], dtype=np.uint8) * 255

        kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (5, 5))
        fg_mask = cv.morphologyEx(fg_mask, cv.MORPH_OPEN, kernel)

        frame_centered, mask_centered = centerer.apply(frame, fg_mask)

        frame_gray = cv.cvtColor(frame_centered, cv.COLOR_BGR2GRAY)
        
        if previous_image.shape != frame_gray.shape:
            print(f"ERREUR DE TAILLE : prev {previous_image.shape} vs next {frame_gray.shape}")
        
        try:
            flow = cv.calcOpticalFlowFarneback(previous_image, frame_gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
        except Exception as e:
            print(f"Le calcul Farneback a crashé : {e}")
            break 
        
        flow = cv.calcOpticalFlowFarneback(previous_image, frame_gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)

        mask_bool = mask_centered > 0
        flow[~mask_bool] = 0
        mag, ang = cv.cartToPolar(flow[..., 0], flow[..., 1], angleInDegrees=True)
            
        frames_data.append(flow)

        hsv[..., 0] = ang * 180 / np.pi / 2
        hsv[..., 2] = cv.normalize(mag, None, 0, 255, cv.NORM_MINMAX)
        bgr = cv.cvtColor(hsv, cv.COLOR_HSV2BGR)
        img = cv.addWeighted(frame_centered, 1, bgr, 1, 0)
        previous_image = frame_gray
        
        if callback_progress:
            callback_progress(len(frames_data), total_frames)
            
        if callback_image:
            callback_image(img)

    print("Terminé.")
    return np.array(frames_data)