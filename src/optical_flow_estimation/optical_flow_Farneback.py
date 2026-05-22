import numpy as np
import cv2 as cv
from signal_processing.analyzer import *
from utils.enums import Centering
    
def _init_kalman():
    kalman = cv.KalmanFilter(4, 2)
    kalman.measurementMatrix = np.array([[1,0,0,0],[0,1,0,0]], np.float32)
    kalman.transitionMatrix = np.array([[1,0,1,0],[0,1,0,1],[0,0,1,0],[0,0,0,1]], np.float32)
    kalman.processNoiseCov = np.eye(4, dtype=np.float32) * 0.01
    kalman.measurementNoiseCov = np.eye(2, dtype=np.float32) * 50
    return kalman

def run_Farneback(cap, mask, centering, callback_progress=None, callback_image=None):
    # Initialisation
    ret, frame = cap.read()
    if not ret:
        print("Erreur : Impossible de lire la première image.")
        return
    total_frames = int(cap.get(cv.CAP_PROP_FRAME_COUNT))
    previous_image = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    hsv = np.zeros_like(frame)
    hsv[..., 1] = 255

    # Variables EMA
    smooth_tx, smooth_ty = 0, 0
    alpha = 0.1
    first_detection = True

    # Variables Kalman
    kalman = _init_kalman()
    kalman_initialized = False

    M_transform = np.float32([[1, 0, 0], [0, 1, 0]])
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
        h, w = frame.shape[:2]

        if centering != Centering.NoCentering and mask is not None:
            total_area = h * w
            seuil_pourcentage = 0.0001
            contours, _ = cv.findContours(fg_mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
            if contours:
                c = max(contours, key=cv.contourArea)
                if cv.contourArea(c) > (total_area * seuil_pourcentage):
                    x, y, w_obj, h_obj = cv.boundingRect(c)
                    cx = x + w_obj // 2
                    cy = y + h_obj // 2

                    if centering == Centering.ExponentialMovingAverage:
                        target_tx = (w // 2) - cx
                        target_ty = (h // 2) - cy
                        if first_detection:
                            smooth_tx = target_tx
                            smooth_ty = target_ty
                            first_detection = False
                        else:
                            smooth_tx = (1 - alpha) * smooth_tx + alpha * target_tx
                            smooth_ty = (1 - alpha) * smooth_ty + alpha * target_ty
                        M_transform = np.float32([[1, 0, smooth_tx], [0, 1, smooth_ty]])

                    elif centering == Centering.Kalman:
                        measurement = np.array([[np.float32(cx)], [np.float32(cy)]])
                        if not kalman_initialized:
                            kalman.statePre = np.array([[cx], [cy], [0], [0]], np.float32)
                            kalman.statePost = np.array([[cx], [cy], [0], [0]], np.float32)
                            kalman_initialized = True
                        kalman.predict()
                        estimated = kalman.correct(measurement)
                        cx_smooth = estimated[0, 0]
                        cy_smooth = estimated[1, 0]
                        tx = (w // 2) - cx_smooth
                        ty = (h // 2) - cy_smooth
                        M_transform = np.float32([[1, 0, tx], [0, 1, ty]])

            frame_centered = cv.warpAffine(frame, M_transform, (w, h))
            mask_centered  = cv.warpAffine(fg_mask, M_transform, (w, h))
        else:
            frame_centered = frame
            mask_centered  = fg_mask

        frame_gray = cv.cvtColor(frame_centered, cv.COLOR_BGR2GRAY)
        flow = cv.calcOpticalFlowFarneback(previous_image, frame_gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
        mask_bool = mask_centered > 0
        flow[~mask_bool] = 0
        mag, ang = cv.cartToPolar(flow[..., 0], flow[..., 1], angleInDegrees=True)
        
        mask_bool = mask_centered > 0
        if np.any(mask_bool):
            score = float(np.sum(mag[mask_bool])) / (h * w)
            angle = float(np.mean(ang[mask_bool]))
        else:
            score, angle = 0.0, 0.0
        frames_data.append((score, angle))
        
        hsv[..., 0] = ang * 180 / np.pi / 2
        hsv[..., 2] = cv.normalize(mag, None, 0, 255, cv.NORM_MINMAX)
        bgr = cv.cvtColor(hsv, cv.COLOR_HSV2BGR)
        img = cv.addWeighted(frame_centered, 1, bgr, 1, 0)
        previous_image = frame_gray
        
        
        if callback_progress:
            callback_progress(len(frames_data), total_frames)
            
        if callback_image:
            callback_image(img)

    cv.destroyAllWindows()
    print("Terminé.")
    return np.array(frames_data)