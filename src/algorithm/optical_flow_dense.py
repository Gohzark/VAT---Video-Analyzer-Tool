import numpy as np
import cv2 as cv
from tracker.analyzerFourier import *
    
def run_dense(cap, mask, tracker, centering):
    
    # Initialisation
    ret, frame = cap.read()
    if not ret:
        print("Erreur : Impossible de lire la première image.")
        return
    previous_image = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    hsv = np.zeros_like(frame)
    hsv[..., 1] = 255
    hauteur, largeur, _ = frame.shape
    fps = cap.get(cv.CAP_PROP_FPS)

    # Variables pour le lissage
    smooth_tx, smooth_ty = 0, 0
    alpha = 0.1
    M_transform = np.float32([[1, 0, 0], [0, 1, 0]])  # identité par défaut

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

        # Centrage uniquement si activé ET masque disponible
        if centering and mask is not None:
            total_area = h * w
            seuil_pourcentage = 0.0001
            contours, _ = cv.findContours(fg_mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
            if contours:
                c = max(contours, key=cv.contourArea)
                if cv.contourArea(c) > (total_area * seuil_pourcentage):
                    x, y, w_obj, h_obj = cv.boundingRect(c)
                    cx = x + w_obj // 2
                    cy = y + h_obj // 2
                    target_tx = (w // 2) - cx
                    target_ty = (h // 2) - cy
                    smooth_tx = (1 - alpha) * smooth_tx + alpha * target_tx
                    smooth_ty = (1 - alpha) * smooth_ty + alpha * target_ty
                    M_transform = np.float32([[1, 0, smooth_tx], [0, 1, smooth_ty]])
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
        tracker.update(FlowData(mag=mag, ang=ang))

        hsv[..., 0] = ang * 180 / np.pi / 2
        hsv[..., 2] = cv.normalize(mag, None, 0, 255, cv.NORM_MINMAX)
        bgr = cv.cvtColor(hsv, cv.COLOR_HSV2BGR)
        overlay = cv.addWeighted(frame_centered, 1, bgr, 1, 0)
        cv.imshow('Frame', overlay)
        previous_image = frame_gray

        if cv.waitKey(1) == 27:
            print("Arrêt par l'utilisateur.")
            break

    cap.release()
    cv.destroyAllWindows()
    print("Calcul des mouvements finaux...")
    tracker.detectMovements(fps)
    print("Terminé.")
