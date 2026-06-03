import numpy as np
import cv2 as cv
from utils.enums import Centering


def _init_kalman():
    kalman = cv.KalmanFilter(4, 2)
    kalman.measurementMatrix = np.array([[1,0,0,0],[0,1,0,0]], np.float32)
    kalman.transitionMatrix = np.array([[1,0,1,0],[0,1,0,1],[0,0,1,0],[0,0,0,1]], np.float32)
    kalman.processNoiseCov = np.eye(4, dtype=np.float32) * 0.01
    kalman.measurementNoiseCov = np.eye(2, dtype=np.float32) * 50
    return kalman


class Centerer:
    """
    Encapsule la logique de centrage (EMA, Kalman, ou aucun).
    Maintient l'état entre les frames.
    Usage : frame_c, mask_c = centerer.apply(frame, fg_mask)
    """

    SEUIL_POURCENTAGE = 0.0001

    def __init__(self, centering: Centering, has_mask: bool):
        self.centering = centering
        self.has_mask = has_mask

        # État EMA
        self.smooth_tx = 0.0
        self.smooth_ty = 0.0
        self.alpha = 0.1
        self.first_detection = True

        # État Kalman
        self.kalman = _init_kalman()
        self.kalman_initialized = False

        # Transformation courante (identité par défaut)
        self.M_transform = np.float32([[1, 0, 0], [0, 1, 0]])

    def apply(self, frame: np.ndarray, fg_mask: np.ndarray):
        """
        Applique le centrage sur frame et fg_mask.
        Retourne (frame_centree, mask_centre).
        """
        h, w = frame.shape[:2]

        if self.centering == Centering.NoCentering or not self.has_mask:
            return frame, fg_mask

        contours, _ = cv.findContours(fg_mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
        if contours:
            c = max(contours, key=cv.contourArea)
            if cv.contourArea(c) > (h * w * self.SEUIL_POURCENTAGE):
                x, y, w_obj, h_obj = cv.boundingRect(c)
                cx = x + w_obj // 2
                cy = y + h_obj // 2
                self.M_transform = self._compute_transform(cx, cy, w, h)

        frame_centered = cv.warpAffine(frame, self.M_transform, (w, h))
        mask_centered  = cv.warpAffine(fg_mask, self.M_transform, (w, h))
        return frame_centered, mask_centered

    def _compute_transform(self, cx, cy, w, h) -> np.ndarray:
        if self.centering == Centering.ExponentialMovingAverage:
            target_tx = (w // 2) - cx
            target_ty = (h // 2) - cy
            if self.first_detection:
                self.smooth_tx = target_tx
                self.smooth_ty = target_ty
                self.first_detection = False
            else:
                self.smooth_tx = (1 - self.alpha) * self.smooth_tx + self.alpha * target_tx
                self.smooth_ty = (1 - self.alpha) * self.smooth_ty + self.alpha * target_ty
            return np.float32([[1, 0, self.smooth_tx], [0, 1, self.smooth_ty]])

        elif self.centering == Centering.Kalman:
            measurement = np.array([[np.float32(cx)], [np.float32(cy)]])
            if not self.kalman_initialized:
                self.kalman.statePre  = np.array([[cx], [cy], [0], [0]], np.float32)
                self.kalman.statePost = np.array([[cx], [cy], [0], [0]], np.float32)
                self.kalman_initialized = True
            self.kalman.predict()
            estimated = self.kalman.correct(measurement)
            tx = (w // 2) - estimated[0, 0]
            ty = (h // 2) - estimated[1, 0]
            return np.float32([[1, 0, tx], [0, 1, ty]])

        return self.M_transform  # inchangé si mode inconnu
