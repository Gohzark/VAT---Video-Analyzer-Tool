import numpy as np
from scipy.ndimage import minimum_filter
from .analyzer import Analyzer
from utils.flow_data import FlowData


class AnalyzerStartStop(Analyzer):

    means_x: list
    means_y: list
    threshold: float

    def __init__(self, height: int, width: int, threshold: float = 0.2):
        super().__init__(height, width)
        self.means_x   = []
        self.means_y   = []
        self.threshold = threshold

    def update(self, flow_data: FlowData) -> None:

        if flow_data.is_sparse():
            curr = flow_data.current_points
            old  = flow_data.old_points
            self._update_from_points(curr, old)

        elif flow_data.is_dense():
            mag = flow_data.mag
            ang = flow_data.ang
            ys, xs = np.where(mag > self.threshold)
            if len(xs) == 0:
                self._append_fallback()
                return
            # current_points = position des pixels en mouvement
            curr = np.column_stack((xs.astype(float), ys.astype(float)))
            # old_points = position décalée en sens inverse du flux
            old  = curr - np.column_stack((
                mag[ys, xs] * np.cos(np.radians(ang[ys, xs])),
                mag[ys, xs] * np.sin(np.radians(ang[ys, xs]))
            ))
            self._update_from_points(curr, old)

        else:
            self._append_fallback()

    def _update_from_points(self, curr: np.ndarray, old: np.ndarray) -> None:
        if len(curr) == 0 or len(old) == 0:
            self._append_fallback()
            return
        magn = np.linalg.norm(curr - old, axis=1)
        moving_points = curr[magn > self.threshold]
        if len(moving_points) > 0:
            means_xy = np.mean(moving_points, axis=0)
            self.means_x.append(means_xy[0])
            self.means_y.append(self.image_height - means_xy[1])
        else:
            self._append_fallback()

    def _append_fallback(self) -> None:
        if self.means_x:
            self.means_x.append(self.means_x[-1])
            self.means_y.append(self.means_y[-1])
        else:
            self.means_x.append(self.image_width  / 2)
            self.means_y.append(self.image_height / 2)


    def detectMovements(self):
        diffs_x      = np.diff(self.means_x)
        diffs_y      = np.diff(self.means_y)
        matrix_diffs = np.column_stack((diffs_x, diffs_y))
        norms        = np.linalg.norm(matrix_diffs, axis=1)
        clean_window = minimum_filter(norms, size=5, mode='nearest')
        vecteur_bool = clean_window > 0
        changes      = np.diff(vecteur_bool.astype(int))
        print(changes)

        nb_movement             = np.sum(changes == 1)
        indices_start_movements = np.argwhere(changes == 1)
        indices_end_movements   = np.argwhere(changes == -1)
        start_serie             = indices_start_movements[0].item()
        end_serie               = indices_end_movements[-1].item()

        if len(vecteur_bool) > 0 and vecteur_bool[0]:
            nb_movement += 1

        return round(nb_movement, 2), start_serie, end_serie

    def getRythm(self, nbMovements: float, nbFrame: int, frameRate: float) -> float:
        time  = nbFrame / frameRate
        rythm = nbMovements / time
        return round(rythm, 2)