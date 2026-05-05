import os
import numpy as np
from scipy.ndimage import minimum_filter
from .analyzer import *
from utils.flow_data import FlowData


class AnalyzerStartStop(Analyzer):

    means_x: list
    means_y: list
    threshold: float

    def __init__(self, image_path: str, height: int, width: int, algorithm: Algorithm, mask: str, threshold: float = 0.2):
        super().__init__(image_path, height, width, algorithm, mask)
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

    def detectMovements(self, fps: int):
        if len(self.means_x) < 2:
            print("[AnalyzerStartStop] Pas assez de données pour détecter des mouvements.")
            return

        diffs_x = np.diff(self.means_x)
        diffs_y = np.diff(self.means_y)
        matrix_diffs = np.column_stack((diffs_x, diffs_y))
        norms = np.linalg.norm(matrix_diffs, axis=1)
        clean_window = minimum_filter(norms, size=5, mode='nearest')
        vecteur_bool = clean_window > 0
        changes = np.diff(vecteur_bool.astype(int))

        indices_start_movements = np.argwhere(changes == 1)
        indices_end_movements   = np.argwhere(changes == -1)

        # Cas dégénérés : signal entièrement immobile ou entièrement en mouvement
        if len(indices_start_movements) == 0 and len(indices_end_movements) == 0:
            if vecteur_bool.any():
                # Toute la séquence est en mouvement
                start_serie = 0
                end_serie   = len(vecteur_bool) - 1
                nb_movement = 1
            else:
                print("[AnalyzerStartStop] Aucun mouvement détecté dans la séquence.")
                return
        else:
            # Bornage : si le signal commence déjà en mouvement, start = 0
            if len(indices_start_movements) == 0:
                start_serie = 0
            else:
                start_serie = indices_start_movements[0].item()

            # Bornage : si le signal finit encore en mouvement, end = dernière frame
            if len(indices_end_movements) == 0:
                end_serie = len(vecteur_bool) - 1
            else:
                end_serie = indices_end_movements[-1].item()

            nb_movement = len(indices_start_movements)
            if len(vecteur_bool) > 0 and vecteur_bool[0]:
                nb_movement += 1

        if end_serie <= start_serie:
            print(f"[AnalyzerStartStop] Durée de série invalide : start={start_serie}, end={end_serie}.")
            return

        frame_count_serie = end_serie - start_serie
        duration_serie    = frame_count_serie / fps
        timer_start_serie = start_serie / fps
        timer_end_serie   = end_serie / fps

        if duration_serie <= 0:
            print("[AnalyzerStartStop] Durée nulle, rythme incalculable.")
            return

        rythm_serie = self.getRythm(nb_movement, duration_serie)
        self.writeData(frame_count_serie, timer_start_serie, timer_end_serie, nb_movement, rythm_serie)

          

    def writeData(self, frame_count_serie: int, timer_start_serie: float, timer_end_serie: float, nb_movement: int, rythm_serie:float):
        file_path = f"data_{self.prefixe_file}.txt"
        with open(os.path.join(self.output_dir, file_path), "w", encoding="utf-8") as f:
            f.write(f"nombre de frames de la série de mouvements = {frame_count_serie}\n")
            f.write(f"temps de début de la série de mouvements = {timer_start_serie} sec\n")
            f.write(f"temps de fin de la série de mouvements = {timer_end_serie} sec\n")
            f.write(f"durée totale de la série de mouvements = {timer_end_serie-timer_start_serie} sec\n")
            f.write(f"mouvements détectés = {nb_movement}\n")
            f.write(f"mouvements par seconde = {rythm_serie}\n")
            f.write(f"mouvements par minute = {rythm_serie*60}\n")  
            
        
    def getRythm(self, nbMovements: float, duration_serie: float) -> float:
        rythm = nbMovements / duration_serie
        return round(rythm, 2)
    
    def toString(self) -> str:
        return "trackerSS"