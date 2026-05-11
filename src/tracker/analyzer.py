import json
import os
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.ndimage import minimum_filter
from utils.flow_data import FlowData
from utils.enums import Algorithm, Analyze, Mask

#TODO: Ajouter le lissage du déplacement de la caméra

class Analyzer():
    video_name: str
    output_dir: str
    image_width: int
    image_height: int
    mean_angles: list
    mean_magnitudes: list
    means_x: list
    means_y: list
    analyze: Analyze
    threshold: float
    algorithm: Algorithm
    mask: Mask
    centering: bool

    def __init__(
        self,
        video_name: str,
        height: int,
        width: int,
        algorithm: Algorithm,
        mask: Mask,
        analyze: Analyze,
        centering: bool,
        threshold: float = 0.2,
    ):
        self.video_name = video_name
        self.image_height = height
        self.image_width = width
        self.algorithm = algorithm
        self.analyze = analyze
        self.mask = mask
        self.centering = centering
        script_dir = os.path.dirname(os.path.abspath(__file__))
        self.output_dir = os.path.join(script_dir, "../../outputs")
        os.makedirs(self.output_dir, exist_ok=True)
        self.analyze = analyze
        if (centering and mask is not None):
            centering = True
        self.threshold = threshold
        self.mean_magnitudes = []
        self.mean_angles = []
        self.means_x = []
        self.means_y = []


    def update(self, flow_data: FlowData) -> None:
        if self.analyze == Analyze.SS:
            self._update_startstop(flow_data)
        else:
            self._update_flow(flow_data)

    def _update_flow(self, flow_data: FlowData) -> None:
        if flow_data.is_dense():
            magn = flow_data.mag
            ang  = flow_data.ang
        elif flow_data.is_sparse():
            curr = flow_data.current_points
            old  = flow_data.old_points
            if len(curr) == 0 or len(old) == 0:
                self.mean_angles.append(0.0)
                self.mean_magnitudes.append(0.0)
                return
            diff = curr - old
            magn = np.linalg.norm(diff, axis=1)
            ang  = np.degrees(np.arctan2(diff[:, 1], diff[:, 0]))
        else:
            self.mean_angles.append(0.0)
            self.mean_magnitudes.append(0.0)
            return

        mask_mouvement = magn > 0
        if np.any(mask_mouvement):
            self.mean_angles.append(float(np.mean(ang[mask_mouvement])))
            score = float(np.sum(magn)) / (self.image_width * self.image_height)
            self.mean_magnitudes.append(score)
        else:
            self.mean_angles.append(0.0)
            self.mean_magnitudes.append(0.0)

    def _update_startstop(self, flow_data: FlowData) -> None:
        if flow_data.is_sparse():
            self._update_from_points(flow_data.current_points, flow_data.old_points)
        elif flow_data.is_dense():
            mag = flow_data.mag
            ang = flow_data.ang
            ys, xs = np.where(mag > self.threshold)
            if len(xs) == 0:
                self._append_fallback()
                return
            curr = np.column_stack((xs.astype(float), ys.astype(float)))
            old  = curr - np.column_stack((
                mag[ys, xs] * np.cos(np.radians(ang[ys, xs])),
                mag[ys, xs] * np.sin(np.radians(ang[ys, xs])),
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
            mean_xy = np.mean(moving_points, axis=0)
            self.means_x.append(float(mean_xy[0]))
            self.means_y.append(float(self.image_height - mean_xy[1]))
        else:
            self._append_fallback()

    def _append_fallback(self) -> None:
        if self.means_x:
            self.means_x.append(self.means_x[-1])
            self.means_y.append(self.means_y[-1])
        else:
            self.means_x.append(self.image_width  / 2)
            self.means_y.append(self.image_height / 2)

    def detectMovements(self, fps: float) -> None:
        match self.analyze:
            case Analyze.FFT:
                self._detectFFT(fps)
            case Analyze.Sliding:
                self._detectBySliding(fps)
            case Analyze.SS:
                self._detectStartStop(fps)

    def _detectStartStop(self, fps: float) -> None:
        if len(self.means_x) < 2:
            print("[GoodAnalyzer] Pas assez de données pour détecter des mouvements.")
            return

        norms = np.linalg.norm(
            np.column_stack((np.diff(self.means_x), np.diff(self.means_y))),
            axis=1,
        )
        clean_window = minimum_filter(norms, size=5, mode='nearest')
        vecteur_bool = clean_window > 0

        starts = np.where((vecteur_bool[:-1] == False) & (vecteur_bool[1:] == True))[0] + 1
        ends   = np.where((vecteur_bool[:-1] == True) & (vecteur_bool[1:] == False))[0] + 1

        # Cas particulier : commence en mouvement
        if vecteur_bool[0]:
            starts = np.insert(starts, 0, 0)
        # Cas particulier : finit en mouvement
        if vecteur_bool[-1]:
            ends = np.append(ends, len(vecteur_bool) - 1)

        nb_reps = len(starts)
        if nb_reps < 1:
            print("[GoodAnalyzer] Aucun mouvement distinct trouvé.")
            return

        # Durée totale entre le tout premier mouvement et la fin du dernier
        t_total = (ends[-1] - starts[0]) / fps
        
        if nb_reps > 1:
            # La période est le temps entre les débuts de mouvements successifs
            avg_period_frames = np.mean(np.diff(starts))
            period = avg_period_frames / fps
            frequency = 1 / period
        else:
            # Un seul mouvement : la période est sa durée propre
            period = t_total
            frequency = 1 / t_total if t_total > 0 else 0

        self._writeResults({
            "periode_sec": round(period, 2),
            "frequence_hz": round(frequency, 2),
            "nb_cycles_mouvement": nb_reps,
            "duree_active_sec": round(t_total, 2)
        })
        
    def _detectFFT(self, fps: float) -> None:
        array = np.array(self.mean_magnitudes)
        indices_non_nuls = np.nonzero(array)[0]

        if len(indices_non_nuls) < 2:
            print("[GoodAnalyzer] Pas assez de données non nulles pour la FFT.")
            return

        idx_debut  = indices_non_nuls[0]
        idx_fin    = indices_non_nuls[-1]
        N          = idx_fin - idx_debut
        segment    = array[idx_debut:idx_fin + 1]

        fft_result = np.fft.fft(segment)
        freqs      = np.fft.fftfreq(N, d=1 / fps)
        freqs_pos  = freqs[:N // 2]
        amplitudes = np.abs(fft_result[:N // 2]) * 2 / N

        self._plotFFT(freqs_pos, amplitudes)

        mask          = freqs_pos >= 0.1
        idx_pic       = np.argmax(amplitudes[mask])
        frequence_dom = round(float(freqs_pos[mask][idx_pic]), 2)
        amp_pic       = amplitudes[mask][idx_pic]
        ratio         = round(amp_pic / np.mean(amplitudes[mask]), 2)

        self._plotEvolution(self.mean_magnitudes, self.mean_angles, fps)
        self._writeResults({
            "periode_sec": round(1 / frequence_dom, 2),
            "frequence_hz": frequence_dom,
            "ratio" : ratio,
            "nb_cycles_mouvement": round(len(self.mean_magnitudes) / fps * frequence_dom, 2),
            "plot_fft": os.path.join(self._plot_dir()+"/plot_fft.png"),
            "plot_evolution": os.path.join(self._plot_dir()+"/plot_evolution.png"),
        })
        
    def _plotFFT(self, freqs_pos: np.ndarray, amplitudes: np.ndarray) -> None:
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(freqs_pos, amplitudes, color='steelblue')
        ax.set_xlabel("Fréquence (Hz)")
        ax.set_ylabel("Amplitude")
        ax.set_title("Spectre FFT")
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(self._plot_dir()+"/plot_fft.png", dpi=150)
        plt.close()
    
    
    #TODO: essayer de minimiser la période également
    def _detectBySliding(self, fps: float) -> None:
        array            = np.array(self.mean_magnitudes)
        indices_non_nuls = np.nonzero(array)[0]
        duree_totale = len(self.mean_magnitudes) / fps
        
        if len(indices_non_nuls) < 10:
            print("[GoodAnalyzer] Signal trop court pour l'analyse par fenêtre glissante.")
            return

        idx_debut         = indices_non_nuls[0]
        idx_fin           = indices_non_nuls[-1]
        magnitudes_utiles = array[idx_debut:idx_fin + 1]
        N                 = len(magnitudes_utiles)

        min_gap = max(1, int(0.2 * fps))
        max_gap = N // 2

        if min_gap >= max_gap:
            print("[GoodAnalyzer] Signal trop court pour la fréquence minimale demandée.")
            return

        best_gap, best_error = 0, sys.float_info.max
        for gap in range(min_gap, max_gap):
            cost = self._costByGap(magnitudes_utiles, gap)
            if cost < best_error:
                best_error, best_gap = cost, gap

        if best_gap > 0:
            periode = best_gap / fps
            self._writeResults({
                "periode_sec": round(periode, 2),
                "frequence_hz": round(1 / periode, 2),
                "nb_cycles_mouvement": round(len(self.mean_magnitudes) / fps / periode, 2),
            })

    def _costByGap(self, signal: np.ndarray, gap: int) -> float:
        if gap <= 0 or gap >= len(signal):
            return sys.float_info.max
        diff = signal[gap:] - signal[:-gap]
        return float(np.mean(diff ** 2))

    def _writeResults(self, data: dict) -> None:
        file_path = os.path.join(self.output_dir, "results.json")

        results = {}
        if os.path.exists(file_path):
            with open(file_path, "r", encoding="utf-8") as f:
                results = json.load(f)

        node = results
        for key in [self.video_name, self.algorithm.value, self.analyze.value, self.mask.value]:
            node = node.setdefault(key, {})

        node["centering"] = self.centering
        node.update(data)

        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=4, ensure_ascii=False)

    def _plotEvolution(self, magnitudes: list, angles: list, fps: float) -> None:
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6))
        t_mag = np.linspace(0, len(magnitudes) / fps, len(magnitudes))
        ax1.plot(t_mag, magnitudes, color='blue', label='Magnitude du mouvement')
        ax1.set_title("Évolution du mouvement")
        ax1.set_ylabel("Magnitude")
        ax1.legend()

        t_ang = np.linspace(0, len(angles) / fps, len(angles))
        ax2.plot(t_ang, angles, color='red', label='Angle moyen')
        ax2.set_xlabel("Temps (s)")
        ax2.set_ylabel("Angle (degrés)")
        ax2.legend()

        plt.tight_layout()
        plt.savefig(self._plot_dir()+"/plot_evolution.png")
        plt.close()

    def _plot_dir(self) -> str:
        path = os.path.join(
            self.output_dir,
            self.video_name,
            self.algorithm.value,
            self.analyze.value,
            self.mask.value if self.mask else "no_mask",
            "centering" if self.centering else "no_centering",
        )
        os.makedirs(path, exist_ok=True)
        return path

    def toString(self) -> str:
        match self.analyze:
            case Analyze.FFT:
                return "trackerFFT"
            case Analyze.Sliding:
                return "trackerSliding"
            case Analyze.StartStop:
                return "trackerSS"
            case _:
                return "tracker"
