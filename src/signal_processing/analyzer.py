import json
import os
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.ndimage import minimum_filter
from utils.enums import Algorithm, Analyze, Mask, Centering

class Analyzer():
    video_name: str
    threshold: float
    algorithm: Algorithm
    mask: Mask
    centering: bool
    analyze: Analyze
    output_dir: str
    
    def __init__(
        self,
        video_name: str,
        algorithm: Algorithm,
        mask: Mask,
        centering: Centering,
        analyze: Analyze,
        threshold: float = 0.2,
    ):
        self.video_name = video_name
        self.algorithm = algorithm
        self.analyze = analyze
        self.mask = mask
        self.centering = centering
        script_dir = os.path.dirname(os.path.abspath(__file__))
        self.output_dir = os.path.join(script_dir, "../../outputs")
        os.makedirs(self.output_dir, exist_ok=True)
        if (centering and mask is not None):
            centering = True
        self.threshold = threshold

        
        
    def _detectStartStop(self, magnitudes: np.ndarray, fps: float) -> None:
        if np.size(magnitudes) < 2:
            print("[Analyse] Pas assez de données pour détecter des mouvements.")
            return

        norms = np.array(magnitudes)
    
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
            print("[Analyse] Aucun mouvement distinct trouvé.")
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
        
    def _detectFFT(self, magnitudes: np.ndarray, fps: float) -> None:
        array = np.array(magnitudes)
        indices_non_nuls = np.nonzero(array)[0]

        if len(indices_non_nuls) < 2:
            print("[Analyse] Pas assez de données non nulles pour la FFT.")
            return

        idx_debut  = indices_non_nuls[0]
        idx_fin    = indices_non_nuls[-1]
        segment = array[idx_debut:idx_fin + 1]
        N       = len(segment)

        fft_result = np.fft.fft(segment)
        freqs      = np.fft.fftfreq(N, d=1 / fps)
        freqs_pos  = freqs[:N // 2]
        amplitudes = np.abs(fft_result[:N // 2]) * 2 / N

        self._plotFFT(freqs_pos, amplitudes)

        mask          = freqs_pos >= 0.1
        idx_pic       = np.argmax(amplitudes[mask])
        frequence_dom = round(float(freqs_pos[mask][idx_pic]), 2)
        amp_pic       = amplitudes[mask][idx_pic]
        ratio = round(float(amp_pic) / float(np.mean(amplitudes[mask])), 2)
    
        self._writeResults({
            "periode_sec": round(1 / frequence_dom, 2),
            "frequence_hz": frequence_dom,
            "ratio" : ratio,
            "nb_cycles_mouvement": round(len(magnitudes) / fps * frequence_dom, 2),
            "plot_fft": os.path.join(self._plot_dir(), self.analyze.value, "plot_fft.png"),
            "plot_evolution": os.path.join(self._plot_dir(), self.analyze.value, "plot_evolution.png"),
        })
        
    def _plotFFT(self, freqs_pos: np.ndarray, amplitudes: np.ndarray) -> None:
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(freqs_pos, amplitudes, color='steelblue')
        ax.set_xlabel("Fréquence (Hz)")
        ax.set_ylabel("Amplitude")
        ax.set_title("Spectre FFT")
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        dir = self._plot_dir()
        os.makedirs(os.path.dirname(dir), exist_ok=True)
        plt.savefig(dir , dpi=150)        
        plt.close()
    
    
    def _detectBySliding(self, magnitudes: np.ndarray, fps: float) -> None:
        
        array            = np.array(magnitudes)
        indices_non_nuls = np.nonzero(array)[0]
        
        if len(indices_non_nuls) < 10:
            print("[Analyse] Signal trop court pour l'analyse par décalage du signal.")
            return

        idx_debut         = indices_non_nuls[0]
        idx_fin           = indices_non_nuls[-1]
        magnitudes_utiles = array[idx_debut:idx_fin + 1]
        N                 = len(magnitudes_utiles)

        min_gap = max(1, int(0.2 * fps))
        max_gap = N // 2

        if min_gap >= max_gap:
            print("[Analyse] Signal trop court pour la fréquence minimale demandée.")
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
                "nb_cycles_mouvement": round(len(magnitudes) / fps / periode, 2),
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
        if self.centering != Centering.NoCentering or self.mask is not None:
            centering_key = self.centering.value
        else :
            centering_key = "NoCentering"
        node.setdefault(centering_key, {}).update(data)
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=4, ensure_ascii=False)

    def _plot_dir(self) -> str:
        path = os.path.join(
            self.output_dir,
            self.video_name,
            self.algorithm.value,
            self.mask.value,
            self.centering.value,
        )
        os.makedirs(path, exist_ok=True)
        return path

    def toString(self) -> str:
        match self.analyze:
            case Analyze.FastFourierTransformation:
                return "trackerFFT"
            case Analyze.Sliding:
                return "trackerSliding"
            case Analyze.StartStop:
                return "trackerSS"
            case _:
                return "tracker"
