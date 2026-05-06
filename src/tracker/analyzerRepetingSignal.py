import os
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from .analyzer import *
from utils.flow_data import FlowData

class AnalyzerRepetingSignal(Analyzer):

    mean_angles: list
    mean_magnitudes: list
    script_dir: str
    
    def __init__(self, image_path: str, height: int, width: int, algorithm: Algorithm, mask: str, centering: bool):
        super().__init__(image_path, height, width, algorithm, mask, centering)
        self.mean_magnitudes = []
        self.mean_angles = []


    def update(self, flow_data: FlowData) -> None:
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
            # Aucune donnée valide
            self.mean_angles.append(0.0)
            self.mean_magnitudes.append(0.0)
            return

        mask_mouvement = magn > 0
        if np.any(mask_mouvement):
            self.mean_angles.append(np.mean(ang[mask_mouvement]))
            mouvement_total = np.sum(magn)
            score = mouvement_total / (self.image_width * self.image_height)
            self.mean_magnitudes.append(score)
        else:
            self.mean_angles.append(0.0)
            self.mean_magnitudes.append(0.0)

    def detectMovements(self, fps: float):
        mean_magnitudes_array = np.array(self.mean_magnitudes)
        indices_non_nuls = np.nonzero(mean_magnitudes_array)[0]

        if len(indices_non_nuls) < 10: # Augmenté car 2 points ne font pas un cycle
            return

        idx_debut = indices_non_nuls[0]
        idx_fin = indices_non_nuls[-1]
        magnitudes_utiles = mean_magnitudes_array[idx_debut : idx_fin + 1]
        N = len(magnitudes_utiles)
        
        min_gap = max(1, int(0.2 * fps)) 
        max_gap = N // 2
        
        if min_gap >= max_gap:
            print("[Warning] Signal trop court pour la fréquence minimale demandée.")
            return

        bestGap = 0
        bestError = sys.float_info.max

        for i in range(min_gap, max_gap):
            current_cost = self.costByGap(magnitudes_utiles, i)
            
            if current_cost < bestError:
                bestError = current_cost
                bestGap = i
                
        if bestGap > 0:
            self.writeData(bestGap / fps)
        
    def costByGap(self, l1, gap):
        if gap <= 0 or gap >= len(l1):
            return sys.float_info.max
        
        diff = l1[gap:] - l1[:-gap]
        return np.mean(diff**2)
        
    def writeData(self, periode: float):
        file_path = f"data_{self.prefixe_file}.txt"
        with open(os.path.join(self.output_dir, file_path), "w", encoding="utf-8") as f:
            f.write(f"Période entre chaque répétition de mouvements : {periode} secondes\n")

    def plot_evolution(self, magnitudes, angles, fps):
        file_path = f"mouvement_{self.prefixe_file}.png"
        plt.figure(figsize=(10, 6))

        plt.subplot(2, 1, 1)
        plt.plot(np.linspace(0, len(magnitudes) / fps, len(magnitudes)),
                 magnitudes, color='blue', label='Magnitude du mouvement')
        plt.title("Évolution du mouvement")
        plt.ylabel("Magnitude")
        plt.legend()

        plt.subplot(2, 1, 2)
        plt.plot(np.linspace(0, len(angles) / fps, len(angles)),
                 angles, color='red', label='Angle moyen')
        plt.xlabel("Temps")
        plt.ylabel("Angle (degrés)")
        plt.legend()

        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, file_path))
        plt.close()
        
    def toString(self) -> str:
        return "trackerRS"