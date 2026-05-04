import numpy as np
import matplotlib
matplotlib.use('Agg') 
import matplotlib.pyplot as plt
from analyst.analyst import Analyst
from scipy.ndimage import minimum_filter
from scipy.fft import fft, fftfreq
from scipy.signal import find_peaks
import os



class AnalystFourier(Analyst):
    
    mean_angles: list[float]
    mean_magnitudes:list[float]
  
    script_dir: str
    
    def __init__(self, height, width):
        super().__init__(height, width)
        self.mean_magnitudes = []
        self.mean_angles = []
        self.script_dir = os.path.dirname(os.path.abspath(__file__))
        self.output_dir = os.path.join(self.script_dir, "../outputs")
        os.makedirs(self.output_dir, exist_ok=True)

    def update(self, magn, ang):
        mask_mouvement = magn > 0
        if np.any(mask_mouvement):
            self.mean_angles.append(np.mean(ang[mask_mouvement]))
            mouvement_total = np.sum(magn)
            score = mouvement_total / (self.image_width * self.image_height)
            self.mean_magnitudes.append(score)
        else:
            self.mean_angles.append(0.0)
            self.mean_magnitudes.append(0.0)
        
    #Basé sur les changements de directions brusque
    def detectMovements(self, fps, image_path):
        N = len(self.mean_magnitudes)
        fft_result = np.fft.fft(self.mean_magnitudes)
        freqs = np.fft.fftfreq(N, d=1/fps)
        freqs_pos = freqs[:N//2]
        amplitudes = np.abs(fft_result[:N//2]) * 2 / N
        
        couples = list(zip(freqs_pos, amplitudes))
        couples_triees = sorted(couples, key=lambda x: x[1], reverse=True)
        
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(freqs_pos, amplitudes, color='steelblue')
        ax.set_xlabel("Fréquence (Hz)")
        ax.set_ylabel("Amplitude")
        ax.set_title("Spectre FFT")
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        file_path = os.path.join(self.output_dir, f"fft_output_{image_path}.png")
        plt.savefig(file_path, dpi=150)
        plt.close()
        
        f_min = 0.1  
        mask = freqs_pos >= f_min
        idx_pic = np.argmax(amplitudes[mask])
        frequence_dominante = freqs_pos[mask][idx_pic]
        frequence_dominante = round(frequence_dominante, 2)
        amp_pic = amplitudes[mask][idx_pic]
        amp_moyenne = np.mean(amplitudes[mask])
        ratio = amp_pic / amp_moyenne
        duree = N / fps
        nb_mouvements_theoriques = int(duree * frequence_dominante)
        rythm_theorique = self.getRythm(nb_mouvements_theoriques, N, fps)
        file_path = os.path.join(self.output_dir, f"data_{image_path}.txt")
        with open(file_path, "w", encoding="utf-8") as fichier:
            fichier.write(f"--INFOS SUPPOSÉES SUR LES MOUVEMENTS --\n")
            fichier.write(f"Fréquence dominante : {frequence_dominante:.2f} Hz\n")
            fichier.write(f"Ratio (combien de fois plus grand que la moyenne) : {ratio:.1f}x\n")
            fichier.write(f"Nombre de mouvements : {nb_mouvements_theoriques}\n")
            fichier.write(f"Rythme de mouvements : {rythm_theorique} mouvements par seconde\n")
            fichier.write(f"Mouvements par minute : {rythm_theorique*60}\n")
        
        self.plot_evolution(self.mean_magnitudes, self.mean_angles, fps, image_path)
        
        
    def getRythm(self, nbMovements, nbFrame, frameRate):
        time = nbFrame / frameRate
        rythm = nbMovements / time
        return round(rythm, 2)
        
    def plot_evolution(self, magnitudes, angles, fps, image_path):

        plt.figure(figsize=(10, 6))
        
        # Graphique des Magnitudes
        plt.subplot(2, 1, 1)
        plt.plot(np.linspace(0, len(magnitudes)/fps, len(magnitudes)),magnitudes, color='blue', label='Magnitude du mouvement')
        plt.title("Évolution du mouvement")
        plt.ylabel("Magnitude")
        plt.legend()
        
        # Graphique des Angles
        plt.subplot(2, 1, 2)
        plt.plot(np.linspace(0, len(angles)/fps, len(angles)), angles, color='red', label='Angle moyen')
        plt.xlabel("Temps")
        plt.ylabel("Angle (degrés)")
        plt.legend()
        
        plt.tight_layout()
        file_path = os.path.join(self.output_dir, f"resultat_mouvement_{image_path}.png")
        plt.savefig(file_path)
        plt.close()
        
        