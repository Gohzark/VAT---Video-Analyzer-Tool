from dataclasses import dataclass, field
import numpy as np
from typing import Optional


@dataclass
class FlowData:
    """
    Interface commune entre les algorithmes de flux optique et les trackers.
    
    Dense (Farneback) produit :  mag, ang
    Sparse (Lucas-Kanade) produit : current_points, old_points
    
    Chaque tracker extrait ce dont il a besoin.
    """
    # --- Données dense (Farneback) ---
    mag: Optional[np.ndarray] = None       # magnitude du flux (H x W)
    ang: Optional[np.ndarray] = None       # angle du flux     (H x W)

    # --- Données sparse (Lucas-Kanade) ---
    current_points: Optional[np.ndarray] = None   # shape (N, 2)
    old_points: Optional[np.ndarray] = None       # shape (N, 2)

    def is_dense(self) -> bool:
        return self.mag is not None and self.ang is not None

    def is_sparse(self) -> bool:
        return self.current_points is not None and self.old_points is not None
