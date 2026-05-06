from enum import Enum

class Algorithm(Enum):
    LK = "LK"
    Farneback = "Farneback"

    def __str__(self):
        return self.name
    
class Mask(Enum):
    MOG2 = "MOG2"
    NoMask = "NoMask"
    
    def __str__(self):
        return self.name
    
class Tracker(Enum):
    Fourier = "Fourier"
    SS = "StartStop"
    RS = "RepetingSignal"
    
    def __str__(self):
        return self.name
    