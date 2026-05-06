from enum import Enum

class Algorithm(Enum):
    LK = "LK"
    FARNEBACK = "Farneback"

    def __str__(self):
        return self.name
    
class Mask(Enum):
    MOG2 = "MOG2"
    NoMask = "NoMask"
    
    def __str__(self):
        return self.name
    
class Tracker(Enum):
    Fourier = "Fourier"
    StartStop = "StartStop"
    
    def __str__(self):
        return self.name