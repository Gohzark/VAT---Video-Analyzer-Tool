from abc import ABC, abstractmethod

class Analyst(ABC):
    image_width: int
    image_height: int
    
    def __init__(self, height, width):
        self.image_height = height
        self.image_width = width

    @abstractmethod
    def update  (self, frame, mask):
        pass
    
    @abstractmethod
    def detectMovements(self):
        pass