from enum import Enum, auto

class Datasets(Enum):
    VEHICLE_ID = auto()
    VRIC = auto()
    CARS196 = auto()
    BOXCARS116K = auto()
    COMP_CARS = auto()
    VERI = auto()
    COMBINED = auto()
    
    def __str__(self):
        return self.name
    
    @staticmethod
    def from_string(s):
        try:
            return Datasets[s]
        except KeyError:
            raise ValueError()