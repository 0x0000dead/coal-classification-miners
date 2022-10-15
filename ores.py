from enum import Enum, unique


@unique
class Ores(Enum):
    def __init__(self, min_size, max_size, percent, description=None):
        self.min_size = min_size
        self.max_size = max_size
        self.percent = percent
        self.description = f"from {min_size}mm to {max_size}mm" if description is None else description

    XXXL = (250, -1, 24.82, "more than 250mm")
    XXL = (150, 250, 30.92)
    XL = (100, 150, 11.77)
    L = (80, 100, 4.53)
    M = (70, 80, 2.81)
    S = (40, 70, 7.08)
    XS = (0, 40, 18.07)
