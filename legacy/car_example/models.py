from dataclasses import dataclass
from typing import Tuple, List

# @dataclass
# class Genome:
#     t: float  # veličina guma
#     P: float  # snaga motora
#     B: float  # kapacitet baterije

@dataclass
class Car:
    x: List[float]  # (t, P, B) genome
    f: Tuple[float, float] = (0.0, 0.0) # (ubrzanje, distanca, cijena)
    rank: int = 0  # rang u populaciji, niže je bolje
    crowding_distance: float = 0.0  # crowding distance, veće je bolje
    # def __init__(self, x: Genome):
    #     self.x = x
    #     self.f = (0.0, 0.0)
    #     self.rank = 0
    #     self.crowding_distance = 0.0