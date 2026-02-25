import random
from constants import BOUNDS

# Ova funkcija će nam pomoći da ograničimo vrijednosti unutar određenih granica, npr mutacija je promijenila snagu na 300 kWh a ne možemo imati više of 250
def clamp(value, min_val, max_val):
    return max(min_val, min(value, max_val))

# Vraća jedno random rješenje u granicama koje smo postavili, tj. random veličinu guma, snagu motora i kapacitet baterije
def random_individual():
    """Jedno random rješenje x = [t, P, B]."""
    t = random.uniform(*BOUNDS["t"])
    P = random.uniform(*BOUNDS["P"])
    B = random.uniform(*BOUNDS["B"])
    return [t, P, B]

