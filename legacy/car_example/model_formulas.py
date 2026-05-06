from .constants import ALPHA_M, ALPHA_P, ALPHA_P2, ALPHA_PM, ALPHA_T2, ALPHA_T_CONS, BETA_T1, BETA_T2, BETA_T_ACCEL, C0, ETA, K_B, MO, P_REF, T_REF
from .models import Car

# Ovdje sam napravila intuitivno prvu verziju u kojoj sam rekla, okej ako povećamo snagu motora, ubrzanje će se smanjiti,
# ali neću se baviti time koliko će se smanjiti, nego ću reći da je ubrzanje obrnuto proporcionalno snazi motora, da distanca ovisi direktno
# o kapacitetu baterije i to cu evaluirati #
def accel_v0(P):
    return 1.0 / P

def range_v0(B):
    return B  

def evaluate_v0(car:Car):
    t = car.x[0]
    P = car.x[1]
    B = car.x[2]

    f1 = accel_v0(P)
    f2 = -range_v0(B)
    return (f1, f2)

# E hajmo sad na drugu verziju, ovdje ćemo uvesti trade-off da baterija povećava masu
def mass(B):
    return MO + K_B * B

def accel_v1(P, B):
    m = mass(B)
    return m / P

def range_v1(B):
    return B * 5.0 # i dalje je domet proporcionalan kapacitetu baterije, ali sada kažemo da je 1 kWh baterije dovoljno za 5 km, što je realno, jer ako imamo bateriju od 100 kWh, to bi nam trebalo biti dovoljno za 500 km, što je realno za današnje električne automobile

def evaluate_v1(car:Car):
    t = car.x[0]
    P = car.x[1]
    B = car.x[2]

    f1 = accel_v1(P, B)
    f2 = -range_v1(B)
    return (f1, f2)

# Idemo na treću fazu, uključiti trade off za snagu motora. 
# Povećanje snage će uticati na potrošnju baterije a samim time to utiče na domet 
# kog može postići automobil

def accel_v2(P, B):
    m = mass(B)
    return m / P

def consumption(P, B):
    return C0 * (1+ ALPHA_P *(P/P_REF))

def range_v2(P, B):
    c = consumption(P, B)
    return (B * ETA) / c

def evaluate_v2(car:Car):
    t = car.x[0]
    P = car.x[1]
    B = car.x[2]
    f1 = accel_v2(P, B)
    f2 = -range_v2(P, B)
    return (f1, f2)

# E sad u četrtoj fazi dodajemo i gume, ako povećamo gume to utiče da se malo poveća potrošnja i da se malo smanji domet ali ne srastično
def accel_v3(t, P, B):
    base_accel = accel_v2(P, B)
    return base_accel * (1 + BETA_T_ACCEL * (t - T_REF)) #Znači ovdje sad dodajemo taj uticaj guma

def consumption_v3(t, P, B):
    base_consumption = consumption(P, B)
    return base_consumption * (1 + ALPHA_T_CONS * (t - T_REF)) #Znači ovdje sad dodajemo taj uticaj guma

def range_v3(t, P, B):
    c = consumption_v3(t, P, B)
    return (B * ETA) / c

def evaluate_v3(car:Car):
    t = car.x[0]
    P = car.x[1]
    B = car.x[2]
    f1 = accel_v3(t, P, B)
    f2 = -range_v3(t, P, B)
    return (f1, f2)

# Idemo na peti slučaj, kombinacija parametara
# na potrošnju utiču realno i masa i snaga i gume i sad ćemo tako predstaviti funkciju potrošnje

def consumption_v5(t, P, B):
    dt = (t / T_REF - 1)
    dp = (P / P_REF - 1)
    m_rel = (mass(B) / MO - 1)

    cons = C0 * (1
                 + ALPHA_P * dp
                 + ALPHA_P2 * (dp**2)
                 + ALPHA_M * m_rel
                 + ALPHA_T_CONS * dt
                 + ALPHA_T2 * (dt**2)
                 + ALPHA_PM * dp * m_rel)
    return max(0.05, cons) # postavljamo minimalnu potrošnju da ne bi bila nula ili negativna, što ne bi imalo smisla

def range_v5(t, P, B):
    c = consumption_v5(t, P, B)
    return (B * ETA) / c

def accel_v5(t, P, B):
    base = accel_v2(P, B)
    dt = (t / T_REF - 1.0)
    return base * (1.0 + BETA_T1 * dt + BETA_T2 * (dt*dt))

def evaluate_v5(car:Car):
    t = car.x[0]
    P = car.x[1]
    B = car.x[2]
    f1 = accel_v5(t, P, B)
    f2 = -range_v5(t, P, B)
    return (f1, f2)



# Hajmo probati metodom penalizacije, da vidimo oće li se išta promijeniti

P_KNEE = 180.0          # iznad ovoga penal
ALPHA_P_KNEE = 0.6  
    # jačina penala
B_KNEE = 90.0
ALPHA_B_KNEE = 0.4

def consumption_v6(t, P, B):
    dt = (t / T_REF - 1)
    dp = (P / P_REF - 1)
    m_rel = (mass(B) / MO - 1)
    b_excess = max(0.0, B / B_KNEE - 1.0)
    p_excess = max(0.0, P / P_KNEE - 1.0)   # 0 do P_KNEE, pa raste
    cons = C0 * (1
                 + ALPHA_P * dp
                 + ALPHA_P2 * (dp**2)
                 + ALPHA_M * m_rel
                 + ALPHA_T_CONS * dt
                 + ALPHA_T2 * (dt**2)
                 + ALPHA_PM * dp * m_rel)
    
    # penalizacije (dodaj odmah poslije ovog)
    p_excess = max(0.0, P / P_KNEE - 1.0)
    b_excess = max(0.0, B / B_KNEE - 1.0)
    pen_factor = 1.0 + ALPHA_P_KNEE*(p_excess**2) + ALPHA_B_KNEE*(b_excess**2)

    cons = cons * pen_factor
    return max(0.05, cons)

def range_v6(t, P, B):
    c = consumption_v6(t, P, B)
    return (B * ETA) / c

def accel_v6(t, P, B):
    base = accel_v2(P, B)
    dt = (t / T_REF - 1.0)
    return base * (1.0 + BETA_T1 * dt + BETA_T2 * (dt*dt))

def evaluate_v6(car:Car):
    t = car.x[0]
    P = car.x[1]
    B = car.x[2]
    f1 = accel_v6(t, P, B)
    f2 = -range_v6(t, P, B)
    return (f1, f2)

# A šta bi bilo kad bi imali tri cilja pa dodamo i cijenu 
PRICE0 = 18000.0 # bazna cijena šasije/elektronike (EUR) - konstanta
K_BATT = 120.0 # €/kWh (baterija)
K_POWER = 35.0 # €/kW (jači motor = skuplje)
K_TIRE = 180.0 # € po inču iznad 17" (gume/felge skuplje)

T_BASE = 17.0 # baza za gume u granicama

def price(t: float, P: float, B: float) -> float:
    # t u inčima, P u kW, B u kWh
    return (PRICE0
            + K_BATT * B
            + K_POWER * P
            + K_TIRE * max(0.0, t - T_BASE))

def evaluate_v7(car:Car):
    t = car.x[0]
    P = car.x[1]
    B = car.x[2]
    f1 = accel_v5(t, P, B)       
    f2 = -range_v5(t, P, B)       
    f3 = price(t, P, B)          
    return (f1, f2, f3)
