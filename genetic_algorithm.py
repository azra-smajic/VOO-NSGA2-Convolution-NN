
import random
from typing import List
from constants import BOUNDS
from cnn_models import BATCH_CHOICES, FILTERS_BASE_CHOICES, KERNEL_CHOICES, N_BLOCKS_CHOICES, random_genome, evaluate_genome
from cnn_models import Individual


# Ovo nam je ono pareto pravilo, A dominira B ako je u oba cilja bolji ili jednak ili bar u jednom strogo bolji
def dominates(a:Individual, b:Individual):
    (a1, a2) = a.f
    (b1, b2) = b.f
    return (a1 <= b1 and a2 <= b2) and (a1 < b1 or a2 < b2)

def clamp(v: float) -> float:
    return max(0.0, min(1.0, v))

# def dominates(a: Car, b: Car) -> bool:
#     # a dominira b ako je a <= b u svim ciljevima
#     # i striktno bolje u barem jednom
#     better_or_equal_all = True
#     strictly_better_any = False

#     for av, bv in zip(a.f, b.f):
#         if av > bv:
#             better_or_equal_all = False
#             break
#         if av < bv:
#             strictly_better_any = True

#     return better_or_equal_all and strictly_better_any

# E sad s obzirom da NSGA ne radi na osnovu fitness u smislu da imamo jednu cifru kao obični GA, ovdje je fitness 
# dvodimenzionalan pa se kvalitet prvo mjeri kroz
# pareto rank - frontovi i kroz crowding distance - rspodjela

# Cilj ove f-je da napravi listu frontova i svakoj dodijeli rank, što manji rank to bolje
# dakle šaljemo populaciju a ona nam vraća listu frontova f[0] nedominirana rjesenja, f[1] najbolja rjesenja kad uklonimo f[0]..
# i svakoj jedinki dodjeljujemo rant u ovisnosti kojem frontu pripada.
def fast_non_dominated_sort(population:List[Individual]):
    S = {id(p): [] for p in population}  # S[p] je lista jedinki koje dominira p - za svaku jedinku pravimo po jednu praznu listu u koju ćemo smještati one koje dominira
    n = {id(p): 0 for p in population}   # n[p] je koliko njih dominira p - za svaku jedinku pravimo po jedan brojač koji nam govori koliko ih dominira
    fronts:List[List[Individual]] = [[]] # imamo listu frontova a svaki front je lista jedinki
    for p in population:
        for q in population:
            if p is q:
                continue
            if dominates(p, q):
                S[id(p)].append(q)  # p dominira q, dodajemo q u listu onih koje dominira p
            elif dominates(q, p):
                n[id(p)] += 1  # q dominira p, povećavamo brojač onih koji dominiraju p
        if n[id(p)] == 0:  # ako niko ne dominira p, onda je p na prvom frontu
            fronts[0].append(p)
            p.rank = 0  # rang prve fronte je 0
    i = 0
    while fronts[i]:  # dok ima jedinki u trenutnom frontu
        next_front = []
        for p in fronts[i]:
            for q in S[id(p)]:  # za svaku jedinku q koju dominira p
                n[id(q)] -= 1  # smanjujemo brojač onih koji dominiraju q jer je p već iz boljeg fronta i praktično ga uklanjamo iz konkurencije
                if n[id(q)] == 0:  # ako niko više ne dominira q, onda je q na sljedećem frontu
                    next_front.append(q)
                    q.rank = i + 1  # rang sljedećeg fronta je i+2 jer prvi front je 0, sljedeći rank je dva, kad je i=1 onda je rank 3, itd
        i += 1
        fronts.append(next_front)  # dodajemo sljedeći front u listu frontova
    fronts.pop()  # uklanjamo zadnji prazan front
    return fronts

# Sad rješavamo problem kad su jedinke u istom Pareto frontu tj imaju isti rank, NSGA mora odlučiti koje da zadrzi
# da front bude ravnomjerano pokriven a ne da sve bude nagurano u jedan dio.
# Ova funkcija daje bonus jedinkama koje su u rijeđim dijelovima prostora ciljeva
def calculate_crowding_distance(front:List[Individual]):
    if not front:
        return
    for p in front:
        p.crowding_distance = 0.0  # inicijaliziramo crowding distance na 0 jer će se računati iz početka za svaku generaciju
    num_objectives = 2  # broj ciljeva, u našem slučaju 2
    for m in range(num_objectives):
        front.sort(key=lambda car: car.f[m])  # sortiramo po m-tom cilju, prvo po vremenu ubrzanja pa onda range. Ovo će nam posložiti jedinke
        #po vrijednosti cilja, tako da najmanja vrijednost cilja bude na početku a najveća na kraju. Tako svaka tačka ima komšije lijevo i deno po tom trenutnom cilju
        
        front[0].crowding_distance = float('inf')  # krajnje jedinke imaju beskonačnu crowding distance
        front[-1].crowding_distance = float('inf') # drugi ekstrem i može biti najgori po jednom cilju a najbolji po drugom, ovo sve radimo jer trebamo sačuvat rubove
        # pronalazak minimalne i maksimalne vrijednosti cilja u frontu, ovo treba za normalizaciju, da crowding ne zavisi od skale.
        f_min = front[0].f[m]
        f_max = front[-1].f[m]
        if f_max == f_min:  # ako su svi isti po ovom cilju, crowding distance ostaje 0 za sve osim krajnjih. Uglavnom u tom slučaju ne koristimo taj cilj jer nam ne daje informaciju o razmaknutosti
            continue
        # za svaku unutrašnju tačku, koja nije ivica min ili maks, uzimamo lijevog i desnog komšiju i računamo razmak po tom cilju.
        # ako je tačka u gužvi, razlika je mala, crowding je mali
        # ako je tačka u rjeđem dijelu, razlika je velika, crowding je veliki
        for i in range(1, len(front) - 1):
            front[i].crowding_distance += (front[i + 1].f[m] - front[i - 1].f[m]) / (f_max - f_min)  # normalizirani crowding distance da nam ne uništi računicu

# Ova funkcija će nam pomoći da se izabere jedan roditelj iz populacije za proces ukrštanja
# Nećemo birati random. nego biramo najbolja rješenja ali ipet da budu raznolika, odnosno dobro distribuirana
def tournament_selection(population:List[Individual], k=2):
    # uzmemo dvije različite jedinke
    a, b = random.sample(population, 2)
    # prvo ih poredimo po frontu
    if a.rank < b.rank:
        return a
    if b.rank < a.rank:
        return b
    # onda ih poredimo po razmaku
    return a if a.crowding_distance > b.crowding_distance else b

# def crossover(parent1:Individual, parent2:Individual):
#     child = []
#     for (x1, x2), key in zip(zip(parent1.x, parent2.x), ["t", "P", "B"]): # pravimo listu parova (parent1.t, parent2.t), (parent1.P, parent2.P), (parent1.B, parent2.B) i iteriramo kroz njih zajedno sa ključevima da znamo koje su granice
#         lo, hi = BOUNDS[key]
#         alpha = random.random()  # uzimamo random alpha između 0 i 1, 0 znaci uzmi sve iz roditelja 2 a 1 znači uzmi sve iz roditelja 1, a između je mješavina tipa 0.3 je 30% roditelj 1 i 70% roditelj2
#         v = alpha * x1 + (1 - alpha) * x2  # mješavina dva roditelja
#         child.append(clamp(v, lo, hi))
#     return Individual(child)

def crossover(p1: Individual, p2: Individual, p_gene_swap: float = 0.5) -> Individual:
    """
    Uniform crossover za diskretne gene + miješanje za kontinuirane gene.
    p1.x i p2.x su genome liste.
    """
    g1 = p1.x
    g2 = p2.x
    child = []

    # 0..3: diskretni indeksi
    for i in range(4):
        child.append(g1[i] if random.random() < p_gene_swap else g2[i])

    # 4..5: kontinuirani geni u [0,1]
    for i in range(4, 6):
        w = random.random()
        # s obzirom da se radi o kontinuiranim genima možemo koristiti linearno miješanje, gdje uzimamo težinsku kombinaciju roditeljskih gena, što nam daje više varijacija u potomstvu nego da samo uzimamo jedan od roditelja, a i dalje zadržavamo informacije iz oba roditelja
        v = w * g1[i] + (1 - w) * g2[i]
        child.append(clamp(v))
    return Individual(child)

# ovdje ćemo malo prodrmati nove jedinke da ne bi bile iste kao i roditelji da dobijemo neko poboljšanje, ne mora nužno biti poboljšanje, može biti i pogoršanje
# def mutate(ind: Individual, pm: float = 0.2, sigma_frac: float = 0.08):
#     for i, key in enumerate(["t", "P", "B"]):
#         if random.random() < pm: # 20% šanse da mutiramo svaki od gena, tj. veličinu guma, snagu motora i kapacitet baterije
#             lo, hi = BOUNDS[key]
#             sigma = sigma_frac * (hi - lo) # određivanje jačine mutacije, mutacija je stavkjena ja oko 8% raspona, 
#             # npr ako je raspon za snagu motora 170 kWh, sigma je oko 13.6 kWh, što znači da će mutacija biti u prosjeku oko 13.6 kWh, 
#             # ali može biti i manje ili više zbog gausove distribucije, ovo nam daje realne mutacije koje nisu prevelike da bi uništile rješenje ali nisu ni premale da ne bi imale efekta
#             # dodajemo gausov šum tj normalnu raspodjelušto znači, većina promjena je mala, povremeno promjena bude veća ali rijetko - koristimo normalnu raspodjelu
#             ind.x[i] = clamp(ind.x[i] + random.gauss(0, sigma), lo, hi)
def mutate(ind: Individual, pm: float = 0.2, sigma_u: float = 0.12) -> None:
    """
    Mutacija za mješoviti genom:
    - diskretni indeksi: random promjena na drugi indeks
    - kontinuirani u geni: gaussian šum, clamp na [0,1]
    """
    g = ind.x

    # diskretni (0..3)
    choices_lens = [
        len(N_BLOCKS_CHOICES),
        len(FILTERS_BASE_CHOICES),
        len(KERNEL_CHOICES),
        len(BATCH_CHOICES),
    ]

    for i in range(4):
        if random.random() < pm:
            k = choices_lens[i]
            # izaberi drugi validan indeks (da ne ostane isti)
            old = g[i]
            new = random.randrange(k)
            while new == old and k > 1:
                new = random.randrange(k)
            g[i] = new

    # kontinuirani u (4..5)
    for i in range(4, 6):
        if random.random() < pm:
            g[i] = clamp(g[i] + random.gauss(0.0, sigma_u))

# E sad nam treba funkcija koja će uzeti roditelje P i djecu Q i spojiti ih u R.
# to se radi tako što ih se rangira po Pareto frontu i crowding distanci i izabere se N
# najboljih za novu populaciju
def make_next_generation(P:List[Individual], Q:List[Individual], N: int):
    R = P + Q  # spojimo roditelje i djecu
    fronts = fast_non_dominated_sort(R)  # rangiramo ih po frontovima
    for f in fronts:
        calculate_crowding_distance(f)  # izračunamo crowding distance za svaki front
    # ovo nam je bitno jer nam treba da znamo koji su frontovi najbolji, 
    # ali ako ne možemo uzeti cijeli front, onda ćemo uzeti one sa najvećim crowding distanceom da sačuvamo raznolikost
    
    next_gen = []
    for f in fronts:
        # idemo logikom ako mozemo cijeli prvi front ubaciti, ubacujemo ga, i tako redom dok ne
        # dodjemo do slucaja da ne moze cijeli front stat pa ga onda tebamo odrezati
        if len(next_gen) + len(f) <= N:
            next_gen.extend(f)
        else:
            # ako ne možemo uključiti cijeli front, uzmemo najbolje jedinke iz tog fronta
            # i onda uzmemo onoliku veličinu koliko možemo uzeti da popunimo novu populaciju
            f.sort(key=lambda x: x.crowding_distance, reverse=True)
            next_gen.extend(f[:N - len(next_gen)])
            break
    return next_gen

# Ova funkcija je glavna funkcija NSGA2 algortiam i ona pokreće sve prethodno definisane funkcije
# N - velicina populacije
# G - broj generacija
# pc - vjerovatnoća ukrštanja
# pm - vjerovatnoća mutacije
# seed - random seed za reproducibilnost - da mozemo ponavljati eksperimente sa istim rezultatima
def nsga2(N = 100, G = 200, pc = 0.9, pm = 0.2, seed = 1):
    random.seed(seed) # Svaki put kad pokrenemo sa istim seedom dobit ćemo istu inicijalnu populaciju i iste rezultate, što nam je važno za testiranje i analizu rezultata
    population = [Individual(random_genome()) for _ in range(N)]  # inicijaliziramo populaciju sa N nasumičnih jedinki
    # svaka jedinka ima random vrijednosti za parametre t, P i B ali u granicama okvira.
    
    for individual in population:
        individual.f = evaluate_genome(individual.x)  # evaluiramo svaku jedinku da dobijemo njihove ciljeve
        # ovo nam je bitno da bi znali f kako bi se poslije mogla raditi dominacija, rank i selekcija


    # sad idemo na glavnu petlju genetskog algoritma
    # za svaku generaciju ćemo rangirati po frontovima, napraviti djecu i izabrati novu populaciju
    for gen in range(G):
        fronts = fast_non_dominated_sort(population)  # rangiramo populaciju po frontovima
        for f in fronts:
            calculate_crowding_distance(f)  # izračunamo crowding distance za svaki front
        # svaku populaciju u narednoj generaciji moramo izacunati rank i distibuciju 
        # jer nam to treba kako bi izabrali roditelje. Po onoj gore formuli prvo preferiramo rank
        # pa onda preferiramo ona rješenja koja imaju bolji crowding distance
        Q = [] # e sad radimo ukrštanje. Ovdje ćemo praviti djecu dok ne dobijemo N djece,
        # jer nam treba N djece da bi mogli napraviti novu generaciju od N roditelja i N djece, ukupno 2N, pa onda izabrati najboljih N za sljedeću generaciju
        while len(Q) < N:
            parent1 = tournament_selection(population)  # biramo roditelje turnirskom selekcijom
            parent2 = tournament_selection(population)
            if random.random() < pc:  # sa vjerovatnoćom pc radimo crossover, znači 90% vremena pravimo dijete miješanjem p1 i p2
                # a 10% vremena samo pravimo kopiju prvog roditelja
                # ovo radimo zbog onog pravila da zadržimo čiste jedinke koje su dobre i da se ne uvodi uvijek miješanje
                child = crossover(parent1, parent2)
            else:
                child = Individual(parent1.x.copy())  # ako ne radimo crossover, dijete je kopija jednog roditelja
            mutate(child, pm)  # mutiramo dijete sa vjerovatnoćom pm da malo promijenimo vrijednosti komponenti
            child.f = evaluate_genome(child.x)  # evaluiramo dijete da dobijemo njegove ciljeve
            Q.append(child)
        population = make_next_generation(population, Q, N)  # pravimo novu generaciju od roditelja i djece preko onog gore elitističkog izbora
    final_fronts = fast_non_dominated_sort(population)
    return population, final_fronts[0]



