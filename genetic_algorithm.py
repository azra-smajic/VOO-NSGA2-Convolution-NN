import logging
import random
from typing import List

from cnn_models import (
    BATCH_CHOICES,
    FILTERS_BASE_CHOICES,
    KERNEL_CHOICES,
    N_BLOCKS_CHOICES,
    Individual,
    evaluate_genome,
    random_genome,
)


LOGGER = logging.getLogger("genetic_algorithm")


def dominates(a: Individual, b: Individual):
    a1, a2 = a.f
    b1, b2 = b.f
    return (a1 <= b1 and a2 <= b2) and (a1 < b1 or a2 < b2)


def clamp(v: float) -> float:
    return max(0.0, min(1.0, v))


def fast_non_dominated_sort(population: List[Individual]):
    s_map = {id(p): [] for p in population}
    n_map = {id(p): 0 for p in population}
    fronts: List[List[Individual]] = [[]]

    for p in population:
        for q in population:
            if p is q:
                continue
            if dominates(p, q):
                s_map[id(p)].append(q)
            elif dominates(q, p):
                n_map[id(p)] += 1
        if n_map[id(p)] == 0:
            fronts[0].append(p)
            p.rank = 0

    i = 0
    while fronts[i]:
        next_front = []
        for p in fronts[i]:
            for q in s_map[id(p)]:
                n_map[id(q)] -= 1
                if n_map[id(q)] == 0:
                    next_front.append(q)
                    q.rank = i + 1
        i += 1
        fronts.append(next_front)

    fronts.pop()
    return fronts


def calculate_crowding_distance(front: List[Individual]):
    if not front:
        return

    for p in front:
        p.crowding_distance = 0.0

    num_objectives = 2
    for m in range(num_objectives):
        front.sort(key=lambda ind: ind.f[m])
        front[0].crowding_distance = float("inf")
        front[-1].crowding_distance = float("inf")

        f_min = front[0].f[m]
        f_max = front[-1].f[m]
        if f_max == f_min:
            continue

        for i in range(1, len(front) - 1):
            front[i].crowding_distance += (front[i + 1].f[m] - front[i - 1].f[m]) / (f_max - f_min)


def tournament_selection(population: List[Individual], k=2):
    a, b = random.sample(population, 2)
    if a.rank < b.rank:
        return a
    if b.rank < a.rank:
        return b
    return a if a.crowding_distance > b.crowding_distance else b


def crossover(p1: Individual, p2: Individual, p_gene_swap: float = 0.5) -> Individual:
    g1 = p1.x
    g2 = p2.x
    child = []

    for i in range(4):
        child.append(g1[i] if random.random() < p_gene_swap else g2[i])

    for i in range(4, 6):
        w = random.random()
        v = w * g1[i] + (1 - w) * g2[i]
        child.append(clamp(v))

    return Individual(child)


def mutate(ind: Individual, pm: float = 0.2, sigma_u: float = 0.12) -> None:
    g = ind.x

    choices_lens = [
        len(N_BLOCKS_CHOICES),
        len(FILTERS_BASE_CHOICES),
        len(KERNEL_CHOICES),
        len(BATCH_CHOICES),
    ]

    for i in range(4):
        if random.random() < pm:
            k = choices_lens[i]
            old = g[i]
            new = random.randrange(k)
            while new == old and k > 1:
                new = random.randrange(k)
            g[i] = new

    for i in range(4, 6):
        if random.random() < pm:
            g[i] = clamp(g[i] + random.gauss(0.0, sigma_u))


def make_next_generation(P: List[Individual], Q: List[Individual], N: int):
    R = P + Q
    fronts = fast_non_dominated_sort(R)
    for front in fronts:
        calculate_crowding_distance(front)

    next_gen = []
    for front in fronts:
        if len(next_gen) + len(front) <= N:
            next_gen.extend(front)
        else:
            front.sort(key=lambda x: x.crowding_distance, reverse=True)
            next_gen.extend(front[: N - len(next_gen)])
            break

    return next_gen


def nsga2(N=100, G=200, pc=0.9, pm=0.2, seed=1):
    LOGGER.info("NSGA-II start: N=%s, G=%s, pc=%s, pm=%s, seed=%s", N, G, pc, pm, seed)
    random.seed(seed)
    population = [Individual(random_genome()) for _ in range(N)]

    for individual in population:
        individual.f = evaluate_genome(individual.x)

    LOGGER.info("Inicijalna populacija evaluirana.")

    for gen in range(G):
        LOGGER.info("Pocinje generacija %s/%s", gen + 1, G)
        fronts = fast_non_dominated_sort(population)
        for front in fronts:
            calculate_crowding_distance(front)

        Q = []
        while len(Q) < N:
            parent1 = tournament_selection(population)
            parent2 = tournament_selection(population)
            if random.random() < pc:
                child = crossover(parent1, parent2)
            else:
                child = Individual(parent1.x.copy())
            mutate(child, pm)
            child.f = evaluate_genome(child.x)
            Q.append(child)

        population = make_next_generation(population, Q, N)
        LOGGER.info("Generacija %s zavrsena. Trenutna velicina populacije=%s", gen + 1, len(population))

    final_fronts = fast_non_dominated_sort(population)
    LOGGER.info("NSGA-II gotov. Finalni Pareto front ima %s rjesenja.", len(final_fronts[0]))
    return population, final_fronts[0]
