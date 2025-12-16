from __future__ import annotations

import argparse
import math
import random
from typing import List, Tuple


# Basit amaç fonksiyonu:
# f(x) = x^2 fonksiyonunun minimumunu [-10, 10] aralığında yaklaşık bulmaya çalışacağız.
# GA, x'i ikili (binary) kromozom olarak temsil edecek.


def decode(chromosome: List[int], xmin: float = -10.0, xmax: float = 10.0) -> float:
    """Binary kromozomu gerçek sayıya çevir."""
    # kromozom: [0,1,1,0, ...]
    bits = len(chromosome)
    int_val = 0
    for b in chromosome:
        int_val = (int_val << 1) | b
    max_int = (1 << bits) - 1
    x = xmin + (xmax - xmin) * int_val / max_int
    return x


def fitness(chromosome: List[int]) -> float:
    """Amaç f(x) = x^2; minimum arıyoruz. Fitness'i 1 / (1 + f(x)) yapalım."""
    x = decode(chromosome)
    fx = x * x
    return 1.0 / (1.0 + fx)


def random_chromosome(bits: int) -> List[int]:
    return [random.randint(0, 1) for _ in range(bits)]


def tournament_selection(
    population: List[List[int]],
    k: int = 3,
) -> List[int]:
    """Kişiler arasından turnuva ile seçim."""
    selected = random.sample(population, k)
    selected.sort(key=lambda c: fitness(c), reverse=True)
    return selected[0].copy()


def one_point_crossover(
    parent1: List[int],
    parent2: List[int],
    crossover_prob: float,
) -> Tuple[List[int], List[int]]:
    if random.random() > crossover_prob or len(parent1) < 2:
        return parent1.copy(), parent2.copy()
    point = random.randint(1, len(parent1) - 1)
    child1 = parent1[:point] + parent2[point:]
    child2 = parent2[:point] + parent1[point:]
    return child1, child2


def mutate(chromosome: List[int], mutation_prob: float) -> None:
    for i in range(len(chromosome)):
        if random.random() < mutation_prob:
            chromosome[i] = 1 - chromosome[i]


def best_individual(population: List[List[int]]):
    best = max(population, key=lambda c: fitness(c))
    return best, fitness(best), decode(best)


def run_ga(
    pop_size: int = 50,
    bits: int = 16,
    generations: int = 50,
    crossover_prob: float = 0.8,
    mutation_prob: float = 0.01,
    seed: int = 42,
):
    random.seed(seed)

    # Başlangıç popülasyonu
    population = [random_chromosome(bits) for _ in range(pop_size)]

    print("=== Genetic Algorithm: minimize x^2 on [-10, 10] ===")
    for gen in range(generations):
        new_population: List[List[int]] = []

        # Elitizm: en iyi bireyi doğrudan taşıyalım
        best, best_fit, best_x = best_individual(population)
        new_population.append(best.copy())

        # Yeni bireyler üret
        while len(new_population) < pop_size:
            p1 = tournament_selection(population)
            p2 = tournament_selection(population)
            c1, c2 = one_point_crossover(p1, p2, crossover_prob)
            mutate(c1, mutation_prob)
            mutate(c2, mutation_prob)
            new_population.append(c1)
            if len(new_population) < pop_size:
                new_population.append(c2)

        population = new_population

        if (gen + 1) % 10 == 0 or gen == 0:
            best, best_fit, best_x = best_individual(population)
            print(
                f"Gen {gen+1:3d}: "
                f"best x = {best_x:.4f}, "
                f"f(x) = {best_x*best_x:.6f}, "
                f"fitness = {best_fit:.6f}"
            )

    # Son jenerasyon sonrası en iyi birey
    best, best_fit, best_x = best_individual(population)
    print("\n=== Sonuç ===")
    print(f"En iyi x = {best_x:.6f}")
    print(f"f(x) = {best_x*best_x:.8f}")
    print(f"Fitness = {best_fit:.8f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pop-size", type=int, default=50)
    parser.add_argument("--bits", type=int, default=16)
    parser.add_argument("--generations", type=int, default=50)
    parser.add_argument("--crossover-prob", type=float, default=0.8)
    parser.add_argument("--mutation-prob", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    run_ga(
        pop_size=args.pop_size,
        bits=args.bits,
        generations=args.generations,
        crossover_prob=args.crossover_prob,
        mutation_prob=args.mutation_prob,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
