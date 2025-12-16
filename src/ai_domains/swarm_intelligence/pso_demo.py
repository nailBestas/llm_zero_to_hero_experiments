from __future__ import annotations

import argparse
import math
import random
from typing import List, Tuple


# Amaç fonksiyonu:
# f(x, y) = x^2 + y^2  (2D)  -> global minimum (0, 0)
# PSO ile bu minimumu yaklaşık bulacağız.


def objective(x: float, y: float) -> float:
    return x * x + y * y


class Particle:
    def __init__(self, bounds: Tuple[float, float]):
        xmin, xmax = bounds
        self.position = [
            random.uniform(xmin, xmax),
            random.uniform(xmin, xmax),
        ]  # [x, y]
        self.velocity = [
            random.uniform(-(xmax - xmin), (xmax - xmin)) * 0.1,
            random.uniform(-(xmax - xmin), (xmax - xmin)) * 0.1,
        ]
        self.best_position = self.position.copy()
        self.best_value = objective(*self.position)

    def update_velocity(
        self,
        global_best: List[float],
        w: float,
        c1: float,
        c2: float,
    ):
        r1 = random.random()
        r2 = random.random()
        for i in range(2):
            cognitive = c1 * r1 * (self.best_position[i] - self.position[i])
            social = c2 * r2 * (global_best[i] - self.position[i])
            self.velocity[i] = w * self.velocity[i] + cognitive + social

    def update_position(self, bounds: Tuple[float, float]):
        xmin, xmax = bounds
        for i in range(2):
            self.position[i] += self.velocity[i]
            # Basit sınır kontrolü
            if self.position[i] < xmin:
                self.position[i] = xmin
                self.velocity[i] *= -0.5
            elif self.position[i] > xmax:
                self.position[i] = xmax
                self.velocity[i] *= -0.5

        val = objective(*self.position)
        if val < self.best_value:
            self.best_value = val
            self.best_position = self.position.copy()


def run_pso(
    num_particles: int = 30,
    iterations: int = 50,
    bounds: Tuple[float, float] = (-10.0, 10.0),
    w: float = 0.7,
    c1: float = 1.5,
    c2: float = 1.5,
    seed: int = 42,
):
    random.seed(seed)

    particles = [Particle(bounds) for _ in range(num_particles)]

    # Başlangıç global en iyi
    gbest_pos = particles[0].best_position.copy()
    gbest_val = particles[0].best_value
    for p in particles:
        if p.best_value < gbest_val:
            gbest_val = p.best_value
            gbest_pos = p.best_position.copy()

    print("=== Particle Swarm Optimization: minimize x^2 + y^2 ===")
    print(f"Initial global best: f({gbest_pos[0]:.4f}, {gbest_pos[1]:.4f}) = {gbest_val:.6f}")

    for it in range(1, iterations + 1):
        for p in particles:
            p.update_velocity(gbest_pos, w=w, c1=c1, c2=c2)
            p.update_position(bounds)

        # Global en iyiyi güncelle
        for p in particles:
            if p.best_value < gbest_val:
                gbest_val = p.best_value
                gbest_pos = p.best_position.copy()

        if it == 1 or it % 10 == 0 or it == iterations:
            print(
                f"Iter {it:3d}: "
                f"gbest = ({gbest_pos[0]:.5f}, {gbest_pos[1]:.5f}), "
                f"f = {gbest_val:.8f}"
            )

    print("\n=== Sonuç ===")
    print(f"En iyi pozisyon: x = {gbest_pos[0]:.6f}, y = {gbest_pos[1]:.6f}")
    print(f"f(x, y) = {gbest_val:.8f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--particles", type=int, default=30)
    parser.add_argument("--iterations", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    run_pso(
        num_particles=args.particles,
        iterations=args.iterations,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
