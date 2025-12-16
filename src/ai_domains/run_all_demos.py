from __future__ import annotations

import sys


def main():
    # Beklenen çağrı şekli:
    # python src/ai_domains/run_all_demos.py classical_ml
    # python src/ai_domains/run_all_demos.py vision
    # ...
    if len(sys.argv) != 2:
        print("Kullanım: python run_all_demos.py <module>")
        print("Modüller: classical_ml, vision, rl, rules, search, fuzzy, evolutionary, swarm")
        sys.exit(1)

    module = sys.argv[1]

    if module == "classical_ml":
        from src.ai_domains.classical_ml.train_classic_ml import main as run
    elif module == "vision":
        from src.ai_domains.vision.vision_demo import main as run
    elif module == "rl":
        from src.ai_domains.rl.rl_cartpole import main as run
    elif module == "rules":
        from src.ai_domains.expert_systems.rule_engine import main as run
    elif module == "search":
        from src.ai_domains.planning_search.search_algos import main as run
    elif module == "fuzzy":
        from src.ai_domains.fuzzy_logic.fuzzy_controller import main as run
    elif module == "evolutionary":
        from src.ai_domains.evolutionary.genetic_algorithm_demo import main as run
    elif module == "swarm":
        from src.ai_domains.swarm_intelligence.pso_demo import main as run
    else:
        print(f"Bilinmeyen modül: {module}")
        sys.exit(1)

    run()


if __name__ == "__main__":
    main()
