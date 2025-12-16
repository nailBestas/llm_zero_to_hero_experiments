import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--module", required=True,
                        choices=[
                            "classical_ml",
                            "vision",
                            "rl",
                            "rules",
                            "search",
                            "fuzzy",
                            "evolutionary",
                            "swarm",
                        ])
    args = parser.parse_args()

    if args.module == "classical_ml":
        from src.ai_domains.classical_ml.train_classic_ml import main as run
    elif args.module == "vision":
        from src.ai_domains.vision.vision_demo import main as run
    elif args.module == "rl":
        from src.ai_domains.rl.rl_cartpole import main as run
    elif args.module == "rules":
        from src.ai_domains.expert_systems.rule_engine import main as run
    elif args.module == "search":
        from src.ai_domains.planning_search.search_algos import main as run
    elif args.module == "fuzzy":
        from src.ai_domains.fuzzy_logic.fuzzy_controller import main as run
    elif args.module == "evolutionary":
        from src.ai_domains.evolutionary.genetic_algorithm_demo import main as run
    elif args.module == "swarm":
        from src.ai_domains.swarm_intelligence.pso_demo import main as run
    else:
        raise ValueError("Unknown module")

    run()

if __name__ == "__main__":
    main()
