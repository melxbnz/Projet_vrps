"""
Étape 8 — Regroupement (Main)
Assemble toutes les étapes et lance l'optimisation.
"""
from .contracts import Instance, Solution
from .instance_loader import load_instance
from .initial_solution import build_initial_solution
from .evaluation import evaluate_solution
from .optimization_loop import optimization_loop

def run_pipeline(path_or_name: str = "A-n32-k5.vrp"):
    # 1) Chargement
    instance: Instance = load_instance(path_or_name)
    # 2) Construction initiale
    init: Solution = build_initial_solution(instance)
    # 3) Évaluation initiale
    init = evaluate_solution(init, instance)
    # 4) Optimisation
    history = optimization_loop(instance, init, max_iter=800)
    return init, history

if __name__ == "__main__":
    run_pipeline()
