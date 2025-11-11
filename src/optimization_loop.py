# MEL#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Module pour la boucle d'optimisation (VND/ALNS).
(Implémentation de l'Étape 7)

Version initiale: utilise un 'mock move' (2-opt aléatoire)
en attendant l'implémentation complète d'ALNS.
"""

import random
import copy
import sys
from typing import List, Dict, Optional

# Importation des contrats et fonctions d'évaluation
try:
    # Import relatif si ce module est importé (cas normal)
    from .contracts import Instance, Solution
    from .evaluation import evaluate_solution
except ImportError:
    # Fallback pour exécution en tant que script (pour les tests __main__)
    # Cela suppose que contracts.py et evaluation.py sont au même niveau
    try:
        from contracts import Instance, Solution, Route
        from evaluation import evaluate_solution
    except ImportError:
        print(
            "Erreur: Impossible d'importer 'contracts' ou 'evaluation'. "
            "Assurez-vous qu'ils sont accessibles.",
            file=sys.stderr
        )
        # Création de stubs pour que le fichier se charge
        from dataclasses import dataclass, field
        NodeId = int
        Route = List[NodeId]
        @dataclass
        class Instance:
            name: str
            distance_matrix: List[List[float]]
            demand: List[int]
            capacity: int
            ready_time: Optional[List[float]] = None
            due_time: Optional[List[float]] = None
            service_time: Optional[List[float]] = None
            Kmax: Optional[int] = None
            Tmax: Optional[float] = None
        @dataclass
        class Solution:
            routes: List[Route] = field(default_factory=list)
            cost: float = float("inf")
            feasible: bool = False
            meta: Dict[str, float] = field(default_factory=dict)
        
        def evaluate_solution(sol: Solution, instance: Instance) -> Solution:
            print("Utilisation d'un 'evaluate_solution' STUB pour les tests.")
            sol.cost = 0.0
            sol.feasible = True
            for r in sol.routes:
                sol.cost += (len(r)-1) * 10
            return sol


# --- Helpers (Mouvement Mock) ---

#:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::JUSTE POUR LES TESTES à SUUPRIMER QAUND Y'ARA ALNS


def _apply_mock_move(solution: Solution) -> Solution:
    """
    Applique un mouvement 2-opt aléatoire sur une route valide.
    Retourne une NOUVELLE solution (copiée), NON ÉVALUÉE.
    Si aucun mouvement n'est possible, retourne la copie.
    
    Note: ne prend pas l'instance, car le 2-opt ne dépend que de la
    structure de la route. L'évaluation se fait dans la boucle.
    """
    neighbor = copy.deepcopy(solution)
    
    # 1. Trouver les routes éligibles pour un 2-opt (longueur >= 4)
    # [0, i, j, 0] (len 4) -> 2-opt -> [0, j, i, 0]. Indices valides [1, 2]
    eligible_routes_idx = [
        idx for idx, route in enumerate(neighbor.routes)
        if len(route) >= 4 
    ]
    
    if not eligible_routes_idx:
        return neighbor # Pas de mouvement possible

    # 2. Choisir une route et des indices
    route_idx = random.choice(eligible_routes_idx)
    route = neighbor.routes[route_idx]
    
    # Indices valides : 1 à len(route)-2
    try:
        idx_range = range(1, len(route) - 1)
        if len(idx_range) < 2:
            return neighbor # Ne devrait pas arriver avec len >= 4
        
        # Choisir 2 indices différents
        i, j = sorted(random.sample(idx_range, 2))
    
    except ValueError:
        # Cas improbable
        return neighbor

    # 3. Appliquer le 2-opt (inversion en place sur la copie)
    neighbor.routes[route_idx][i : j + 1] = \
        neighbor.routes[route_idx][i : j + 1][::-1]
    
    # Retourne la solution modifiée, mais non évaluée
    return neighbor
#:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::JUSTE POUR LES TESTES à SUUPRIMER QAUND Y'ARA ALNS

# --- Boucle d'optimisation ---

def optimization_loop(
    instance: Instance,
    init_solution: Solution,
    max_iter: int = 800,
    patience: int = 100,
    seed: int = 42,
) -> dict[str, list[float]]:
    """
    Exécute une boucle d'optimisation (descente locale simple).
    
    Utilise un 'mock move' (2-opt aléatoire) pour générer des voisins
    en attendant une implémentation d'ALNS.
    
    Retourne:
      history = {"iter": [...], "cost_current": [...], "cost_best": [...]}
    """
    
    # 1. Initialisation
    random.seed(seed)
    
    history: Dict[str, List[float]] = {
        "iter": [], 
        "cost_current": [], 
        "cost_best": []
    }
    
    # Garantir l'immuabilité de l'input
    current_solution = copy.deepcopy(init_solution)
    
    # S'assurer que la solution initiale est évaluée
    # (le brief indique qu'elle l'est, mais vérifions)
    if current_solution.cost == float("inf"):
        evaluate_solution(current_solution, instance)

    best_solution = copy.deepcopy(current_solution)
    
    # Si la solution initiale n'est pas faisable, best_cost est infini
    if not best_solution.feasible:
        best_solution.cost = float("inf")

    no_improve_count = 0

    # 2. Boucle principale
    for i in range(max_iter):
        
        # 2a. Enregistrer l'état (avant le mouvement)
        history["iter"].append(float(i))
        history["cost_current"].append(current_solution.cost)
        history["cost_best"].append(best_solution.cost)

        # 2b. Générer un voisin (Mock Move)
        # TODO: Remplacer par un appel à alns_step ou VND quand dispo
        #
        # if ALNS_AVAILABLE:
        #   neighbor_solution = alns_step(current_solution, instance, ...)
        # else:
        #   neighbor_solution = _apply_mock_move(current_solution)
        
        neighbor_solution = _apply_mock_move(current_solution) ##:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::JUSTE POUR LES TESTES à SUUPRIMER QAUND Y'ARA ALNS

        
        # Évaluer le voisin (en place)
        evaluate_solution(neighbor_solution, instance)

        # 2c. Règle d'acceptation (Descente simple)
        is_accepted = False
        if neighbor_solution.cost < current_solution.cost:
            current_solution = neighbor_solution # neighbor est déjà une deepcopy
            is_accepted = True

        # 2d. Critère d'aspiration (Mise à jour du 'best')
        if neighbor_solution.feasible and neighbor_solution.cost < best_solution.cost:
            best_solution = copy.deepcopy(neighbor_solution)
            no_improve_count = 0
            
            # Si cette solution 'best' n'avait pas été acceptée
            # (ex: current était infaisable), on l'accepte quand même
            if not is_accepted:
                current_solution = best_solution
                is_accepted = True
                
        elif is_accepted:
             # On a accepté (meilleur que current) mais pas meilleur que 'best'
             no_improve_count += 1
        else:
             # Mouvement non accepté
            no_improve_count += 1
        
        # 2e. Critère d'arrêt (Patience)
        if no_improve_count >= patience:
            # print(f"Arrêt anticipé à l'itération {i} (patience atteinte).")
            break
    
    # 3. Retourner l'historique
    return history


# --- Tests Rapides (Quick Check) ---

if __name__ == "__main__":
    """
    Section de tests exécutable via : python -m src.optimization_loop
    (Nécessite contracts.py et evaluation.py au même niveau ou PYTHONPATH)
    """
    print("🚀 Lancement des tests rapides pour src/optimization_loop.py...")
    
    # 1. Dépendances de test (besoin d'evaluation.py et contracts.py)
    # On suppose qu'ils sont au même niveau
    
    # 2. Création d'une mini-instance (inline)
    # 5 nœuds, tous très proches (sauf du dépôt)
    DM_test = [
        [0.0, 10.0, 10.0, 10.0, 10.0],
        [10.0, 0.0, 2.0, 2.0, 2.0],
        [10.0, 2.0, 0.0, 2.0, 2.0],
        [10.0, 2.0, 2.0, 0.0, 2.0],
        [10.0, 2.0, 2.0, 2.0, 0.0]
    ]
    
    tiny_instance = Instance(
        name="tiny_5_nodes",
        distance_matrix=DM_test,
        demand=[0, 1, 1, 1, 1],
        capacity=3
    )
    
    # 3. Création d'une solution initiale (triviale)
    # 4 routes, une par client
    sol_init = Solution(
        routes=[
            [0, 1, 0],
            [0, 2, 0],
            [0, 3, 0],
            [0, 4, 0]
        ]
    )
    
    # Évaluation initiale (doit être faite avant la boucle)
    sol_init = evaluate_solution(sol_init, tiny_instance)
    # Coût = (10+10) * 4 = 80
    # Faisable = True (Cap 1 <= 3 pour chaque route)
    
    print(f"Solution initiale: Coût={sol_init.cost}, Faisable={sol_init.feasible}")
    assert sol_init.cost == 80.0
    assert sol_init.feasible == True

    # 4. Exécution de la boucle
    MAX_ITER_TEST = 20
    PATIENCE_TEST = 10
    history = optimization_loop(
        tiny_instance, 
        sol_init, 
        max_iter=MAX_ITER_TEST,
        patience=PATIENCE_TEST
    )
    
    print(f"Boucle exécutée sur {len(history['iter'])} itérations.")
    
    # 5. Vérification des résultats
    assert "iter" in history
    assert "cost_current" in history
    assert "cost_best" in history
    print(f"Clés de l'historique : {list(history.keys())} ✅")
    
    # La boucle doit s'exécuter jusqu'au bout (MAX_ITER_TEST)
    # car le mock move (2-opt) ne peut pas améliorer ces routes
    # [0,1,0] len 3 < 4 -> 2-opt impossible.
    # Le no_improve_count ne devrait pas être déclenché car aucun move n'est fait.
    
    # Modifions la solution initiale pour permettre le 2-opt
    sol_init_2opt = Solution(
        routes=[
            [0, 1, 2, 0], # Coût 10+2+10 = 22. Dem 1+1=2.
            [0, 3, 4, 0]  # Coût 10+2+10 = 22. Dem 1+1=2.
        ]
    )
    sol_init_2opt = evaluate_solution(sol_init_2opt, tiny_instance)
    # Coût total = 44. Faisable.
    print(f"\nTest 2 (avec routes 2-opt-ables): Coût init={sol_init_2opt.cost}")

    history2 = optimization_loop(
        tiny_instance, 
        sol_init_2opt, 
        max_iter=MAX_ITER_TEST,
        patience=PATIENCE_TEST
    )

    print(f"Boucle 2 exécutée sur {len(history2['iter'])} itérations.")
    assert len(history2["iter"]) > 0
    # Dans cette instance symétrique, 2-opt (ex: [0,2,1,0]) a le même coût (22).
    # La descente n'acceptera jamais, 'best' ne bougera pas.
    # La boucle devrait s'arrêter par patience.
    assert len(history2["iter"]) == PATIENCE_TEST
    print(f"Taille de l'historique : {len(history2['iter'])} (Attendu: {PATIENCE_TEST}) ✅")
    
    assert history2["cost_best"][-1] == 44.0
    print(f"Coût final (best): {history2['cost_best'][-1]} (Attendu: 44.0) ✅")
    
    print("\n🎉 Tous les tests d'optimisation ont réussi!")