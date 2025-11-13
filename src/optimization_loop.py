#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Module pour la boucle d'optimisation (Pilote ALNS).
(Implémentation de l'Étape 7, version finale)

Ce module agit comme un "pilote" (driver) :
- Il initialise la métaheuristique (ex: ALNS).
- Il gère la boucle principale (max_iter, patience).
- Il collecte l'historique des coûts (current, best).
- Il appelle l'itération de la métaheuristique (ex: alns.run_iteration()).
"""

import random
import copy
import sys
import numpy as np  # Requis par ALNS pour les probabilités
from typing import List, Dict, Optional

# Importation des contrats (la *seule* source de vérité)
try:
    from .contracts import Instance, Solution
    # On importe notre propre évaluateur (le "juge")
    from .evaluation import evaluate_solution
    
    # --- LA MODIFICATION CLÉ ---
    # On importe la classe ALNS d'Olivier
    from .alns import ALNS
    
except ImportError:
    # ... (Bloc de stubs pour les tests, au cas où) ...
    print(
        "Erreur: Impossible d'importer 'contracts', 'evaluation' ou 'alns'. "
        "Assurez-vous qu'ils existent et sont compatibles.",
        file=sys.stderr
    )
    from dataclasses import dataclass, field
    NodeId = int
    Route = List[NodeId]
    @dataclass
    class Instance: ... # Stub
    @dataclass
    class Solution:
        routes: List[Route] = field(default_factory=list)
        cost: float = float("inf")
        feasible: bool = False
        meta: Dict[str, float] = field(default_factory=dict)
        def copy(self): return copy.deepcopy(self) # ALNS a besoin de .copy()
    
    # STUB pour la classe ALNS si elle n'est pas trouvée
    class ALNS:
        def __init__(self, instance: Instance, initial_solution: Solution):
            print("Utilisation d'un ALNS STUB.")
            self.current_solution = initial_solution.copy()
            self.best_solution = initial_solution.copy()
            self.best_solution.cost = initial_solution.cost
            self.best_solution.feasible = initial_solution.feasible
        def run_iteration(self) -> bool:
            print("ALNS STUB: run_iteration()")
            if self.best_solution.cost > 10:
                self.best_solution.cost -= 10 # Simule une amélioration
                self.current_solution.cost = self.best_solution.cost
                return True # A amélioré
            return False # N'a pas amélioré

    def evaluate_solution(sol: Solution, instance: Instance) -> Solution:
        print("Utilisation d'un 'evaluate_solution' STUB.")
        sol.cost = 100.0
        sol.feasible = True
        return sol


# --- Boucle d'optimisation (Version Pilote ALNS) ---

def optimization_loop(
    instance: Instance,
    init_solution: Solution,
    max_iter: int = 800,
    patience: int = 100,
    seed: int = 42,
) -> dict[str, list[float]]:
    """
    Exécute la boucle d'optimisation en pilotant la classe ALNS.
    
    Retourne:
      history = {"iter": [...], "cost_current": [...], "cost_best": [...]}
    """
    
    # 1. Initialisation
    random.seed(seed)
    np.random.seed(seed) # ALNS utilise numpy, on fixe son seed aussi
    
    history: Dict[str, List[float]] = {
        "iter": [], 
        "cost_current": [], 
        "cost_best": []
    }
    
    # S'assurer que la solution initiale est évaluée (par notre "juge")
    if init_solution.cost == float("inf"):
        evaluate_solution(init_solution, instance)

    # 2. Initialiser l'orchestrateur ALNS
    try:
        # On donne l'instance et la solution de départ à la classe ALNS
        alns_orchestrator = ALNS(instance, init_solution)
    except Exception as e:
        print(f"--- ERREUR CRITIQUE ---", file=sys.stderr)
        print(f"Impossible d'initialiser la classe ALNS.", file=sys.stderr)
        print(f"Vérifiez que 'alns.py' est compatible avec 'contracts.py'.", file=sys.stderr)
        print(f"Erreur: {e}", file=sys.stderr)
        return history

    no_improve_count = 0
    
    # Récupérer le coût initial de "best"
    best_cost_so_far = alns_orchestrator.best_solution.cost
    if not alns_orchestrator.best_solution.feasible:
        best_cost_so_far = float("inf")


    # 3. Boucle principale (pilotage de l'ALNS)
    for i in range(max_iter):
        
        # 3a. Enregistrer l'état (avant l'itération)
        current_cost = alns_orchestrator.current_solution.cost
        
        history["iter"].append(float(i))
        history["cost_current"].append(current_cost)
        history["cost_best"].append(best_cost_so_far)

        # 3b. Lancer UNE itération de l'ALNS
        # C'est elle qui fait destroy/repair/VND/accept/adapt
        try:
            # run_iteration() fait tout le travail et
            # retourne True si elle a trouvé un nouveau "best"
            best_was_improved = alns_orchestrator.run_iteration()
        
        except NotImplementedError as e:
            print(f"--- ERREUR D'EXÉCUTION ALNS ---", file=sys.stderr)
            print(f"La fonction '{e}' n'est pas implémentée dans 'alns.py' ou 'neighborhoods.py'.", file=sys.stderr)
            print("Vous devez implémenter 'generate_candidates' ou une fonction similaire.", file=sys.stderr)
            break # Arrêter la boucle
        except Exception as e:
            print(f"Erreur inconnue dans alns.run_iteration: {e}", file=sys.stderr)
            break

        # 3c. Gérer la patience (le pilote gère l'arrêt)
        if best_was_improved:
            no_improve_count = 0
            # Mettre à jour notre "meilleur coût" local
            best_cost_so_far = alns_orchestrator.best_solution.cost
        else:
            no_improve_count += 1
        
        if no_improve_count >= patience:
            # print(f"Arrêt anticipé à l'itération {i} (patience atteinte).")
            break
    
    # 4. Retourner l'historique
    return history


# --- Tests Rapides (Quick Check) ---

if __name__ == "__main__":
    """
    Section de tests exécutable via : python -m src.optimization_loop
    (Nécessite que TOUS les imports fonctionnent)
    """
    print("🚀 Lancement des tests rapides pour src/optimization_loop.py (Mode Pilote)...")
    
    # 1. Création d'une mini-instance (inline)
    # (Doit correspondre à la vraie dataclass Instance)
    try:
        from contracts import Instance, Solution
    except ImportError:
        # Le bloc stub du haut gère déjà ça
        pass 
        
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
        capacity=3,
        Kmax=4
    )
    
    # 2. Création d'une solution initiale (triviale)
    sol_init = Solution(
        routes=[ [0, 1, 0], [0, 2, 0], [0, 3, 0], [0, 4, 0] ]
    )
    
    # 3. Évaluation initiale (doit être faite avant la boucle)
    # On appelle le VRAI évaluateur (pas le STUB)
    try:
        from evaluation import evaluate_solution
        sol_init = evaluate_solution(sol_init, tiny_instance)
        print(f"Solution initiale (réelle): Coût={sol_init.cost}, Faisable={sol_init.feasible}")
        assert sol_init.cost == 80.0
        assert sol_init.feasible == True
    except ImportError:
        print("Impossible d'importer le VRAI évaluateur. Test annulé.")
        sys.exit(1)
    except AssertionError:
        print(f"Échec du calcul de coût initial. Attendu: 80.0, Obtenu: {sol_init.cost}")
        sys.exit(1)


    # 4. Exécution de la boucle
    MAX_ITER_TEST = 20
    PATIENCE_TEST = 10
    
    print(f"\nLancement de la boucle pilote ALNS pour {MAX_ITER_TEST} iters (patience {PATIENCE_TEST})...")
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
    
    # On vérifie que le coût s'est amélioré (ou est resté le même)
    final_best_cost = history["cost_best"][-1]
    print(f"Coût initial: 80.0, Coût final (best): {final_best_cost}")
    assert final_best_cost <= 80.0
    
    print("\n🎉 Tous les tests du pilote d'optimisation ont réussi!")