#RROOOOOOMMMMMMAAAAAAAIIIIIIIIN

from typing import List, Dict, Tuple
from .contracts import Instance, Solution
from .evaluation import check_feasibility, evaluate_solution
import math
import sys


# --- IMPORTS DE L'ARCHITECTURE ---
# Ces imports (relatifs) sont corrects pour
# quand 'main.py' ou 'alns.py' importeront ce module.
# Ils échoueront si on lance le script directement, c'est normal.
try:
    from .contracts import Instance, Solution, Route
    from .evaluation import evaluate_solution, check_feasibility
except ImportError:
    # Ce fallback est pour le cas où __name__ == "__main__"
    # Le vrai import se fera dans le bloc de test ci-dessous
    pass

Route = List[int]

def build_clarke_wright_solution(instance: Instance) -> Solution:
    n = len(instance.demand)
    D = instance.distance_matrix
    routes: List[Route] = [[0, i, 0] for i in range(1, n)]

    economies = []
    for i in range(1, n):
        for j in range(i + 1, n):
            s = D[0][i] + D[0][j] - D[i][j]
            economies.append((s, i, j))
    economies.sort(reverse=True)

    for s, i, j in economies:
        ri = rj = None
        for idx, r in enumerate(routes):
            if i in r[1:-1]:
                ri = idx
            if j in r[1:-1]:
                rj = idx
        if ri is None or rj is None or ri == rj:
            continue

        route_i, route_j = routes[ri], routes[rj]
        if route_i[-2] == i and route_j[1] == j:
            merged = route_i[:-1] + route_j[1:]
            # Vérification de la faisabilité de la fonction précédente
            if check_feasibility(merged, instance):
                for idx in sorted([ri, rj], reverse=True):
                    routes.pop(idx)
                routes.append(merged)

    solution = Solution(routes=routes)
    evaluate_solution(solution, instance)
    return solution


# --- [BLOC DE TEST CORRIGÉ] ---

if __name__ == "__main__":
    """
    Section de tests exécutable via : python -m src.initial_solution
    """
    print("🚀 Lancement des tests rapides pour src/initial_solution.py...")
    import sys # Import manquant
    import math # Import manquant

    # --- Dépendances de test ---
    # Quand on lance avec "python -m src.module",
    # la racine (Projet_vrp) est dans le path.
    # Les imports doivent être absolus depuis la racine.
    try:
        from src.contracts import Instance, Solution
        from src.evaluation import evaluate_solution, check_feasibility
    except ImportError:
        print("❌ ÉCHEC: Impossible d'importer 'src.contracts' ou 'src.evaluation'.")
        print("   Assurez-vous d'être à la racine 'Projet_vrp' et de lancer avec 'python -m ...'")
        sys.exit(1)

    # --- Données de test ---
    DM_test = [
        [0.0, 10.0, 10.0, 10.0, 10.0], # 0
        [10.0, 0.0, 2.0, 8.0, 8.0], # 1
        [10.0, 2.0, 0.0, 8.0, 8.0], # 2
        [10.0, 8.0, 8.0, 0.0, 2.0], # 3
        [10.0, 8.0, 8.0, 2.0, 0.0]  # 4
    ]
    
    tiny_instance = Instance(
        name="tiny_5_nodes_cw",
        distance_matrix=DM_test,
        demand=[0, 1, 1, 1, 1], # 4 clients
        capacity=3, # Peut prendre 3 clients
        Kmax=4
    )
    
    # --- Action ---
    solution = build_clarke_wright_solution(tiny_instance)
    
    print(f"\nSolution initiale trouvée pour '{tiny_instance.name}':")
    print(f"Routes: {solution.routes}")
    print(f"Coût: {solution.cost}")
    print(f"Faisable: {solution.feasible}")
    
    # --- Vérifications ---
    # Coût attendu = (10+2+10) + (10+2+10) = 44.0
    
    assert solution.feasible == True, "La solution devrait être faisable"
    assert len(solution.routes) == 2, f"Attendu 2 routes, obtenu {len(solution.routes)}"
    assert math.isclose(solution.cost, 44.0), f"Attendu coût 44.0, obtenu {solution.cost}"
    
    print("\n🎉 Tous les tests de solution initiale (Clarke & Wright) ont réussi!")