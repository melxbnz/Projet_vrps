#!/usr/bin/env python
# -*- coding: utf-8 -*-
import math
from typing import List, Tuple
import sys

# from contracts import Solution, Instance  # type: ignore

# --- IMPORTS DE L'ARCHITECTURE ---
try:
    # 1. On importe les "Contrats"
    from .contracts import Instance, Solution, Route
    # 2. On importe le "Juge" (pour les deltas et la faisabilité)
    # Note : delta_cost_two_opt est dans evaluation.py !
    from .evaluation import check_feasibility, delta_cost_two_opt

except ImportError:
    # ... (Stubs de fallback pour les tests) ...
    print("Erreur: Impossible d'importer contracts/evaluation dans neighborhoods.py", file=sys.stderr)
    from dataclasses import dataclass, field
    from typing import Optional, Dict
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
        def copy(self): import copy; return copy.deepcopy(self)

    def check_feasibility(route: Route, instance: Instance) -> bool:
        return sum(instance.demand[n] for n in route[1:-1]) <= instance.capacity
    def delta_cost_two_opt(route: Route, i: int, j: int, dm: List[List[float]]) -> float:
        return 1.0 # STUB


# 1. TWO-OPT (inversion d’un segment [i..j])

def apply_two_opt(sol: Solution, k: int, i: int, j: int):
    """
    Applique le mouvement Two-Opt sur la route k en inversant le segment entre 
    les indices i et j (inclus).
    
    i et j sont des index DANS la liste complète [0, c1, ..., cn, 0].
    """
    r = sol.routes[k]
    # Validation pour s'assurer qu'on ne touche pas aux dépôts (0 et len(r)-1)
    if not (0 < i < j < len(r) - 1):
        raise ValueError(f"Index i ({i}) ou j ({j}) invalide pour two_opt sur route {k}. Doit être 1 <= i < j < len(route)-1.")

    # Inverse la sous-liste de clients (les indices i et j sont directs)
    r[i:j+1] = list(reversed(r[i:j+1]))


# 2. RELOCATE (déplacer un client)

def apply_relocate(sol: Solution, k1: int, i: int, k2: int, j: int):
    """
    Déplace le client à l'index i de la route k1 et l'insère 
    à la position j de la route k2.
    
    i et j sont des index DANS la liste complète [0, c1, ..., cn, 0].
    """
    r1, r2 = sol.routes[k1], sol.routes[k2]
    
    # Validation de l'indice de suppression 'i'
    if not (0 < i < len(r1) - 1):
        raise ValueError(f"Index 'i' ({i}) invalide pour relocate (suppression) sur route {k1}. Doit être 1 <= i < len(r1)-1.")
    
    # Validation de l'indice d'insertion 'j'
    if not (0 < j < len(r2) or (j == len(r2) and r2[-1] == 0)):
         # 'j' peut aller jusqu'à l'index du dépôt final (avant l'insertion)
         raise ValueError(f"Index 'j' ({j}) invalide pour relocate (insertion) sur route {k2}. Doit être 1 <= j <= len(r2)-1.")


    # Pop et insert sur la liste complète, les indices sont directs
    u = r1.pop(i)
    r2.insert(j, u)


# 3. SWAP (échange de deux clients)

def apply_swap(sol: Solution, k1: int, i: int, k2: int, j: int):
    """
    Échange le client à l'index i de la route k1 avec le client 
    à l'index j de la route k2.
    
    i et j sont des index DANS la liste complète [0, c1, ..., cn, 0].
    """
    r1, r2 = sol.routes[k1], sol.routes[k2]
    
    # Validation pour s'assurer qu'on ne touche pas aux dépôts (0 et len(r)-1)
    if not (0 < i < len(r1) - 1):
        raise ValueError(f"Index 'i' ({i}) invalide pour swap sur route {k1}. Doit être 1 <= i < len(r1)-1.")
    if not (0 < j < len(r2) - 1):
        raise ValueError(f"Index 'j' ({j}) invalide pour swap sur route {k2}. Doit être 1 <= j < len(r2)-1.")

    # Échange direct
    r1[i], r2[j] = r2[j], r1[i]


# Fonctions de calcul de Delta-Cost (Delta)


# 2. RELOCATE (déplacer un client)

def delta_relocate(sol: Solution, instance: Instance, k1: int, i: int, k2: int, j: int) -> float:
    """
    Calcule le changement de coût résultant du déplacement du client 
    à l'index i de la route k1 vers la position j de la route k2.
    
    i et j sont des index DANS la liste complète [0, c1, ..., cn, 0].
    """
    r1, r2 = sol.routes[k1], sol.routes[k2]
    
    # --- VALIDATION DES INDICES ---
    if not (0 < i < len(r1) - 1):
        return math.inf # Index invalide pour suppression
    if not (0 < j <= len(r2) - 1):
        return math.inf # Index invalide pour insertion (doit être avant le 0 final)
    # --- FIN VALIDATION ---
    
    c = instance.distance_matrix
    Q = instance.capacity
    q = instance.demand

    u = r1[i]
    
    # Vérification de la contrainte de capacité pour un déplacement inter-route
    if k1 != k2:
        # --- CORRECTION DU BOGUE : Somme uniquement les clients (r2[1:-1]) ---
        current_demand_r2 = sum(q[x] for x in r2[1:-1]) 
        if current_demand_r2 + q[u] > Q: 
            return math.inf

    # Suppression dans la route k1 (a1->u->b1 remplacé par a1->b1)
    a1 = r1[i-1]
    b1 = r1[i+1] 
    
    # Coût retiré : -(a1->u) - (u->b1) + (a1->b1)
    rm = -c[a1][u] - c[u][b1] + c[a1][b1]
    
    # Insertion dans la route k2 (a2->b2 remplacé par a2->u->b2)
    a2 = r2[j-1]
    b2 = r2[j]
    
    # Coût inséré : -(a2->b2) + (a2->u) + (u->b2)
    ins = -c[a2][b2] + c[a2][u] + c[u][b2]
    
    return rm + ins

# 3. SWAP (échange de deux clients)

def delta_swap(sol: Solution, instance: Instance, k1: int, i: int, k2: int, j: int) -> float:
    """
    Calcule le changement de coût résultant de l'échange du client 
    à l'index i de la route k1 (u) avec le client à l'index j de la route k2 (v).
    
    i et j sont des index DANS la liste complète [0, c1, ..., cn, 0].
    """
    r1, r2 = sol.routes[k1], sol.routes[k2]
    
    # --- VALIDATION DES INDICES ---
    if not (0 < i < len(r1) - 1):
        return math.inf
    if not (0 < j < len(r2) - 1):
        return math.inf
    # --- FIN VALIDATION ---
    
    c = instance.distance_matrix
    Q = instance.capacity
    q = instance.demand
    
    u, v = r1[i], r2[j]
    
    # Échange inutile si même position
    if k1 == k2 and i == j:
        return math.inf
        
    # Vérification des contraintes de capacité pour un échange inter-route
    if k1 != k2:
        # --- CORRECTION DU BOGUE : Somme uniquement les clients (r[1:-1]) ---
        # Route k1: retire u, ajoute v
        if sum(q[x] for x in r1[1:-1]) - q[u] + q[v] > Q: 
            return math.inf
            
        # Route k2: retire v, ajoute u
        if sum(q[x] for x in r2[1:-1]) - q[v] + q[u] > Q: 
            return math.inf

    # Arcs autour de u dans r1
    a1 = r1[i-1]
    b1 = r1[i+1]
    
    # Arcs autour de v dans r2
    a2 = r2[j-1]
    b2 = r2[j+1]

    # Cas spécial: u et v sont voisins sur la même route (k1=k2 et |i-j|=1)
    if k1 == k2 and abs(i - j) == 1:
        # Simplification de la formule : seul le tronçon a1-u-v-b2 devient a1-v-u-b2
        if i > j: # S'assurer que i est l'indice le plus petit
            i, j = j, i
            u, v = v, u
            # a1 est le prédécesseur de l'arc échangé, b2 est le successeur
            a1 = r1[i-1] # Prédécesseur de l'arc échangé
            b2 = r1[j+1] # Successeur de l'arc échangé

        old = c[a1][u] + c[u][v] + c[v][b2]
        new = c[a1][v] + c[v][u] + c[u][b2]
        
    else:
        # Cas général : arcs disjoints (ou k1 != k2)
        old = c[a1][u] + c[u][b1] + c[a2][v] + c[v][b2]
        new = c[a1][v] + c[v][b1] + c[a2][u] + c[u][b2]

    return new - old

# 4. CALCUL DU DELTA-COST RAPIDE (wrapper)

def delta_cost(sol: Solution, instance: Instance, move_type: str, *args) -> float:
    """
    Wrapper pour calculer le delta_cost en fonction du type de mouvement.
    """
    # NOTE: delta_two_opt doit être importé de evaluation.py dans l'environnement cible.
    
    if move_type == "relocate":  
        if len(args) != 4: return math.inf
        return delta_relocate(sol, instance, *args)
        
    if move_type == "swap":      
        if len(args) != 4: return math.inf
        return delta_swap(sol, instance, *args)
        
    return math.inf


# --- [BLOC DE TEST CORRIGÉ] ---

if __name__ == "__main__":
    """
    Section de tests exécutable via : python -m src.neighborhoods
    """
    print("🚀 Lancement des tests rapides pour src/neighborhoods.py...")
    import sys
    import math

    # --- Dépendances de test ---
    try:
        from src.contracts import Instance, Solution
        from src.evaluation import evaluate_solution
    except ImportError:
        print("❌ ÉCHEC: Impossible d'importer 'src.contracts' ou 'src.evaluation'.")
        print("   Assurez-vous d'être à la racine 'Projet_vrp' et de lancer avec 'python -m ...'")
        sys.exit(1)

    # --- Données de test ---
    DM_test = [
        [0.0, 10.0, 10.0, 100.0, 100.0], # 0
        [10.0, 0.0, 2.0, 100.0, 100.0], # 1
        [10.0, 2.0, 0.0, 100.0, 100.0], # 2
        [100.0, 100.0, 100.0, 0.0, 5.0],  # 3
        [100.0, 100.0, 100.0, 5.0, 0.0]   # 4
    ]
    tiny_instance = Instance(
        name="test_moves",
        distance_matrix=DM_test,
        demand=[0, 1, 1, 1, 1],
        capacity=3
    )
    
    # Solution de base: [0,1,2,0] (coût 22) et [0,3,4,0] (coût 205)
    # Coût total = 227.0
    sol_base = Solution(routes=[[0, 1, 2, 0], [0, 3, 4, 0]])
    evaluate_solution(sol_base, tiny_instance)
    cost_initial = sol_base.cost
    
    # --- [CORRECTION ICI] ---
    assert math.isclose(cost_initial, 227.0)
    print(f"Solution initiale (coût {cost_initial:.2f}) chargée.")

    # --- 1. Test Relocate ---
    print("\n--- Test 1: Relocate (move client 2 -> route 1) ---")
    sol_test = sol_base.copy()
    
    # k1=0, i=2 (client 2)
    # k2=1, j=1 (entre 0 et 3)
    
    try:
        k1, i = 0, 2 
        k2, j = 1, 1 

        delta_calc = delta_relocate(sol_test, tiny_instance, k1, i, k2, j)
        print(f"Delta (Relocate) calculé: {delta_calc:.2f}")

        apply_relocate(sol_test, k1, i, k2, j)
        
        evaluate_solution(sol_test, tiny_instance)
        cost_new_reel = sol_test.cost
        print(f"Nouveau coût (réel): {cost_new_reel:.2f}")
        
        cost_new_attendu = cost_initial + delta_calc
        print(f"Nouveau coût (attendu): {cost_new_attendu:.2f}")
        
        assert math.isclose(cost_new_reel, cost_new_attendu), "Delta Relocate est INCOHÉRENT"
        print("✅ Delta Relocate cohérent.")

    except Exception as e:
        print(f"❌ ÉCHEC: Erreur lors du test Relocate: {e}")

    # --- 2. Test Swap ---
    print("\n--- Test 2: Swap (client 1 <-> client 3) ---")
    sol_test = sol_base.copy()
    
    k1, i = 0, 1 
    k2, j = 1, 1 
    
    try:
        delta_calc = delta_swap(sol_test, tiny_instance, k1, i, k2, j)
        print(f"Delta (Swap) calculé: {delta_calc:.2f}")

        apply_swap(sol_test, k1, i, k2, j)
        
        evaluate_solution(sol_test, tiny_instance)
        cost_new_reel = sol_test.cost
        print(f"Nouveau coût (réel): {cost_new_reel:.2f}")
        
        cost_new_attendu = cost_initial + delta_calc
        print(f"Nouveau coût (attendu): {cost_new_attendu:.2f}")

        assert math.isclose(cost_new_reel, cost_new_attendu), "Delta Swap est INCOHÉRENT"
        print("✅ Delta Swap cohérent.")
        
    except Exception as e:
        print(f"❌ ÉCHEC: Erreur lors du test Swap: {e}")

    print("\n🎉 Tous les tests de voisinages ont réussi!")