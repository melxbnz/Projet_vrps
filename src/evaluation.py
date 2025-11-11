#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Module pour l'évaluation des solutions VRP/VRPTW.
(Implémentation de l'Étape 4)

Fonctions fournies :
- compute_route_cost: Calcule le coût (distance) d'une route.
- check_feasibility: Vérifie la faisabilité (Capacité, TW, Tmax) d'une route.
- evaluate_solution: Évalue une solution complète (coût et faisabilité globale).
- delta_cost_two_opt: Calcule le delta-coût O(1) d'un 2-opt.
"""

from typing import List, Dict, Optional
import sys

# Importation des contrats (supposés être dans le PYTHONPATH ou un dossier parent)
try:
    # Import relatif si ce module est importé (cas normal)
    from .contracts import Instance, Solution, Route, NodeId
except ImportError:
    # Fallback pour exécution en tant que script (pour les tests __main__)
    # Cela suppose que contracts.py est dans le même répertoire
    try:
        from contracts import Instance, Solution, Route, NodeId
    except ImportError:
        print(
            "Erreur: Impossible d'importer 'contracts'. Assurez-vous que "
            "contracts.py est accessible.",
            file=sys.stderr
        )
        # Création de stubs pour que le fichier se charge pour les tests
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


# --- Fonctions d'Évaluation ---

def compute_route_cost(route: Route, distance_matrix: List[List[float]]) -> float:
    """
    Calcule le coût (distance) total d'une route.
    Précondition: la route contient au moins [0,0].
    """
    if len(route) <= 1:
        return 0.0  # Gère le cas [0] ou [] de manière robuste

    total_cost = 0.0
    for i in range(len(route) - 1):
        u = route[i]
        v = route[i+1]
        total_cost += distance_matrix[u][v]
    return total_cost

def check_feasibility(route: Route, instance: Instance) -> bool:
    """
    Vérifie la faisabilité (Capacité, TW, Tmax) d'une seule route.
    Retourne True si la route est faisable, False sinon.
    """
    
    # 1. Vérification de la capacité
    # Somme des demandes des clients (indices 1 à -2)
    total_demand = sum(instance.demand[node] for node in route[1:-1])
    if total_demand > instance.capacity:
        return False

    # 2. Vérification des TW (Time Windows) et Tmax
    current_time = 0.0
    dist_matrix = instance.distance_matrix
    ready = instance.ready_time
    due = instance.due_time
    service = instance.service_time

    for k in range(len(route) - 1):
        u = route[k]
        v = route[k+1]

        # Ajout du temps de trajet
        travel_time = dist_matrix[u][v]
        current_time += travel_time

        # Gestion des TW au nœud d'arrivée 'v'
        if ready:
            # Attente si arrivée en avance
            current_time = max(current_time, ready[v])

        if due:
            # Vérification si arrivée trop tardive
            if current_time > due[v]:
                return False

        if service:
            # Ajout du temps de service
            current_time += service[v]

    # 3. Vérification Tmax (durée max de la route)
    # current_time inclut maintenant le retour au dépôt (dernier arc)
    # et le service éventuel au dépôt (géré par la boucle)
    if instance.Tmax is not None and current_time > instance.Tmax:
        return False

    return True

def evaluate_solution(sol: Solution, instance: Instance) -> Solution:
    """
    Évalue une solution complète (coût et faisabilité).
    Met à jour la solution 'sol' en place et la retourne.
    """
    total_cost = 0.0
    all_routes_feasible = True

    if not sol.routes:
        # Cas d'une solution vide (ex: 0 clients)
        sol.cost = 0.0
        sol.feasible = True # Une solution vide est faisable
        sol.meta["nb_routes"] = 0.0
        return sol

    for route in sol.routes:
        # Une route doit au moins être [0, 0]
        if len(route) < 2:
            all_routes_feasible = False # Route invalide
            continue # Ne pas essayer de la coûter

        total_cost += compute_route_cost(route, instance.distance_matrix)
        
        if not check_feasibility(route, instance):
            all_routes_feasible = False

    sol.cost = total_cost
    sol.feasible = all_routes_feasible

    # Vérification contrainte globale Kmax
    if instance.Kmax is not None:
        if len(sol.routes) > instance.Kmax:
            sol.feasible = False
    
    sol.meta["nb_routes"] = float(len(sol.routes))
    
    return sol

def delta_cost_two_opt(
    route: Route, 
    i: int, 
    j: int, 
    distance_matrix: List[List[float]]
) -> float:
    """
    Calcule le delta-coût O(1) d'un mouvement 2-opt sur le segment [i..j].
    i et j sont les indices DANS la route (ex: route[i] et route[j]).
    
    Le 2-opt inverse route[i:j+1].
    Les arêtes (i-1, i) et (j, j+1) sont remplacées par
    les arêtes (i-1, j) et (i, j+1).
    """
    
    # Validation des indices (ne doivent pas toucher aux dépôts)
    if not (0 < i < j < len(route) - 1):
        raise ValueError(
            f"Indices 2-opt invalides (i={i}, j={j}) pour "
            f"une route de longueur {len(route)}. "
            "Doit respecter 0 < i < j < len(route)-1."
        )

    # Nœuds impliqués dans le swap
    node_A = route[i - 1] # Avant le début du segment
    node_B = route[i]     # Début du segment
    node_C = route[j]     # Fin du segment
    node_D = route[j + 1] # Après la fin du segment

    # Coût des arêtes existantes (A-B et C-D)
    old_cost = distance_matrix[node_A][node_B] + distance_matrix[node_C][node_D]
    
    # Coût des nouvelles arêtes (A-C et B-D)
    new_cost = distance_matrix[node_A][node_C] + distance_matrix[node_B][node_D]

    delta = new_cost - old_cost
    return delta


# --- Tests Rapides (Quick Check) ---

if __name__ == "__main__":
    """
    Section de tests exécutable via : python -m src.evaluation
    (Nécessite contracts.py dans le même dossier ou le PYTHONPATH)
    """
    print("🚀 Lancement des tests rapides pour src/evaluation.py...")
    
    # --- Données de test ---
    # Matrice 5x5 (0=Dépôt, 1, 2, 3, 4=Clients)
    # C'est la distance (ou le coût) pour aller du sommet i au sommet j.
    DM = [
        [0.0, 10.0, 15.0, 20.0, 25.0], # 0 dsit avec lui mm, 0 dist avec client 1, dist avec client 2 ... avec 0= depot
        [10.0, 0.0, 5.0, 30.0, 30.0], # 1
        [15.0, 5.0, 0.0, 10.0, 35.0], # 2
        [20.0, 30.0, 10.0, 0.0, 5.0],  # 3
        [25.0, 30.0, 35.0, 5.0, 0.0]   # 4
    ]
    
    # Instance VRP simple (sans TW)
    instance_vrp = Instance(
        name="test_vrp",
        distance_matrix=DM,
        demand=[0, 5, 5, 8, 10], # Demandes (D[0]=0)
        capacity=20,             # Capacité camion
        Kmax=2                   # Max 2 camions
    )
    
    # Instance VRPTW
    instance_tw = Instance(
        name="test_tw",
        distance_matrix=DM,
        demand=[0, 5, 5, 8, 10],
        capacity=20,
        ready_time=  [0.0, 10.0, 20.0, 0.0, 40.0], # Heure ouverture
        due_time=    [1000.0, 25.0, 35.0, 1000.0, 55.0], # Heure fermeture
        service_time=[0.0, 2.0, 2.0, 3.0, 1.0],  # Temps de service
        Tmax=100.0
    )
    
    # --- 1. Tests compute_route_cost ---
    print("\n--- 1. Test compute_route_cost ---")
    route1 = [0, 1, 2, 0] # 0->1 (10) + 1->2 (5) + 2->0 (15) = 30
    cost1 = compute_route_cost(route1, DM)
    assert cost1 == 30.0
    print(f"Coût [0,1,2,0] : {cost1} (Attendu: 30.0) ✅")
    
    route_min = [0, 0] # 0->0 (0)
    cost_min = compute_route_cost(route_min, DM)
    assert cost_min == 0.0
    print(f"Coût [0,0] : {cost_min} (Attendu: 0.0) ✅")

    # --- 2. Tests check_feasibility (Capacité) ---
    print("\n--- 2. Test check_feasibility (Capacité) ---")
    # Route [0,1,2,0] -> Demande = D[1]+D[2] = 5+5=10. Cap=20. OK.
    route_cap_ok = [0, 1, 2, 0]
    assert check_feasibility(route_cap_ok, instance_vrp) == True
    print(f"Faisabilité (Cap) [0,1,2,0] (10/20): {True} ✅")
    
    # Route [0,3,4,0] -> Demande = D[3]+D[4] = 8+10=18. Cap=20. OK.
    route_cap_ok2 = [0, 3, 4, 0]
    assert check_feasibility(route_cap_ok2, instance_vrp) == True
    print(f"Faisabilité (Cap) [0,3,4,0] (18/20): {True} ✅")
    
    # Route [0,1,3,4,0] -> Demande = 5+8+10=23. Cap=20. FAIL.
    route_cap_fail = [0, 1, 3, 4, 0]
    assert check_feasibility(route_cap_fail, instance_vrp) == False
    print(f"Faisabilité (Cap) [0,1,3,4,0] (23/20): {False} ✅")

    # --- 3. Tests check_feasibility (TW) ---
    print("\n--- 3. Test check_feasibility (TW/Tmax) ---")
    # R = [0, 1, 2, 0] (Instance TW)
    # 0->1: Trajet=10. Arrive t=10. ready[1]=10. OK. Service (2). Part t=12.
    # 1->2: Trajet=5. Arrive t=17. ready[2]=20. DOIT ATTENDRE.
    #       Service (2) commence t=20. Part t=22.
    # 2->0: Trajet=15. Arrive t=37. ready[0]=0. due[0]=1000. OK.
    #       Service (0) @ t=37. Part t=37.
    #       Tmax=100. OK (37 <= 100).
    route_tw_ok = [0, 1, 2, 0]
    assert check_feasibility(route_tw_ok, instance_tw) == True
    print(f"Faisabilité (TW) [0,1,2,0] (avec attente): {True} ✅")
    
    # R = [0, 2, 1, 0]
    # 0->2: Trajet=15. Arrive t=15. ready[2]=20. ATTENDRE. Service (2) @ 20. Part @ 22.
    # 2->1: Trajet=5. Arrive t=27. ready[1]=10. due[1]=25.
    #       FAIL (Arrive @ 27, mais due[1] @ 25)
    route_tw_fail_due = [0, 2, 1, 0]
    assert check_feasibility(route_tw_fail_due, instance_tw) == False
    print(f"Faisabilité (TW) [0,2,1,0] (arrive en retard): {False} ✅")

    # Test Tmax
    # On reprend la route [0,1,2,0] qui arrive à t=37
    instance_tw_tmax_fail = Instance(
        name="test_tmax",
        distance_matrix=instance_tw.distance_matrix,
        demand=instance_tw.demand,
        capacity=instance_tw.capacity,
        ready_time=instance_tw.ready_time,
        due_time=instance_tw.due_time,
        service_time=instance_tw.service_time,
        Tmax=35.0 # Tmax < 37
    )
    assert check_feasibility(route_tw_ok, instance_tw_tmax_fail) == False
    print(f"Faisabilité (Tmax) [0,1,2,0] (Arrive @ 37 > Tmax 35): {False} ✅")
    
    # --- 4. Tests evaluate_solution ---
    print("\n--- 4. Test evaluate_solution ---")
    sol = Solution(routes=[[0, 1, 2, 0], [0, 3, 0]])
    # Coûts: (10+5+15) + (20+20) = 30 + 40 = 70
    # Faisabilité VRP:
    # R1: Dem 5+5=10 <= 20 (OK)
    # R2: Dem 8 <= 20 (OK)
    # Kmax: 2 routes <= Kmax=2 (OK)
    sol_eval = evaluate_solution(sol, instance_vrp)
    assert sol_eval.cost == 70.0
    assert sol_eval.feasible == True
    assert sol_eval.meta["nb_routes"] == 2.0
    print(f"Éval (VRP) Coût: {sol_eval.cost} (70) | Faisable: {sol_eval.feasible} (True) ✅")

    sol_kmax_fail = Solution(routes=[[0, 1, 0], [0, 2, 0], [0, 3, 0]])
    sol_kmax_eval = evaluate_solution(sol_kmax_fail, instance_vrp) # Kmax=2
    assert sol_kmax_eval.feasible == False
    assert sol_kmax_eval.meta["nb_routes"] == 3.0
    print(f"Éval (VRP) Faisable (Kmax fail): {sol_kmax_eval.feasible} (False) ✅")

    # --- 5. Tests delta_cost_two_opt ---
    print("\n--- 5. Test delta_cost_two_opt ---")
    route_2opt = [0, 1, 4, 3, 2, 0] # Longueur 6
    # Indices i=2 (noeud 4), j=3 (noeud 3).
    # Segment [4, 3]
    # Nœuds: A=route[1]=1, B=route[2]=4, C=route[3]=3, D=route[4]=2
    # Old cost (A->B) + (C->D) = (1->4) + (3->2) = 30 + 10 = 40
    # New cost (A->C) + (B->D) = (1->3) + (4->2) = 30 + 35 = 65
    # Delta = 65 - 40 = 25
    
    delta = delta_cost_two_opt(route_2opt, 2, 3, DM)
    assert delta == 25.0
    print(f"Delta 2-opt (i=2, j=3): {delta} (Attendu: 25.0) ✅")
    
    # Vérification brute
    cost_old = compute_route_cost(route_2opt, DM) # 10+30+5+10+15 = 70
    route_new = [0, 1, 3, 4, 2, 0] # Segment [4,3] inversé en [3,4]
    cost_new = compute_route_cost(route_new, DM) # 10+30+5+35+15 = 95
    assert (cost_new - cost_old) == delta
    print(f"Vérif brute: {cost_new} - {cost_old} = {cost_new - cost_old} ✅")

    try:
        delta_cost_two_opt(route_2opt, 0, 3, DM) # i=0 (invalide)
        assert False, "i=0 aurait dû lever ValueError"
    except ValueError:
        print(f"Gestion erreur (i=0): OK ✅")
    try:
        delta_cost_two_opt(route_2opt, 2, 5, DM) # j=len-1 (invalide)
        assert False, "j=len-1 aurait dû lever ValueError"
    except ValueError:
        print(f"Gestion erreur (j=len-1): OK ✅")
    try:
        delta_cost_two_opt(route_2opt, 3, 2, DM) # i > j (invalide)
        assert False, "i > j aurait dû lever ValueError"
    except ValueError:
        print(f"Gestion erreur (i > j): OK ✅")

    print("\n🎉 Tous les tests d'évaluation ont réussi!")