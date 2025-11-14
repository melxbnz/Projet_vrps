#METHODE EXACTE: PROGRAMMATION LINEAIRE EN ENTIER MIXT

import copy  # Pour copie (si besoin clonage Solution)
import time  # Chronométrage résolution
import numpy as np  # Matrices (distances euclidiennes)
import matplotlib.pyplot as plt  # Visualisation
import pulp  # MILP solver (CBC pour exact)
from typing import List, Dict, Optional  # Annotations types
from dataclasses import dataclass, field  # Structures POO

# Constantes (notebook : M pour big-M, gap non utilisé car exact)
BIG_M = 1e6  # Big-M pour linéarisation TW/subtours (choix : > max c_ij + max b_i)
VERYBIG = 1e10  # Infini (arcs interdits, si besoin)

# --- Structures (contracts.py : uniformité notebook) ---
@dataclass
class Instance:
    name: str
    distance_matrix: List[List[float]]  # c_ij
    demand: List[float]  # q_i (q_0=q_{n+1}=0)
    capacity: float  # Q
    ready_time: Optional[List[float]] = None  # a_i
    due_time: Optional[List[float]] = None  # b_i
    service_time: Optional[List[float]] = None  # s_i
    Kmax: Optional[int] = None  # Kmax
    Tmax: Optional[float] = None  # Tmax
    max_route_length: Optional[int] = None  # Lmax
    posx: Optional[List[float]] = None  # x coord
    posy: Optional[List[float]] = None  # y coord

    def __post_init__(self):
        self.verybig = VERYBIG  # Compat

@dataclass
class Solution:
    routes: List[List[int]] = field(default_factory=list)
    cost: float = float("inf")
    feasible: bool = False
    meta: Dict[str, float] = field(default_factory=dict)

    def copy(self):
        import copy
        return copy.deepcopy(self)

# --- Évaluation (evaluation.py : uniformité) ---
def compute_route_cost(route: List[int], distance_matrix: List[List[float]]) -> float:
    """
    z_R = sum_{(i,j) in R} c_ij.
    """
    if len(route) < 2:
        return 0.0
    return sum(distance_matrix[route[i]][route[i+1]] for i in range(len(route)-1))

def check_feasibility(route: List[int], instance: Instance) -> bool:
    """
    Faisabilité R : sum q_i <= Q, |clients| <= Lmax, simulation t_j = max(a_j, t_i + c_ij + s_i) <= b_j, t_sink <= Tmax.
    """
    n = len(instance.distance_matrix)
    clients = route[1:-1]
    # Capacité
    if sum(instance.demand[i] for i in clients) > instance.capacity:
        return False
    # Lmax
    if instance.max_route_length and len(clients) > instance.max_route_length:
        return False
    # TW/Tmax (si présents)
    if instance.ready_time is None:
        return True
    t = 0.0
    for k in range(len(route)-1):
        i, j = route[k], route[k+1]
        t += instance.distance_matrix[i][j]
        t = max(t, instance.ready_time[j])
        if t > instance.due_time[j]:
            return False
        t += instance.service_time[j]
    if instance.Tmax and t > instance.Tmax:
        return False
    return True

def evaluate_solution(solution: Solution, instance: Instance) -> Solution:
    """
    z = sum z_Rk, feasible = and feasible(R_k) and |R| <= Kmax.
    """
    solution.cost = sum(compute_route_cost(r, instance.distance_matrix) for r in solution.routes)
    solution.feasible = (len(solution.routes) <= instance.Kmax if instance.Kmax else True) and \
                        all(check_feasibility(r, instance) for r in solution.routes)
    solution.meta['nb_routes'] = len(solution.routes)
    return solution

def solve_milp(instance: Instance) -> Solution:
    n = len(instance.demand) - 2  # n_clients=10
    K = instance.Kmax or n  # K <= n (borné)
    sink = n + 1  # sink=11, |V|=n+2=12
    print(f"[MILP] n_clients={n}, |V|={n+2}, K={K}")

    # Problème
    prob = pulp.LpProblem("VRPTW_MILP", pulp.LpMinimize)

    # Vars x_i j k (i,j=0..sink, i!=j, k=1..K)
    x = pulp.LpVariable.dicts("x", ((i, j, k) for i in range(n+2) for j in range(n+2) if i != j for k in range(1, K+1)),
                              cat=pulp.LpBinary)

    # y_i k (i=1..n servi par k)
    y = pulp.LpVariable.dicts("y", ((i, k) for i in range(1, n+1) for k in range(1, K+1)),
                              lowBound=0, upBound=1, cat=pulp.LpContinuous)

    # u_i (MTZ, i=1..n)
    u = pulp.LpVariable.dicts("u", (i for i in range(1, n+1)), lowBound=0, upBound=n, cat=pulp.LpContinuous)

    # t_i (temps, i=0..sink)
    t = pulp.LpVariable.dicts("t", (i for i in range(n+2)), lowBound=0, cat=pulp.LpContinuous)

    # Objectif : min sum_{i,j,k} c_ij x_ijk
    prob += pulp.lpSum(instance.distance_matrix[i][j] * x[(i,j,k)] for i in range(n+2) for j in range(n+2) if i != j for k in range(1,K+1))

    # Constr 1: Chaque client servi exactement une fois (sum_{i,k} x_i j k =1 forall j=1..n)
    for j in range(1, n+1):
        prob += pulp.lpSum(x[(i,j,k)] for i in range(n+2) if i != j for k in range(1,K+1)) == 1, f"serve_{j}"

    # Constr 2: Flux conservation (entree = sortie pour clients j=1..n)
    for j in range(1, n+1):
        for k in range(1, K+1):
            prob += (pulp.lpSum(x[(i,j,k)] for i in range(n+2) if i != j) ==
                     pulp.lpSum(x[(j,l,k)] for l in range(n+2) if l != j)), f"flow_{j}_{k}"

    # Constr 3: Dépôts : sum_k sum_{j=1}^n x_0 j k = sum_k sum_{i=1}^n x_i, sink, k (départs=arrivées)
    for k in range(1, K+1):
        prob += pulp.lpSum(x[(0,j,k)] for j in range(1,n+1)) == pulp.lpSum(x[(i,sink,k)] for i in range(1,n+1)), f"depot_{k}"

    # Constr 4: Définition y_i k = sum_{j != i} x_j i k (indicateur servi par k)
    for i in range(1, n+1):
        for k in range(1, K+1):
            prob += y[(i,k)] == pulp.lpSum(x[(j,i,k)] for j in range(n+2) if j != i), f"def_y_{i}_{k}"

    # Constr 5: Capacité exacte : sum_{i=1}^n q_i y_i k <= Q forall k
    for k in range(1, K+1):
        prob += pulp.lpSum(instance.demand[i] * y[(i,k)] for i in range(1, n+1)) <= instance.capacity, f"cap_{k}"

    # Constr 6: Lmax si set : sum_{i=1}^n y_i k <= Lmax forall k
    if instance.max_route_length is not None:
        for k in range(1, K+1):
            prob += pulp.lpSum(y[(i,k)] for i in range(1, n+1)) <= instance.max_route_length, f"lmax_{k}"

    # Constr 7: MTZ subtours (u_j >= u_i +1 - n(1-x_ij k)) forall k, i!=j=1..n
    for k in range(1, K+1):
        for i in range(1, n+1):
            for j in range(1, n+1):
                if i != j:
                    prob += u[j] >= u[i] + 1 - n * (1 - x[(i,j,k)]), f"mtz_{i}_{j}_{k}"

    # Constr 8: TW et service (t_j >= t_i + c_ij + s_i - M(1-x_ij k)) forall i!=j, k
    if instance.ready_time is not None:
        for i in range(n+2):
            for j in range(n+2):
                if i != j:
                    for k in range(1, K+1):
                        prob += (t[j] >= t[i] + instance.distance_matrix[i][j] + instance.service_time[i] -
                                 BIG_M * (1 - x[(i,j,k)]), f"tw_{i}_{j}_{k}")
        # Bornes TW : a_i <= t_i <= b_i forall i=0..sink
        for i in range(n+2):
            prob += t[i] >= instance.ready_time[i], f"ready_{i}"
            prob += t[i] <= instance.due_time[i], f"due_{i}"
        # Tmax approx : t_sink <= Tmax (loose multi-k)
        if instance.Tmax:
            prob += t[sink] <= instance.Tmax, f"tmax"

    # Solve (CBC avec logs)
    start_time = time.time()
    prob.solve(pulp.PULP_CBC_CMD(msg=1))  # msg=1 : logs CBC
    solve_time = time.time() - start_time

    status = pulp.LpStatus[prob.status]
    if status != "Optimal":
        print(f"[MILP] Statut : {status} (pas de solution optimale)")
        return Solution(routes=[], cost=float('inf'), feasible=False, meta={'solve_time': solve_time})

    z = pulp.value(prob.objective)
    print(f"[MILP] Optimal z={z:.2f} en {solve_time:.2f}s")

    # Extraction routes par k
    routes = []
    for k in range(1, K+1):
        route = [0]  # Départ dépôt
        curr = 0
        while True:
            next_node = None
            for j in range(1, n+2):  # 1..n (clients) + sink=n+1
                if pulp.value(x[(curr, j, k)]) > 0.5:
                    next_node = j
                    break
            if next_node is None:
                break  # Fin route
            route.append(next_node)
            curr = next_node
        # Si route valide (>2 nœuds, i.e., au moins un client)
        if len(route) > 2:
            # Remplacer sink par 0 pour visu (même pos)
            if route[-1] == sink:
                route[-1] = 0
            routes.append(route)

    solution = Solution(routes=routes, meta={'solve_time': solve_time})
    evaluate_solution(solution, instance)
    return solution





# --- Main (test avec instance réelle tronquée C101 : 10 clients) ---
def main():
  
    n_clients = 10  # Petits pour CBC
    n_nodes = n_clients + 2  # |V|=12 (0 depot, 1-10 clients, 11 sink)
    sink = n_nodes - 1  # 11

    # Données de C101 (premiers 10 clients + dépôts)
    # Depot (ligne 1 et 101) : x=35, y=35, d=0, r=0, due=1440, s=0
    # Client 1 (ligne 2) : x=41, y=49, d=10, r=50, due=200, s=10
    # Client 2 (ligne 3) : x=48, y=44, d=15, r=50, due=200, s=10
    # Client 3 (ligne 4) : x=13, y=7, d=20, r=50, due=200, s=10
    # Client 4 (ligne 5) : x=29, y=64, d=25, r=50, due=200, s=10
    # Client 5 (ligne 6) : x=5, y=94, d=30, r=50, due=200, s=10
    # Client 6 (ligne 7) : x=74, y=2, d=35, r=50, due=200, s=10
    # Client 7 (ligne 8) : x=85, y=85, d=40, r=50, due=200, s=10
    # Client 8 (ligne 9) : x=73, y=6, d=45, r=50, due=200, s=10
    # Client 9 (ligne 10) : x=52, y=55, d=50, r=50, due=200, s=10
    # Client 10 (ligne 11) : x=32, y=65, d=55, r=50, due=200, s=10

    posx = [35.0, 41.0, 48.0, 13.0, 29.0, 5.0, 74.0, 85.0, 73.0, 52.0, 32.0, 35.0]  # depot + clients 1-10 + sink
    posy = [35.0, 49.0, 44.0, 7.0, 64.0, 94.0, 2.0, 85.0, 6.0, 55.0, 65.0, 35.0]
    demand = [0.0, 10.0, 15.0, 20.0, 25.0, 30.0, 35.0, 40.0, 45.0, 50.0, 55.0, 0.0]  # q_i
    ready = [0.0, 50.0, 50.0, 50.0, 50.0, 50.0, 50.0, 50.0, 50.0, 50.0, 50.0, 0.0]  # a_i
    due = [1440.0, 200.0, 200.0, 200.0, 200.0, 200.0, 200.0, 200.0, 200.0, 200.0, 200.0, 1440.0]  # b_i
    service = [0.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 0.0]  # s_i

    instance = Instance(
        name="C101_truncated_10",  # Nom pour instance tronquée C101
        distance_matrix=np.zeros((n_nodes, n_nodes)),  # À remplir euclidien
        demand=demand, capacity=200.0,  # Q standard C101
        ready_time=ready, due_time=due, service_time=service,
        Kmax=4, Tmax=1000.0, max_route_length=None,  # Tmax large pour faisabilité
        posx=posx, posy=posy
    )

    # Distances euclidiennes (round 1 décimale, comme Solomon)
    for i in range(n_nodes):
        for j in range(n_nodes):
            dx, dy = posx[i]-posx[j], posy[i]-posy[j]
            instance.distance_matrix[i][j] = round(10 * np.sqrt(dx**2 + dy**2)) / 10

    print(f"[Main] Instance réelle (C101 tronquée): {n_clients} clients, Q={instance.capacity}")

    # Graphe complet (avant algo)
    visualize_full_graph(instance, "full_graph_c101.png", popout=False)

    # Résolution MILP
    solution = solve_milp(instance)

    # Affichage
    print(f"[Main] Solution routes: {solution.routes}")
    print(f"[Main] Coût z = {solution.cost:.2f}")
    print(f"[Main] Faisable: {solution.feasible}")
    print(f"[Main] Temps: {solution.meta['solve_time']:.2f}s")
    print(f"[Main] |R| = {len(solution.routes)} (<= Kmax={instance.Kmax})")


if __name__ == "__main__":
    main()
