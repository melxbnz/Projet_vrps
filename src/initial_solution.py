import vrplib
from dataclasses import dataclass, field
from typing import List, Dict, Optional

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

    def copy(self):
        import copy
        return copy.deepcopy(self)

def check_feasibility(route: Route, instance: Instance) -> bool:
    """Vérifie faisabilité : capacité et fenêtres temporelles."""
    load = sum(instance.demand[i] for i in route if i != 0)
    if load > instance.capacity:
        return False

    if instance.ready_time is not None and instance.due_time is not None:
        time = 0
        D = instance.distance_matrix
        for k in range(len(route)-1):
            i, j = route[k], route[k+1]
            time += D[i][j]
            time = max(time, instance.ready_time[j])
            if time > instance.due_time[j]:
                return False
            if instance.service_time is not None:
                time += instance.service_time[j]

    return True

def evaluate_solution(solution: Solution, instance: Instance):
    """Évalue la solution : coût total et faisabilité globale."""
    total_cost = 0
    for route in solution.routes:
        D = instance.distance_matrix
        cost = sum(D[route[k]][route[k+1]] for k in range(len(route)-1))
        total_cost += cost
    solution.cost = total_cost
    solution.feasible = all(check_feasibility(r, instance) for r in solution.routes)

# Ma partie Clarke et Wright
def build_clarke_wright_solution(instance: Instance) -> Solution:
    """Génère une solution initiale pour le VRP via Clarke & Wright."""
    n = len(instance.demand)
    routes: List[Route] = [[0, i, 0] for i in range(1, n)]

    # On calcule des économies dans les routes
    economies = []
    D = instance.distance_matrix
    for i in range(1, n):
        for j in range(i+1, n):
            s = D[0][i] + D[0][j] - D[i][j]
            economies.append((s, i, j))
    economies.sort(reverse=True)

    # On fusionne les routes pour optimiser
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
            if check_feasibility(merged, instance):
                for idx in sorted([ri, rj], reverse=True):
                    routes.pop(idx)
                routes.append(merged)

    solution = Solution(routes=routes)
    evaluate_solution(solution, instance)
    return solution

# Pour tester C101
if __name__ == "__main__":

    vrp_data = vrplib.read_instance("C101.txt", instance_format="solomon")
    coords = vrp_data["node_coord"]
    n = len(coords)

    def euclidean_distance(a, b):
        return ((a[0]-b[0])**2 + (a[1]-b[1])**2) ** 0.5

    distance_matrix = [[euclidean_distance(coords[i], coords[j]) for j in range(n)] for i in range(n)]

    instance = Instance(
        name="C101",
        distance_matrix=distance_matrix,
        demand=vrp_data["demand"],
        capacity=vrp_data["capacity"],
        ready_time=[tw[0] for tw in vrp_data["time_window"]],
        due_time=[tw[1] for tw in vrp_data["time_window"]],
        service_time=vrp_data["service_time"]
    )

    solution = build_clarke_wright_solution(instance)

    print("Routes trouvées :")
    for r in solution.routes:
        print(r)
    print(f"Coût total : {solution.cost}")
    print(f"Faisable : {solution.feasible}")
