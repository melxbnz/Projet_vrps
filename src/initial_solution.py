from typing import List
from .contracts import Instance, Solution
from .evaluation import check_feasibility, evaluate_solution

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
