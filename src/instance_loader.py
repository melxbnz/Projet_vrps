# ALBAN
from .contracts import Instance, Solution
# --------------------------------------------------- POUR VRPTW -------------------------------------------------------------------------

import vrplib
from typing import List
from dataclasses import dataclass, field
from typing import Dict, Optional
#from notre_solveur import fonction_solve # On importe notre solveur

def load_instance(name_instance: str) -> tuple[Instance, Solution]:
    """Charge une instance Solomon (via vrplib) et renvoie des objets Instance et Solution."""
    data = vrplib.read_instance(f"../data/{name_instance}.txt", instance_format="solomon")
    sol_data = vrplib.read_solution(f"../data/{name_instance}.sol")

    # Construction de la matrice des distances 
    dist_matrix = data.get("edge_weight")
    if dist_matrix is None:
        # Si edge_weight n'est pas fourni, calculer à partir des coordonnées
        import math
        coords = data["node_coord"]
        n = len(coords)
        dist_matrix = [
            [math.hypot(coords[i][0] - coords[j][0], coords[i][1] - coords[j][1]) for j in range(n)]
            for i in range(n)
        ]

    # Création de l'objet Instance 
    instance = Instance(
        name=data["name"],
        distance_matrix=dist_matrix,
        demand=data["demand"],
        capacity=data["capacity"],
        ready_time=data.get("time_window", [None] * len(data["demand"]))[0],
        due_time=data.get("time_window", [None] * len(data["demand"]))[1] if "time_window" in data else None,
        service_time=data.get("service_time", None),
        Kmax=data.get("vehicles", []) if "vehicles" in data else None,
    )

    # Création de l'objet Solution
    solution = Solution(
        routes=sol_data["routes"],
        cost=sol_data["cost"],
        feasible=True,
        meta={"source": "vrplib"}
    )

    print(f"Instance '{instance.name}' chargée ({len(instance.demand) - 1} clients)")
    print(f" → Coût optimal : {solution.cost}")
    return instance, solution

# Exemple d'utilisation
if __name__ == "__main__":
    liste_instances = ["C101", "C1_2_1", "C1_10_2"]
    for inst in liste_instances:
        instance, solution = load_instance(inst)
