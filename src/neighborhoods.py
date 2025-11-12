import math
from typing import List, Tuple

from contracts import Solution, Instance  # type: ignore

# 1. TWO-OPT (inversion d’un segment [i..j])

def apply_two_opt(sol: Solution, k: int, i: int, j: int):
    """
    Applique le mouvement Two-Opt sur la route k en inversant le segment entre 
    les indices i et j (inclus).
    
    i et j sont des index de client dans la liste sol.routes[k].
    """
    # Inverse la sous-liste de clients
    sol.routes[k][i:j+1] = list(reversed(sol.routes[k][i:j+1]))


# 2. RELOCATE (déplacer un client)

def apply_relocate(sol: Solution, k1: int, i: int, k2: int, j: int):
    """
    Déplace le client à l'index i de la route k1 et l'insère 
    à la position j de la route k2.
    """
    # Pop et insert sur la liste de clients
    u = sol.routes[k1].pop(i)
    sol.routes[k2].insert(j, u)


# 3. SWAP (échange de deux clients)

def apply_swap(sol: Solution, k1: int, i: int, k2: int, j: int):
    """
    Échange le client à l'index i de la route k1 avec le client 
    à l'index j de la route k2.
    """
    sol.routes[k1][i], sol.routes[k2][j] = sol.routes[k2][j], sol.routes[k1][i]



# Fonctions de calcul de Delta-Cost (Delta)

# 2. RELOCATE (déplacer un client)

def delta_relocate(sol: Solution, instance: Instance, k1: int, i: int, k2: int, j: int) -> float:
    """
    Calcule le changement de coût résultant du déplacement du client 
    à l'index i de la route k1 vers la position j de la route k2.
    """
    r1, r2 = sol.routes[k1], sol.routes[k2]
    c = instance.distance_matrix
    Q = instance.capacity
    q = instance.demand

    u = r1[i]
    
    # Vérification de la contrainte de capacité pour un déplacement inter-route
    if k1 != k2:
        current_demand_r2 = sum(q[x] for x in r2)
        if current_demand_r2 + q[u] > Q: 
            return math.inf

    # Suppression dans la route k1 (a1->u->b1 remplacé par a1->b1)
    a1 = 0 if i == 0 else r1[i-1]
    b1 = 0 if i == len(r1)-1 else r1[i+1] 
    
    # Coût retiré : -(a1->u) - (u->b1) + (a1->b1)
    rm = -c[a1][u] - c[u][b1] + c[a1][b1]
    
    # Insertion dans la route k2 (a2->b2 remplacé par a2->u->b2)
    a2 = 0 if j == 0 else r2[j-1]
    b2 = 0 if j == len(r2) else r2[j] # j == len(r2) est l'insertion avant le dépôt final
    
    # Coût inséré : -(a2->b2) + (a2->u) + (u->b2)
    ins = -c[a2][b2] + c[a2][u] + c[u][b2]
    
    return rm + ins

# 3. SWAP (échange de deux clients)

def delta_swap(sol: Solution, instance: Instance, k1: int, i: int, k2: int, j: int) -> float:
    """
    Calcule le changement de coût résultant de l'échange du client 
    à l'index i de la route k1 (u) avec le client à l'index j de la route k2 (v).
    """
    c = instance.distance_matrix
    Q = instance.capacity
    q = instance.demand
    r1, r2 = sol.routes[k1], sol.routes[k2]
    
    u, v = r1[i], r2[j]
    
    # Vérification de l'opération (swap d'un élément avec lui-même est inutile)
    if k1 == k2 and i == j:
        return math.inf
        
    # Vérification des contraintes de capacité pour un échange inter-route
    if k1 != k2:
        # Route k1: retire u, ajoute v
        if sum(q[x] for x in r1) - q[u] + q[v] > Q: 
            return math.inf
            
        # Route k2: retire v, ajoute u
        if sum(q[x] for x in r2) - q[v] + q[u] > Q: 
            return math.inf

    # Arcs autour de u dans r1
    a1 = 0 if i == 0 else r1[i-1]
    b1 = 0 if i == len(r1)-1 else r1[i+1]
    
    # Arcs autour de v dans r2
    a2 = 0 if j == 0 else r2[j-1]
    b2 = 0 if j == len(r2)-1 else r2[j+1]

    # Cas spécial: u et v sont voisins sur la même route (k1=k2 et |i-j|=1)
    if k1 == k2 and abs(i - j) == 1:
        if i > j: # Assurer i est le premier
            i, j = j, i
            u, v = v, u
            a1, a2 = a2, a1
            b1, b2 = b2, b1
            
        
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
    
    NOTE: Pour le "two_opt", la fonction doit être importée d'ailleurs (e.g., evaluation.py).
    """
    # La fonction delta_relocate attend 4 arguments: k1, i, k2, j
    if move_type == "relocate":  
        if len(args) != 4: return math.inf
        return delta_relocate(sol, instance, *args)
        
    # La fonction delta_swap attend 4 arguments: k1, i, k2, j
    if move_type == "swap":      
        if len(args) != 4: return math.inf
        return delta_swap(sol, instance, *args)
        
    
    return math.inf
