import math

class Solution:
    def __init__(self, V, K, c, q, Q, routes):
        self.V, self.K, self.c, self.q, self.Q = V, K, c, q, Q
        self.routes = routes                  # routes[k] = liste de clients
        # construction explicite de x_{ijk}
        self.x = [[[0 for _ in range(K)] for _ in V] for _ in V]
        for k in range(K):
            r = routes[k]; prev = 0
            for j in r:
                self.x[prev][j][k] = 1; prev = j
            self.x[prev][0][k] = 1

    def cost(self):
        """ Fonction objectif F = somme_k somme_i somme_j c_ij * x_ijk """
        return sum(self.c[i][j]*self.x[i][j][k]
                   for k in range(self.K) for i in self.V for j in self.V)



#        1. TWO-OPT (inversion d’un segment [i..j])

def delta_two_opt(sol, k, i, j):
    r, c = sol.routes[k], sol.c
    a = 0 if i == 0 else r[i-1]
    b, c1 = r[i], r[j]
    d = 0 if j == len(r)-1 else r[j+1]
    return (c[a][c1] + c[b][d]) - (c[a][b] + c[c1][d])

def apply_two_opt(sol, k, i, j):
    sol.routes[k][i:j+1] = reversed(sol.routes[k][i:j+1])



#        2. RELOCATE (déplacer un client)

def delta_relocate(sol, k1, i, k2, j):
    M = sol
    r1, r2, c = M.routes[k1], M.routes[k2], M.c
    u = r1[i]
    if k1 != k2 and sum(M.q[x] for x in r2)+M.q[u] > M.Q: return math.inf
    a1 = 0 if i == 0 else r1[i-1]
    b1 = 0 if i == len(r1)-1 else r1[i+1]
    rm = -c[a1][u]-c[u][b1]+c[a1][b1]
    a2 = 0 if j == 0 else r2[j-1]
    b2 = 0 if j == len(r2) else r2[j]
    ins = -c[a2][b2]+c[a2][u]+c[u][b2]
    return rm + ins

def apply_relocate(sol, k1, i, k2, j):
    u = sol.routes[k1].pop(i)
    sol.routes[k2].insert(j, u)


#        3. SWAP (échange de deux clients)

def delta_swap(sol, k1, i, k2, j):
    M = sol; c = M.c
    r1, r2 = M.routes[k1], M.routes[k2]
    u, v = r1[i], r2[j]
    if k1 != k2:
        if sum(M.q[x] for x in r1)-M.q[u]+M.q[v] > M.Q: return math.inf
        if sum(M.q[x] for x in r2)-M.q[v]+M.q[u] > M.Q: return math.inf
    a1 = 0 if i == 0 else r1[i-1]; b1 = 0 if i == len(r1)-1 else r1[i+1]
    a2 = 0 if j == 0 else r2[j-1]; b2 = 0 if j == len(r2)-1 else r2[j+1]
    old = c[a1][u]+c[u][b1]+c[a2][v]+c[v][b2]
    new = c[a1][v]+c[v][b1]+c[a2][u]+c[u][b2]
    return new - old

def apply_swap(sol, k1, i, k2, j):
    sol.routes[k1][i], sol.routes[k2][j] = sol.routes[k2][j], sol.routes[k1][i]



#        4. CALCUL DU DELTA-COST RAPIDE (wrapper)

def delta_cost(sol, move_type, *args):
    if move_type == "two_opt":   return delta_two_opt(sol, *args)
    if move_type == "relocate":  return delta_relocate(sol, *args)
    if move_type == "swap":      return delta_swap(sol, *args)
    return math.inf



#         5. RECHERCHE LOCALE AVEC VOISINAGE

def local_search(sol):
    """
    Applique itérativement les mouvements Two-Opt, Relocate et Swap
    tant qu'une amélioration de la fonction objectif F est trouvée.
    """
    improved = True
    iteration = 0

    while improved:
        improved = False
        best_delta = 0
        best_move = None
        iteration += 1

        #  Parcours de tous les véhicules 
        for k1 in range(sol.K):
            r1 = sol.routes[k1]

            #  (1) TWO-OPT sur la même tournée 
            for i in range(len(r1)-1):
                for j in range(i+1, len(r1)):
                    Δ = delta_two_opt(sol, k1, i, j)
                    if Δ < best_delta:
                        best_delta, best_move = Δ, ("two_opt", k1, i, j)

            #  (2) RELOCATE intra- et inter-tournées 
            for i in range(len(r1)):
                for k2 in range(sol.K):
                    r2 = sol.routes[k2]
                    for j in range(len(r2)+1):
                        Δ = delta_relocate(sol, k1, i, k2, j)
                        if Δ < best_delta:
                            best_delta, best_move = Δ, ("relocate", k1, i, k2, j)

            #  (3) SWAP intra- et inter-tournées
            for i in range(len(r1)):
                for k2 in range(sol.K):
                    r2 = sol.routes[k2]
                    for j in range(len(r2)):
                        Δ = delta_swap(sol, k1, i, k2, j)
                        if Δ < best_delta:
                            best_delta, best_move = Δ, ("swap", k1, i, k2, j)

        #  Application du meilleur move trouvé 
        if best_move:
            mv = best_move
            if mv[0] == "two_opt":   apply_two_opt(sol, mv[1], mv[2], mv[3])
            if mv[0] == "relocate":  apply_relocate(sol, mv[1], mv[2], mv[3], mv[4])
            if mv[0] == "swap":      apply_swap(sol, mv[1], mv[2], mv[3], mv[4])
            improved = True
            print(f"Iter {iteration}: {mv[0]} {mv[1:]} -> Δ={best_delta:.2f}, Nouveau coût = {sol.cost():.2f}")

    print("Aucune amélioration trouvée — solution locale atteinte.")
    return sol
