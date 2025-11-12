import math
import vrplib

# Fonction qui calcule la distance entre deux points
def distance(a, b):
    # a et b ce sont deux coordonnées 
    # On calcule la distance euclidienne avec la formule de maths
    return math.sqrt((a[0]-b[0])**2 + (a[1]-b[1])**2)


# On cherche la matrice pour les distances (upd problème réglé)
def matrice_distance(coords):
    # coords liste de points avec lesq coordonnées
    n = len(coords)
    D = [[0]*n for _ in range(n)]  # on crée une matrice carré n x n initialisée à 0

    # On remplit la matrice avec les distances euclidiennes (cf distance)
    for i in range(n):
        for j in range(n):
            D[i][j] = distance(coords[i], coords[j])
    return D


# On vérifie la capacité pour une route
def verif_capacite(route, demandes, capacite):
    # route est une liste de clients [0, i, j, ..., 0]
    # On compte pas (0) car c'est le point de départ, #gingembre
    charge = sum(demandes[i] for i in route if i != 0)
    # Retourne True si la charge est <= capacité, sinon False
    return charge <= capacite


# Pour vérifier les fenêtres temporelles
def verif_fenetre_temps(route, coords, debut, fin, service, D):
    # C le parcours du camion
    # temps le nb de min ou h (min de souvenir) donc 0 car le temps actuel
    temps = 0

    # On avance au fur et à mesure des clients
    for k in range(len(route)-1):
        i = route[k]      # client actuel
        j = route[k+1]    # prochain client

        # Temps de trajet entre i et j
        voyage = D[i][j]
        temps += voyage

        # Si on arrive avant l'ouverture de la fenêtre, on doit attendre
        temps = max(temps, debut[j]) 

        # Si on arrive après la fermeture de la fenêtre, la route est impossible
        if temps > fin[j]:
            return False

        # Temps de service à j
        temps += service[j]

    # Si on passe toutes les fenêtres, la route est faisable
    return True


#L'algorithgme de Clarke Wright

def clarke_wright(instance):
    # On extrait les données de l’instance VRPLIB
    coords = instance["coordonnées"]   # coordonnées des clients
    demandes = instance["demandes"]    # demande de chaque client
    debut = instance["tot"]       # début fenêtre
    fin = instance["tard"]           # fin fenêtre
    service = instance["service"]      # durée de service
    capacite = instance["capacite"]    # capacité du véhicule

    n = len(coords)

    # On  construit la matrice de distances
    D = matrice_distance(coords)

    # On construit une  route initiale pour chaque client
    routes = [[0, i, 0] for i in range(1, n)]

    # On calcules les économies
    # economies(i,j) = coût séparé - coût ensemble
    economies = []
    for i in range(1, n):
        for j in range(i+1, n):
            # économie réalisée en connectant i->j au lieu de les visiter séparément
            s = D[0][i] + D[0][j] - D[i][j]
            economies.append((s, i, j))

    # On trie les economies de la plus grande à la plus petite
    economies.sort(reverse=True)

    # Étape 4 : On essaie de fusionner les routes par ordre des economies
    for s, i, j in economies:

        # On cherche dans quelles routes se trouvent i et j
        ri = rj = None
        for idx, r in enumerate(routes):
            if i in r[1:-1]:  # i doit être dans la partie "clients" (pas le dépôt)
                ri = idx
            if j in r[1:-1]:
                rj = idx

        # Si i et j sont dans la même route → fusion impossible logique
        if ri is None or rj is None or ri == rj:
            continue

        route_i = routes[ri]
        route_j = routes[rj]

        # (fin de route_i + début de route_j)
        # i doit être le dernier client de sa route
        # j doit être le premier client de sa route
        if route_i[-2] == i and route_j[1] == j:

            # Construction de la route fusionnée
            merged = route_i[:-1] + route_j[1:]

            # Pour vérifier si cette fusion respecte les contraintes spatiales et temporelles
            if verif_capacite(merged, demandes, capacite) and \
               verif_fenetre_temps(merged, coords, debut, fin, service, D):

                # Si faisable → on fusionne (on remplace les deux routes par une seule)
                for idx in sorted([ri, rj], reverse=True):
                    routes.pop(idx)
                routes.append(merged)

    return routes

instance = vrplib.read_instance("C101.txt")
sol = clarke_wright(instance)
print(sol)
