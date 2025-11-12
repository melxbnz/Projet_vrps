# ALBAN

# --------------------------------------------------- POUR VRPLIB -------------------------------------------------------------------

import vrplib
#from notre_solveur import fonction_solve # On importe notre solveur
# On prends une instance (Il faut définir le bon chemin)

instance = vrplib.read_instance("../data/X-n101-k25.vrp")
solution = vrplib.read_solution("../data/X-n101-k25.sol")
print("Nom de l'instance : ",instance['name'])
#Exécution de la fonction solve
# my_solution = fonction_solve(instance)

#Calcul du gap optimal (critère d'évaluation)
#gap = 100 * (my_solution["cost"] - solution["cost"]) / solution["cost"]
#print(f"Gap vs référence : {gap:.2f}%") # Doit être < 5% pour les instances < 100 clients

# Afficher les clés de notre instance et de la solution
print("Les clés de notre instance : ",instance.keys())
print("Les clés de notre solution : ",solution.keys())

# Afficher les routes solution
print("Les routes à empruntées :")
for i,route in enumerate(solution["routes"]):
    print(f"Route {i} : {route}")
print("Le cout des routes empruntées : ",solution['cost'])



# --------------------------------------------------- POUR VRPTW -------------------------------------------------------------------------

import vrplib
#from notre_solveur import fonction_solve # On importe notre solveur

# On prends une instance (Il faut définir le bon chemin)
instance = vrplib.read_instance("../data/C101.txt", instance_format="solomon")
solution = vrplib.read_solution("../data/C101.sol")
print("Nom de l'instance : ",instance['name'])
#Exécution de la fonction solve
# my_solution = fonction_solve(instance)

#Calcul du gap optimal (critère d'évaluation)
#gap = 100 * (my_solution["cost"] - solution["cost"]) / solution["cost"]
#print(f"Gap vs référence : {gap:.2f}%") # Doit être < 5% pour les instances < 100 clients

# Afficher les clés de notre instance et de la solution
print("Les clés de notre instance : ",instance.keys())
print("Les clés de notre solution : ",solution.keys())

#Afficher les informations de notre instance
for keys in instance.keys():
   print(keys," : ",instance[f"{keys}"])

# Afficher les routes solution
print("Les routes à empruntées :")
for i,route in enumerate(solution["routes"]):
    print(f"Route {i} : {route}")
print("Le cout des routes empruntées : ",solution['cost'])
