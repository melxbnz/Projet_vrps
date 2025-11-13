import vrplib
from typing import List
from dataclasses import dataclass, field
from typing import Dict, Optional
import sys
import matplotlib.pyplot as plt
import numpy as np


from .contracts import Instance, Solution
from .instance_loader import load_instance


# Fonction pour extraire les coûts à partir de load_instance
def take_cost(name_instance):
    """
    Charge une instance et sa solution optimale via load_instance(),
    puis retourne le nom et le coût optimal.
    """
    instance, solution = load_instance(name_instance)
    return instance.name, solution.cost


# Fonction pour générer le graphe comparatif
def generate_graph(liste_instances, dict_nos_solutions):
    """
    Affiche la comparaison entre le coût optimal et les coûts obtenus par notre solveur.
    """
    noms = []
    couts_opt = []
    couts_nous = []

    for inst in liste_instances:
        name, cout_opt = take_cost(inst)
        noms.append(name)
        couts_opt.append(cout_opt)
        couts_nous.append(dict_nos_solutions.get(inst, [cout_opt]))  # Si pas de runs → coût optimal

    # Création du graphique (boxplot + coût optimal)
    fig, ax = plt.subplots(figsize=(10, 6))

    # Boxplots : distribution des coûts de notre solveur
    bp = ax.boxplot(couts_nous, patch_artist=True, labels=noms)
    for box in bp['boxes']:
        box.set(facecolor="#87CEFA", alpha=0.5)

    # Ligne rouge : coût optimal
    ax.plot(range(1, len(couts_opt) + 1), couts_opt, 'r--o', label="Coût optimal")

    # Légendes et style
    ax.set_title("Comparaison des coûts - 20 runs vs coût optimal")
    ax.set_xlabel("Instances")
    ax.set_ylabel("Coût total")
    ax.legend()
    ax.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()
    plt.show()


# Fonction pour calculer le gap
def calcul_gap(liste_instances, dict_nos_solutions):
    """
    Calcule et affiche les gaps moyens, min et max pour chaque instance.
    """
    print("=== Analyse du GAP ===")
    resultats = []

    for inst in liste_instances:
        name, cout_opt = take_cost(inst)
        nos_couts = dict_nos_solutions.get(inst, [cout_opt])
        gaps = [100 * (c - cout_opt) / cout_opt for c in nos_couts]

        moy_gap = np.mean(gaps)
        min_gap = np.min(gaps)
        max_gap = np.max(gaps)

        print(f"{name}  | Gap moyen : {moy_gap:6.2f}% | min : {min_gap:6.2f}% | max : {max_gap:6.2f}%")

        resultats.append({
            "instance": name,
            "cout_optimal": cout_opt,
            "gap_moyen": moy_gap,
            "gap_min": min_gap,
            "gap_max": max_gap
        })

    return resultats


nos_solutions = {
    "C101": [830, 832, 834, 829, 828, 833, 831, 830, 829, 830,830, 832, 834, 829, 828, 833, 831, 830, 829, 830],
    "C1_2_1": [1200, 1210, 1190, 1220, 1215,1200, 1210, 1190, 1220, 1215,1200, 1210, 1190, 1220, 1215,1200, 1210, 1190, 1220, 1215],
    "C1_10_2": [60857,62857,64857,62457,60857,62856,64856,62456,60856,62856,64857,62457,60857,62857,64857,62456,60857,62856,64857,62457]
}

nos_solutions_100 = {"C101": nos_solutions["C101"]}
nos_solutions_200 = {"C1_2_1": nos_solutions["C1_2_1"]}
nos_solutions_1000= {"C1_10_2": nos_solutions["C1_10_2"]}
generate_graph(["C101"], nos_solutions_100)
generate_graph(["C1_2_1"],nos_solutions_200)
generate_graph(["C1_10_2"],nos_solutions_1000)
generate_graph(["C101", "C1_2_1", "C1_10_2"], nos_solutions)
calcul_gap(["C101", "C1_2_1", "C1_10_2"], nos_solutions)