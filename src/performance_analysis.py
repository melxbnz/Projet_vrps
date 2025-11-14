import vrplib
from typing import List
from dataclasses import dataclass, field
from typing import Dict, Optional
import sys
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd  # ADD
import os #ADD

from .contracts import Instance, Solution
from .instance_loader import load_instance

#ADD
try:
    from .contracts import Instance, Solution
    from .instance_loader import load_instance
except ImportError:
    print("Erreur d'import, assurez-vous de lancer le script comme un module.", file=sys.stderr)
    sys.exit(1)
#INCHANGER
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


# =============================================================================
# NOUVELLES FONCTIONS (POUR LIRE LE CSV)
# =============================================================================

def load_results_from_csv(csv_file: str) -> Dict[str, List[float]]:
    """
    Charge les résultats depuis le CSV généré par le benchmark
    et les formate pour les fonctions d'analyse.
    """
    try:
        # S'assurer que le fichier est lu depuis la racine du projet
        project_root = os.path.dirname(os.path.dirname(__file__)) # Va de src/ -> Projet_vrp/
        csv_path = os.path.join(project_root, csv_file)
        
        df = pd.read_csv(csv_path)
    except FileNotFoundError:
        print(f"--- 🛑 ERREUR ---", file=sys.stderr)
        print(f"Fichier '{csv_path}' non trouvé.", file=sys.stderr)
        print("Veuillez vérifier que 'run_benchmark.py' a bien généré le fichier.", file=sys.stderr)
        return {}
    except Exception as e:
        print(f"Erreur lors de la lecture du CSV: {e}", file=sys.stderr)
        return {}

    # Transformer le DataFrame en dictionnaire de listes de coûts
    # Format attendu: {"C101": [828.94, 828.94, ...], "C1_2_1": [...]}
    results_dict = {}
    for instance_name in df['instance_name'].unique():
        costs = df[df['instance_name'] == instance_name]['final_cost'].tolist()
        results_dict[instance_name] = costs
    
    return results_dict


# =============================================================================
# POINT D'ENTRÉE MODIFIÉ
# =============================================================================

if __name__ == "__main__":
    
    # 1. Nom du fichier CSV généré par run_benchmark.py
    CSV_FILENAME = "benchmark_results_parallel.csv"

    # 2. Charger les données depuis ce fichier
    print(f"--- Chargement des résultats depuis '{CSV_FILENAME}' ---")
    nos_solutions = load_results_from_csv(CSV_FILENAME)

    if not nos_solutions:
        sys.exit(1) # Arrêter si le chargement a échoué

    # 3. Obtenir la liste des instances qui ont été testées
    liste_instances = list(nos_solutions.keys())
    print(f"Instances trouvées dans le fichier: {liste_instances}")

    # 4. Lancer les analyses (en utilisant vos fonctions)
    
    # Générer les graphiques individuels
    for inst in liste_instances:
        print(f"\n--- Analyse de l'instance: {inst} ---")
        generate_graph([inst], {inst: nos_solutions[inst]})

    # Générer le graphique combiné
    print("\n--- Analyse combinée ---")
    generate_graph(liste_instances, nos_solutions)

    # Calculer et afficher les GAPs
    calcul_gap(liste_instances, nos_solutions)

    print("\n✅ Analyse terminée. Les graphiques (.png) sont sauvegardés à la racine du projet.")