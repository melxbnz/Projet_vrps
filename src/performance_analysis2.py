import vrplib
from typing import List, Dict, Optional
import sys
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd  # <-- Assurez-vous d'avoir pandas (pip install pandas)
import os

# --- Configuration pour des graphiques plus clairs ---
plt.style.use('ggplot')
plt.rcParams['figure.figsize'] = (10, 6)
plt.rcParams['savefig.dpi'] = 100 # Bonne résolution pour un rapport

# --- Modules de votre projet ---
try:
    from .contracts import Instance, Solution
    from .instance_loader import load_instance
except ImportError:
    print("Erreur d'import, assurez-vous de lancer le script comme un module.", file=sys.stderr)
    # Stubs pour permettre au script de se charger même en cas d'erreur d'import
    from dataclasses import dataclass, field
    NodeId = int
    Route = List[NodeId]
    @dataclass
    class Instance: pass
    @dataclass
    class Solution: pass
    def load_instance(name):
        print(f"ERREUR: 'load_instance' n'a pas pu être importé. "
              f"Veuillez lancer avec 'python -m src.performance_analysis'", file=sys.stderr)
        sys.exit(1)

# =============================================================================
# FONCTIONS D'ANALYSE
# =============================================================================

def take_cost(name_instance: str) -> tuple[str, float]:
    """
    Charge une instance et sa solution optimale via load_instance(),
    puis retourne le nom et le coût optimal.
    """
    instance, solution = load_instance(name_instance)
    return instance.name, solution.cost

def load_results_from_csv(csv_file: str) -> pd.DataFrame:
    """
    Charge les résultats depuis le CSV généré par le benchmark
    et le retourne sous forme de DataFrame pandas.
    """
    try:
        # S'assurer que le fichier est lu depuis la racine du projet
        # __file__ est le chemin de ce script (src/performance_analysis.py)
        # os.path.dirname(__file__) -> src/
        # os.path.dirname(os.path.dirname(__file__)) -> Projet_vrp/
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        csv_path = os.path.join(project_root, csv_file)
        
        print(f"--- Chargement des résultats depuis '{csv_path}' ---")
        df = pd.read_csv(csv_path)
        return df
        
    except FileNotFoundError:
        print(f"--- 🛑 ERREUR ---", file=sys.stderr)
        print(f"Fichier '{csv_path}' non trouvé.", file=sys.stderr)
        print("Veuillez vérifier que 'run_benchmark.py' a bien généré le fichier.", file=sys.stderr)
        return None
    except Exception as e:
        print(f"Erreur lors de la lecture du CSV: {e}", file=sys.stderr)
        return None

# =============================================================================
# POINT D'ENTRÉE PRINCIPAL
# =============================================================================

def main():
    """
    Fonction principale pour exécuter l'analyse complète.
    """
    
    # 1. Nom du fichier CSV généré par run_benchmark.py
    CSV_FILENAME = "benchmark_results_parallel.csv"

    # 2. Charger les données depuis ce fichier
    df = load_results_from_csv(CSV_FILENAME)

    if df is None:
        sys.exit(1) # Arrêter si le chargement a échoué

    # 3. Calculer le 'gap'
    df['gap'] = 100 * (df['final_cost'] - df['optimal_cost']) / df['optimal_cost']

    # 4. Calculer les statistiques
    summary = df.groupby('instance_name').agg(
        cout_optimal=('optimal_cost', 'first'),
        cout_moyen=('final_cost', 'mean'),
        gap_moyen_pct=('gap', 'mean'),
        gap_min_pct=('gap', 'min'),
        gap_max_pct=('gap', 'max'),
        temps_moyen_sec=('execution_time', 'mean'),
        nb_runs=('run_index', 'count')
    ).reset_index()

    # Ordonner le résumé par taille de problème (Small -> Medium -> Large)
    summary = summary.sort_values(by='cout_optimal')

    # 5. Afficher le tableau d'analyse
    print("\n=== ✅ Analyse Finale du Benchmark (Ordonnée) ===")
    print(summary.to_string(
        formatters={
            'cout_optimal': '{:,.2f}'.format,
            'cout_moyen': '{:,.2f}'.format,
            'gap_moyen_pct': '{:,.2f}%'.format,
            'gap_min_pct': '{:,.2f}%'.format,
            'gap_max_pct': '{:,.2f}%'.format,
            'temps_moyen_sec': '{:,.2f}s'.format,
        },
        index=False
    ))

    # =========================================================================
    # GÉNÉRATION DES 3 GRAPHIQUES CLÉS
    # =========================================================================
    print("\n--- Génération des graphiques ---")
    
    # --- Graphique 1: Le "Gap" (Le plus important) ---
    plt.figure(figsize=(10, 6))
    # Définir des couleurs sémantiques
    colors = ['#4CAF50', '#FFC107', '#D32F2F'] # Vert, Orange, Rouge
    
    # S'assurer que les couleurs correspondent à l'ordre trié
    ordered_instances = summary['instance_name']
    color_map = {name: color for name, color in zip(ordered_instances, colors)}
    
    bars = plt.bar(summary['instance_name'], summary['gap_moyen_pct'], color=[color_map[name] for name in summary['instance_name']])

    plt.title('Gap Moyen (%) par Instance (20 Runs)')
    plt.xlabel('Instances (Small, Medium, Large)')
    plt.ylabel('Gap Moyen (%) vs Optimal')
    plt.grid(True, linestyle="--", alpha=0.6, axis='y')

    # Ajouter les pourcentages au-dessus des barres
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2.0, yval + 1, f'{yval:.2f}%', ha='center', va='bottom')

    plt.savefig("benchmark_gap_analysis.png")
    print("Graphique 1/3 (Le plus important) : 'benchmark_gap_analysis.png' généré.")


    # --- Graphique 2: Boxplot Small & Medium ---
    plt.figure(figsize=(10, 6))
    small_medium_instances = ['C101', 'C1_2_1']
    all_data_sm = []
    labels_sm = []
    optimal_lines_sm = []

    for instance in small_medium_instances:
        if instance in df['instance_name'].values:
            all_data_sm.append(df[df['instance_name'] == instance]['final_cost'])
            labels_sm.append(instance)
            optimal_lines_sm.append(df[df['instance_name'] == instance]['optimal_cost'].iloc[0])

    if all_data_sm:
        bp_sm = plt.boxplot(all_data_sm, patch_artist=True, labels=labels_sm)
        for box in bp_sm['boxes']:
            box.set(facecolor="#87CEFA", alpha=0.7)
        
        plt.plot(range(1, len(labels_sm) + 1), optimal_lines_sm, 'r--o', label="Coût optimal", zorder=10)
        
        plt.title("Distribution des Coûts - Instances 'Small' et 'Medium' (20 Runs)")
        plt.xlabel("Instances")
        plt.ylabel("Coût total")
        plt.legend()
        plt.grid(True, linestyle="--", alpha=0.6)
        plt.tight_layout()
        plt.savefig("boxplot_small_medium_instances.png")
        print("Graphique 2/3 (Détail S/M) : 'boxplot_small_medium_instances.png' généré.")

    # --- Graphique 3: Boxplot Large ---
    plt.figure(figsize=(8, 6))
    instance_large = 'C1_10_2'
    if instance_large in df['instance_name'].values:
        instance_data = df[df['instance_name'] == instance_large]
        
        bp = plt.boxplot(instance_data['final_cost'], patch_artist=True, labels=[instance_large])
        for box in bp['boxes']:
            box.set(facecolor="#AED581", alpha=0.7) # Couleur verte pour "Large"
        
        optimal_cost = instance_data['optimal_cost'].iloc[0]
        plt.axhline(y=optimal_cost, color='r', linestyle='--', linewidth=2, label=f"Coût optimal ({optimal_cost:,.2f})")
        
        plt.title(f"Distribution des Coûts - Instance 'Large' (20 Runs)")
        plt.ylabel("Coût total")
        plt.legend()
        plt.grid(True, linestyle="--", alpha=0.6)
        plt.tight_layout()
        plt.savefig(f"boxplot_{instance_large}.png")
        print(f"Graphique 3/3 (Détail Large) : 'boxplot_{instance_large}.png' généré.")
    
    print("\n✅ Analyse terminée. Les graphiques (.png) sont sauvegardés à la racine du projet.")


if __name__ == "__main__":
    # 1. Vérifier les dépendances
    try:
        import pandas
        import matplotlib
    except ImportError as e:
        print(f"--- 🛑 DÉPENDANCE MANQUANTE : {e.name} ---", file=sys.stderr)
        print(f"Veuillez l'installer avec : pip install {e.name}", file=sys.stderr)
        sys.exit(1)
        
    # 2. Lancer la fonction principale
    main()