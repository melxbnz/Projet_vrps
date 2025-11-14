import sys
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os

# --- Configuration pour des graphiques clairs ---
plt.style.use('ggplot')
plt.rcParams['figure.figsize'] = (12, 7) # Taille par défaut pour les graphiques
plt.rcParams['savefig.dpi'] = 100 # Bonne résolution pour un rapport

def load_results_from_csv(csv_file: str) -> pd.DataFrame:
    """
    Charge les résultats depuis le CSV généré par le benchmark.
    """
    try:
        # On s'attend à ce que le CSV soit dans le même dossier que ce script
        print(f"--- Chargement des résultats depuis '{csv_file}' ---")
        df = pd.read_csv(csv_file)
        return df
        
    except FileNotFoundError:
        print(f"--- 🛑 ERREUR ---", file=sys.stderr)
        print(f"Fichier '{csv_file}' non trouvé.", file=sys.stderr)
        print("Veuillez vérifier que ce script est bien à la racine du projet,", file=sys.stderr)
        print("à côté de 'benchmark_results_parallel.csv'.", file=sys.stderr)
        return None
    except Exception as e:
        print(f"Erreur lors de la lecture du CSV: {e}", file=sys.stderr)
        return None

def generate_gap_plots_per_instance(df: pd.DataFrame):
    """
    Génère un graphique d'analyse du GAP (par run) pour CHAQUE instance.
    """
    if df is None:
        print("Aucune donnée à analyser.")
        return

    # 1. Calculer le 'gap' (écart en %)
    df['gap'] = 100 * (df['final_cost'] - df['optimal_cost']) / df['optimal_cost']

    # 2. Obtenir la liste des instances uniques
    instances = df['instance_name'].unique()
    critere_gap = 7.0

    print(f"Instances trouvées : {instances}")
    print(f"Critère de Gap appliqué : {critere_gap:.1f}%")

    # 3. Boucler sur chaque instance pour créer un graphique
    for instance in instances:
        
        # Isoler les données pour cette instance
        df_instance = df[df['instance_name'] == instance].copy()
        df_instance = df_instance.sort_values(by='run_index')
        
        # Compter les succès et échecs
        success_count = (df_instance['gap'] < critere_gap).sum()
        total_count = len(df_instance['gap'])
        
        print(f"\nAnalyse de l'instance: {instance}")
        print(f"  -> Runs réussis (< {critere_gap}%): {success_count} / {total_count}")
        
        # 4. Assigner les couleurs en fonction du critère
        # Vert si Gap < 7%, Rouge sinon
        colors = ['#4CAF50' if g < critere_gap else '#D32F2F' for g in df_instance['gap']]

        # 5. Générer le graphique
        plt.figure(figsize=(12, 7))
        bars = plt.bar(df_instance['run_index'], df_instance['gap'], color=colors)

        # Ajouter la ligne de critère (la limite de 7%)
        plt.axhline(y=critere_gap, color='blue', linestyle='--', linewidth=2, 
                    label=f"Critère du Gap ({critere_gap:.0f}%)")

        # Ajouter les valeurs de gap sur les barres
        for bar in bars:
            yval = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2.0, yval + 0.2, f'{yval:.2f}%', 
                     ha='center', va='bottom', fontsize=8, rotation=90)

        # 6. Mettre en forme le graphique
        plt.title(f"Performance des 20 Runs - Instance: {instance} (Réussite: {success_count}/{total_count})")
        plt.xlabel("Numéro de l'Essai (Run Index)")
        plt.ylabel("Gap (%) par rapport à l'Optimal")
        plt.xticks(df_instance['run_index'])
        plt.legend()
        plt.grid(True, linestyle="--", alpha=0.6, axis='y')
        
        # Ajuster la limite Y pour laisser de la place aux étiquettes
        # (Prend le max entre le gap le plus haut et le critère)
        plt.ylim(top=max(df_instance['gap'].max() * 1.1, critere_gap * 1.2)) 
        
        plt.tight_layout()

        # 7. Sauvegarder le graphique
        output_filename = f"benchmark_gap_per_run_{instance}.png"
        plt.savefig(output_filename)
        print(f"  -> Graphique '{output_filename}' généré.")

def main():
    """
    Fonction principale:
    1. Vérifie les dépendances
    2. Charge les données
    3. Lance la génération des graphiques
    """
    
    # 1. Vérifier les dépendances
    try:
        import pandas
        import matplotlib
    except ImportError as e:
        print(f"--- 🛑 DÉPENDANCE MANQUANTE : {e.name} ---", file=sys.stderr)
        print(f"Veuillez l'installer avec : pip install {e.name}", file=sys.stderr)
        sys.exit(1)

    # 2. Nom du fichier CSV
    CSV_FILENAME = "benchmark_results_parallel.csv"

    # 3. Charger les données
    df = load_results_from_csv(CSV_FILENAME)
    
    if df is None:
        sys.exit(1) # Arrêter si le fichier n'a pas été trouvé

    # 4. Générer les graphiques
    generate_gap_plots_per_instance(df)
    
    print("\n✅ Analyse terminée. Les 3 graphiques (.png) sont sauvegardés à la racine du projet.")


if __name__ == "__main__":
    main()