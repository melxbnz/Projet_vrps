import csv
import time
import sys
import traceback
from multiprocessing import Pool, cpu_count
from typing import List, Dict, Tuple

# --- Importation des modules de votre projet ---
# (Ces imports fonctionnent si run_benchmark.py est à la racine,
# à côté du dossier 'src')
try:
    from src.contracts import Instance, Solution
    from src.instance_loader import load_instance
    from src.initial_solution import build_clarke_wright_solution
    from src.optimization_loop import optimization_loop
    from src.evaluation import evaluate_solution
except ImportError as e:
    print(f"--- 🛑 ERREUR D'IMPORT CRITIQUE ---", file=sys.stderr)
    print(f"Impossible d'importer les modules depuis 'src'.", file=sys.stderr)
    print(f"Assurez-vous que ce script est bien à la racine du projet 'Projet_vrp'.", file=sys.stderr)
    print(f"Erreur détaillée: {e}", file=sys.stderr)
    sys.exit(1)

# --- Configuration du Benchmark (selon vos instructions) ---

# 1. Les trois instances cibles
INSTANCES_TO_RUN: List[str] = ["C101", "C1_2_1", "C1_10_2"]

# 2. Le nombre d'essais (runs) par instance
RUNS_PER_INSTANCE: int = 20

# 3. Le nombre de cœurs à utiliser en parallèle
CORES_TO_USE: int = 8

# 4. Le nom du fichier de résultats
OUTPUT_FILE: str = "benchmark_results_parallel.csv"

# 5. Paramètres pour l'optimiseur (tirés de votre main.py)
MAX_ITERATIONS: int = 1000
PATIENCE: int = 1000

# --- Fonction de Travail (Worker) ---

def worker_function(args_tuple: Tuple[str, int, int]) -> Dict | None:
    """
    Fonction exécutée par chaque processus du pool.
    Elle lance UNE seule optimisation (un run) pour une instance et un seed donnés.
    """
    instance_name, seed, run_index = args_tuple
    
    # Message de démarrage pour le suivi
    print(f"[Job Start] Instance: {instance_name}, Run: {run_index + 1}/{RUNS_PER_INSTANCE}, Seed: {seed}")
    start_time = time.time()
    
    try:
        # 1. Chargement de l'instance ET de sa solution optimale connue
        # (load_instance retourne les deux, cf. votre code )
        instance, optimal_sol = load_instance(instance_name)
        
        # 2. Construction de la solution initiale (Clarke & Wright) 
        initial_solution = build_clarke_wright_solution(instance)

        # 3. Lancement de la boucle d'optimisation (le pilote ALNS) [cite: 208, 251]
        history = optimization_loop(
            instance=instance,
            init_solution=initial_solution,
            max_iter=MAX_ITERATIONS,
            patience=PATIENCE,
            seed=seed
        )
        
        # 4. Récupération du résultat final
        final_cost = history["cost_best"][-1]
        end_time = time.time()
        exec_time = end_time - start_time
        
        print(f"[Job Done]  Instance: {instance_name}, Run: {run_index + 1}/{RUNS_PER_INSTANCE}. "
              f"Coût: {final_cost:.2f} (Optimal: {optimal_sol.cost:.2f}) (Temps: {exec_time:.2f}s)")
        
        # 5. Retourner un dictionnaire propre pour le CSV
        return {
            "instance_name": instance_name,
            "seed": seed,
            "run_index": run_index + 1,
            "final_cost": final_cost,
            "optimal_cost": optimal_sol.cost,
            "execution_time": exec_time,
        }

    except Exception as e:
        # Gérer les erreurs sans planter tout le benchmark
        print(f"--- 🛑 ERREUR [Job Fail] Instance: {instance_name}, Run: {run_index + 1} ---", file=sys.stderr)
        print(f"Erreur: {e}", file=sys.stderr)
        traceback.print_exc(file=sys.stderr) # Affiche les détails complets de l'erreur
        return None

# --- Fonction Principale (Main) ---

def main():
    """
    Orchestre la création des tâches, le pool de multiprocessing et l'écriture du CSV.
    """
    
    # Vérifier le nombre de cœurs disponibles
    actual_cores = cpu_count()
    cores_to_use = CORES_TO_USE
    if actual_cores < cores_to_use:
        print(f"Attention: Moins de {cores_to_use} cœurs détectés (disponibles: {actual_cores}).")
        print(f"Utilisation de {actual_cores} cœurs pour éviter une surcharge.")
        cores_to_use = actual_cores

    # Créer la liste de toutes les tâches à exécuter
    # (3 instances * 20 runs = 60 tâches)
    jobs = []
    for instance_name in INSTANCES_TO_RUN:
        for i in range(RUNS_PER_INSTANCE):
            seed = i  # Utiliser un seed différent (0 à 19) pour chaque run
            jobs.append((instance_name, seed, i))
            
    print("--- 🚀 Lancement du Benchmark Parallèle ---")
    print(f"  Configuration :")
    print(f"  Instances     : {INSTANCES_TO_RUN}")
    print(f"  Runs/Instance : {RUNS_PER_INSTANCE}")
    print(f"  Total Tâches  : {len(jobs)}")
    print(f"  Cœurs (Workers) : {cores_to_use}")
    print(f"  Fichier Sortie: {OUTPUT_FILE}")
    print("-------------------------------------------------")
    
    start_total_time = time.time()
    
    # Créer et lancer le pool de workers
    # 'with' s'assure que le pool est correctement fermé
    with Pool(processes=cores_to_use) as pool:
        # pool.map exécute 'worker_function' pour chaque item dans 'jobs'
        # et collecte les résultats dans le même ordre.
        results = pool.map(worker_function, jobs)
    
    end_total_time = time.time()
    print("-------------------------------------------------")
    print(f"✅ Benchmark terminé. Temps total: {end_total_time - start_total_time:.2f} secondes.")
    
    # --- Écriture des résultats dans le fichier CSV ---
    
    # Filtrer les jobs qui ont échoué (ceux qui ont retourné None)
    successful_results = [r for r in results if r is not None]
    
    if not successful_results:
        print("Aucun résultat n'a été généré. Le fichier CSV est vide.")
        return

    # Définir les en-têtes du CSV
    headers = ["instance_name", "seed", "run_index", "final_cost", "optimal_cost", "execution_time"]
    
    print(f"Écriture de {len(successful_results)} résultats dans {OUTPUT_FILE}...")
    
    try:
        with open(OUTPUT_FILE, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=headers)
            writer.writeheader()
            writer.writerows(successful_results)
        
        print(f"Succès. Fichier '{OUTPUT_FILE}' généré.")
        if len(successful_results) != len(jobs):
            print(f"Attention: {len(jobs) - len(successful_results)} runs ont échoué (voir logs d'erreur).")
            
    except IOError as e:
        print(f"--- 🛑 ERREUR lors de l'écriture du CSV ---", file=sys.stderr)
        print(f"Erreur: {e}", file=sys.stderr)
    except Exception as e:
        print(f"--- 🛑 ERREUR Inconnue lors de l'écriture du CSV ---", file=sys.stderr)
        print(f"Erreur: {e}", file=sys.stderr)


# Point d'entrée standard pour les scripts Python
if __name__ == "__main__":
    # Cette vérification est OBLIGATOIRE pour que multiprocessing fonctionne
    # correctement sous Windows.
    main()