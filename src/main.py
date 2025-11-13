#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Module principal (Étape 8) - Le Chef d'Orchestre.
[VERSION ADAPTÉE pour le loader vrplib]
"""

import sys
import time
from typing import Dict, List

# --- Importation des modules de notre projet ---
try:
    # 1. Le "Contrat"
    from .contracts import Instance, Solution
    
    # 2. Le "Chargeur" (MODIFIÉ)
    from .instance_loader import load_instance 
    
    # 3. L'"Initialiseur"
    from .initial_solution import build_clarke_wright_solution
    
    # 5. Le "Pilote"
    from .optimization_loop import optimization_loop

except ImportError as e:
    print(f"--- 🛑 ERREUR D'IMPORT CRITIQUE ---", file=sys.stderr)
    print(f"Erreur d'import dans main.py: {e}", file=sys.stderr)
    sys.exit(1)

# --- Configuration de la simulation ---

INSTANCE_NAME = "C1_10_2" # 1000 clients

# Paramètres pour la boucle d'optimisation
MAX_ITERATIONS = 1000
PATIENCE = 1000
SEED = 42

# --- Fonction Principale (ADAPTÉE) ---

def run_optimization():
    """
    Exécute le processus complet de chargement et d'optimisation.
    """
    print(f"--- [Projet VRP] Lancement de l'optimisation sur '{INSTANCE_NAME}' ---")
    
    print("\n--- [1. CHARGEMENT (vrplib)] ---")
    try:
        instance, optimal_sol = load_instance(INSTANCE_NAME)
        
        print(f"Instance '{instance.name}' chargée ({len(instance.demand)-1} clients).")
        print(f"  -> Coût optimal (lu du .sol): {optimal_sol.cost}")

    except FileNotFoundError:
        print(f"❌ Échec: Fichier non trouvé. Avez-vous corrigé le chemin '../data/' en 'data/' dans instance_loader.py ?")
        return
    except Exception as e:
        print(f"❌ Échec du chargement de l'instance: {e}")
        return

    print("\n--- [2. SOLUTION INITIALE (Clarke & Wright)] ---")
    try:
        initial_solution = build_clarke_wright_solution(instance)
        print(f"Solution initiale (C&W) générée.")
        print(f"  -> Nb routes: {len(initial_solution.routes)}")
        print(f"  -> Coût initial: {initial_solution.cost:.2f}")
        print(f"  -> Faisable: {initial_solution.feasible}")
    except Exception as e:
        print(f"❌ Échec de la génération de la solution initiale: {e}")
        return

    print("\n--- [3. OPTIMISATION (ALNS)] ---")
    print(f"Lancement de la boucle (Max iters: {MAX_ITERATIONS}, Patience: {PATIENCE})...")
    
    start_time = time.time()
    
    try:
        # On lance le "Pilote"
        history = optimization_loop(
            instance,
            initial_solution,
            max_iter=MAX_ITERATIONS,
            patience=PATIENCE,
            seed=SEED
        )
    except Exception as e:
        print("\n--- 🛑 ERREUR FATALE LORS DE L'OPTIMISATION ---")
        print(f"Erreur détaillée: {e}")
        return

    end_time = time.time()
    print(f"Optimisation terminée en {end_time - start_time:.2f} secondes.")

    print("\n--- [4. RÉSULTATS] ---")
    if not history["cost_best"]:
        print("Aucun historique n'a été généré.")
        return
        
    final_best_cost = history["cost_best"][-1]
    print(f"Coût C&W    : {initial_solution.cost:.2f}")
    print(f"Meilleur coût: {final_best_cost:.2f}")
    print(f"Coût optimal : {optimal_sol.cost:.2f}")
    
    improvement = (initial_solution.cost - final_best_cost) / initial_solution.cost * 100
    print(f"Amélioration : {improvement:.2f}%")

# --- Point d'entrée du script ---
if __name__ == "__main__":
    """
    Exécutable via : python -m src.main
    """
    run_optimization()