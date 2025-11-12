#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Module pour le chargement des instances (Étape 2).
Lit les fichiers .txt format Solomon et retourne un objet Instance.
"""

import math
import sys
from typing import List, Tuple, Dict, Optional

# --- Importe le "Contrat" ---
try:
    from .contracts import Instance
except ImportError:
    # ... (Stubs de fallback pour les tests) ...
    print("Erreur: Impossible d'importer contracts dans instance_loader.py", file=sys.stderr)
    from dataclasses import dataclass, field
    @dataclass
    class Instance:
        name: str
        distance_matrix: List[List[float]]
        demand: List[int]
        capacity: int
        ready_time: Optional[List[float]] = None
        due_time: Optional[List[float]] = None
        service_time: Optional[List[float]] = None
        Kmax: Optional[int] = None
        Tmax: Optional[float] = None


# --- Fonctions Helpers ---

def _calculate_distance_matrix(coords: List[Tuple[float, float]]) -> List[List[float]]:
    """Calcule une matrice de distance euclidienne n x n."""
    n = len(coords)
    matrix = [[0.0] * n for _ in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            dist = math.sqrt(
                (coords[i][0] - coords[j][0])**2 +
                (coords[i][1] - coords[j][1])**2
            )
            matrix[i][j] = dist
            matrix[j][i] = dist
    return matrix

# --- Fonction Principale ---

def load_solomon_instance(filepath: str) -> Instance:
    """
    Charge une instance au format Solomon (.txt) et la convertit
    en notre dataclass 'Instance'.
    """
    instance_name = ""
    capacity = 0
    kmax = None
    
    coords = []
    demands = []
    ready_times = []
    due_times = []
    service_times = []

    try:
        with open(filepath, 'r') as f:
            lines = f.readlines()

        # --- Parsing ---
        instance_name = lines[0].strip()

        # --- [BLOC CORRIGÉ] ---
        # 2. Capacité du véhicule
        # On cherche la ligne "VEHICLE"
        parsing_vehicle_header = False
        for i, line in enumerate(lines):
            line_s = line.strip()
            
            if line_s.startswith("VEHICLE"):
                parsing_vehicle_header = True
                continue
                
            if parsing_vehicle_header:
                # La ligne suivante doit être "NUMBER CAPACITY" (on l'ignore)
                if line_s.startswith("NUMBER") and "CAPACITY" in line_s:
                    continue
                
                # La ligne suivante contient les données
                parts = line_s.split()
                if len(parts) >= 2:
                    try:
                        kmax = int(float(parts[0]))
                        capacity = int(float(parts[1]))
                        break # On a trouvé, on arrête
                    except (ValueError, IndexError):
                        continue # Ligne vide ou mal formée
                elif line_s:
                    break # Fin de la section
        # --- [FIN BLOC CORRIGÉ] ---

        # 3. Données clients
        parsing_customers = False
        for line in lines:
            if line.strip().startswith("CUST NO."):
                parsing_customers = True
                continue
            if not parsing_customers:
                continue
            
            parts = line.strip().split()
            if not parts: continue

            try:
                coords.append((float(parts[1]), float(parts[2])))
                demands.append(int(float(parts[3])))
                ready_times.append(float(parts[4]))
                due_times.append(float(parts[5]))
                service_times.append(float(parts[6]))
            except (IndexError, ValueError):
                continue

        if not coords:
            raise ValueError(f"Aucune donnée client trouvée dans {filepath}")

        # --- Post-traitement ---
        distance_matrix = _calculate_distance_matrix(coords)
        tmax = due_times[0] if due_times else None

        instance_obj = Instance(
            name=instance_name,
            distance_matrix=distance_matrix,
            demand=demands,
            capacity=capacity,
            ready_time=ready_times,
            due_time=due_times,
            service_time=service_times,
            Kmax=kmax,
            Tmax=tmax
        )
        
        return instance_obj

    except FileNotFoundError:
        print(f"Erreur: Fichier instance non trouvé à '{filepath}'", file=sys.stderr)
        raise
    except Exception as e:
        print(f"Erreur lors du parsing de l'instance '{filepath}': {e}", file=sys.stderr)
        raise


# --- Tests Rapides (Quick Check) ---

if __name__ == "__main__":
    """
    Test exécutable via : python -m src.instance_loader
    (Doit être lancé depuis la racine du projet)
    """
    print("🚀 Lancement des tests rapides pour src/instance_loader.py...")
    
    test_filepath = "data/C101.txt"

    try:
        instance = load_solomon_instance(test_filepath)
        
        print(f"\nInstance '{instance.name}' chargée avec succès.")
        assert instance.name == "C101"
        
        num_nodes = len(instance.demand)
        print(f"Nombre de nœuds: {num_nodes} (1 dépôt + 100 clients)")
        assert num_nodes == 101
        
        print(f"Capacité du véhicule: {instance.capacity}")
        assert instance.capacity == 200 # Le test va maintenant passer
        
        print(f"Kmax (Nb véhicules): {instance.Kmax}")
        assert instance.Kmax == 25
        
        dist_0_1 = instance.distance_matrix[0][1]
        print(f"Distance Dépôt -> Client 1: {dist_0_1:.4f}")
        assert math.isclose(dist_0_1, 18.6815417, rel_tol=1e-5)
        
        print("\n🎉 Tous les tests de chargement ont réussi!")
        
    except FileNotFoundError:
        print(f"\n❌ Échec du test: Fichier de test '{test_filepath}' non trouvé.")
        print("Assurez-vous de lancer ce test depuis la racine du projet (Projet_vrp/) avec :")
        print("python -m src.instance_loader")
    except AssertionError as e:
        print(f"\n❌ Échec du test (AssertionError): {e}")
        print("Les données lues ne correspondent pas aux attentes (ex: Capacité 200).")
    except Exception as e:
        print(f"\n❌ Échec du test lors du chargement: {e}")